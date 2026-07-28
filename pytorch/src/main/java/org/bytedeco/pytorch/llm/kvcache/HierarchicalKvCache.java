package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Two-tier hierarchical KV cache inspired by TensorRT-LLM / vLLM offloading.
 *
 * <ul>
 *   <li><b>Hot tier</b>: device-resident {@link PagedBlockManager} (fast).</li>
 *   <li><b>Cold tier</b>: host-resident {@link PagedBlockManager} (larger, slower).</li>
 * </ul>
 *
 * <p>Sequences live on the hot tier while actively decoding. On pressure, the
 * oldest idle sequence's blocks are copied hot→cold and hot refs released.
 * On reuse, cold→hot promotion restores device residency.
 *
 * <p>This is a control-plane + data-plane sketch suitable for JavaCPP bindings:
 * real async DMA would plug into {@link #copyBlock(PagedBlockManager, int, PagedBlockManager, int)}.
 */
public class HierarchicalKvCache implements AutoCloseable {

    private final PagedBlockManager hot;
    private final PagedBlockManager cold;
    private final ReentrantLock lock = new ReentrantLock();
    private final Map<Long, SeqState> sequences = new HashMap<>();
    private long nextSeqId = 1L;
    private boolean closed = false;

    public final LongAdder promoteCount = new LongAdder();
    public final LongAdder demoteCount = new LongAdder();
    public final LongAdder appendCount = new LongAdder();

    public HierarchicalKvCache(int hotBlocks, int coldBlocks,
                               int numLayers, int blockSize, int numHeads, int headDim) {
        this(hotBlocks, coldBlocks, numLayers, blockSize, numHeads, headDim, null, null);
    }

    public HierarchicalKvCache(int hotBlocks, int coldBlocks,
                               int numLayers, int blockSize, int numHeads, int headDim,
                               Device hotDevice, Device coldDevice) {
        if (hotBlocks <= 0 || coldBlocks <= 0) {
            throw new IllegalArgumentException("hot/cold block counts must be > 0");
        }
        this.hot = new PagedBlockManager(hotBlocks, numLayers, blockSize, numHeads, headDim,
                hotDevice, torch.kFloat(), 0.10, 0.25);
        // Cold defaults to CPU even if hot is on GPU
        Device coldDev = coldDevice != null ? coldDevice : new Device("cpu");
        this.cold = new PagedBlockManager(coldBlocks, numLayers, blockSize, numHeads, headDim,
                coldDev, torch.kFloat(), 0.05, 0.15);
        this.nextSeqId = 1L;
        this.closed = false;
    }

    public PagedBlockManager hot() { return hot; }
    public PagedBlockManager cold() { return cold; }
    public int numLayers() { return hot.numLayers(); }
    public int blockSize() { return hot.blockSize(); }
    public int numHeads() { return hot.numHeads(); }
    public int headDim() { return hot.headDim(); }

    public int liveSequences() {
        lock.lock();
        try { return sequences.size(); }
        finally { lock.unlock(); }
    }

    public long createSequence() {
        lock.lock();
        try {
            ensureOpen();
            long id = nextSeqId++;
            sequences.put(id, new SeqState());
            return id;
        } finally { lock.unlock(); }
    }

    public void releaseSequence(long seqId) {
        lock.lock();
        try {
            SeqState st = sequences.remove(seqId);
            if (st == null) return;
            releaseState(st);
        } finally { lock.unlock(); }
    }

    /**
     * Append one token for all layers onto the hot tier.
     * May demote other sequences if hot is full.
     */
    public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
        Objects.requireNonNull(kLayers);
        Objects.requireNonNull(vLayers);
        if (kLayers.length != hot.numLayers() || vLayers.length != hot.numLayers()) {
            throw new IllegalArgumentException("layer count mismatch");
        }
        lock.lock();
        try {
            ensureOpen();
            SeqState st = require(seqId);
            ensureHot(st);
            int pos = st.length % hot.blockSize();
            if (pos == 0) {
                int b = allocHotWithDemote(seqId);
                st.hotBlocks.add(b);
            }
            int blockId = st.hotBlocks.get(st.hotBlocks.size() - 1);
            // CoW if somehow shared (shouldn't be for exclusive seq, but safe)
            int exclusive = hot.cowIfNeeded(blockId);
            if (exclusive != blockId) {
                st.hotBlocks.set(st.hotBlocks.size() - 1, exclusive);
                blockId = exclusive;
            }
            for (int layer = 0; layer < hot.numLayers(); layer++) {
                hot.writeToken(blockId, layer, pos, kLayers[layer], vLayers[layer]);
            }
            st.length++;
            st.lastUseNs = System.nanoTime();
            appendCount.increment();
        } finally { lock.unlock(); }
    }

    public Tensor[] gather(long seqId, int layer) {
        lock.lock();
        try {
            ensureOpen();
            SeqState st = require(seqId);
            ensureHot(st);
            return hot.gather(st.hotBlocks, layer, st.length);
        } finally { lock.unlock(); }
    }

    public int sequenceLength(long seqId) {
        lock.lock();
        try { return require(seqId).length; }
        finally { lock.unlock(); }
    }

    /** Force demote of a specific sequence to cold tier. */
    public void demote(long seqId) {
        lock.lock();
        try {
            ensureOpen();
            demoteUnlocked(require(seqId));
        } finally { lock.unlock(); }
    }

    /** Force promote of a specific sequence to hot tier. */
    public void promote(long seqId) {
        lock.lock();
        try {
            ensureOpen();
            promoteUnlocked(require(seqId));
        } finally { lock.unlock(); }
    }

    public boolean isHot(long seqId) {
        lock.lock();
        try { return require(seqId).onHot; }
        finally { lock.unlock(); }
    }

    @Override
    public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            for (SeqState st : sequences.values()) releaseState(st);
            sequences.clear();
            hot.close();
            cold.close();
        } finally { lock.unlock(); }
    }

    // ---- internals --------------------------------------------------------

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("HierarchicalKvCache closed");
    }

    private SeqState require(long id) {
        SeqState st = sequences.get(id);
        if (st == null) throw new IllegalArgumentException("unknown sequence " + id);
        return st;
    }

    private void ensureHot(SeqState st) {
        if (!st.onHot) promoteUnlocked(st);
        st.lastUseNs = System.nanoTime();
    }

    private int allocHotWithDemote(long excludeSeqId) {
        try {
            return hot.allocateBlock();
        } catch (IllegalStateException oom) {
            if (!demoteLruExcept(excludeSeqId)) {
                throw new IllegalStateException("hot tier OOM and no demotable sequence", oom);
            }
            return hot.allocateBlock();
        }
    }

    private boolean demoteLruExcept(long excludeSeqId) {
        SeqState victim = null;
        long best = Long.MAX_VALUE;
        Long victimId = null;
        for (Map.Entry<Long, SeqState> e : sequences.entrySet()) {
            if (e.getKey() == excludeSeqId) continue;
            SeqState st = e.getValue();
            if (!st.onHot || st.hotBlocks.isEmpty()) continue;
            if (st.lastUseNs < best) {
                best = st.lastUseNs;
                victim = st;
                victimId = e.getKey();
            }
        }
        if (victim == null) return false;
        demoteUnlocked(victim);
        return true;
    }

    private void demoteUnlocked(SeqState st) {
        if (!st.onHot) return;
        List<Integer> coldIds = new ArrayList<>(st.hotBlocks.size());
        for (int hb : st.hotBlocks) {
            int cb = cold.allocateBlock();
            copyBlock(hot, hb, cold, cb);
            coldIds.add(cb);
            hot.release(hb);
        }
        st.hotBlocks.clear();
        st.coldBlocks.clear();
        st.coldBlocks.addAll(coldIds);
        st.onHot = false;
        demoteCount.increment();
    }

    private void promoteUnlocked(SeqState st) {
        if (st.onHot) return;
        List<Integer> hotIds = new ArrayList<>(st.coldBlocks.size());
        for (int cb : st.coldBlocks) {
            int hb;
            try {
                hb = hot.allocateBlock();
            } catch (IllegalStateException oom) {
                // demote someone else then retry
                if (!demoteLruExcept(-1)) {
                    // roll back partial
                    for (int h : hotIds) hot.release(h);
                    throw new IllegalStateException("hot tier OOM during promote", oom);
                }
                hb = hot.allocateBlock();
            }
            copyBlock(cold, cb, hot, hb);
            hotIds.add(hb);
            cold.release(cb);
        }
        st.coldBlocks.clear();
        st.hotBlocks.clear();
        st.hotBlocks.addAll(hotIds);
        st.onHot = true;
        st.lastUseNs = System.nanoTime();
        promoteCount.increment();
    }

    private static void copyBlock(PagedBlockManager src, int srcId,
                                  PagedBlockManager dst, int dstId) {
        src.ensureAllocated();
        dst.ensureAllocated();
        dst.getBlock(dstId).copy_(src.getBlock(srcId));
    }

    private void releaseState(SeqState st) {
        if (st.onHot) {
            for (int b : st.hotBlocks) hot.release(b);
        } else {
            for (int b : st.coldBlocks) cold.release(b);
        }
        st.hotBlocks.clear();
        st.coldBlocks.clear();
    }

    private static final class SeqState {
        final List<Integer> hotBlocks = new ArrayList<>();
        final List<Integer> coldBlocks = new ArrayList<>();
        boolean onHot = true;
        int length = 0;
        long lastUseNs = System.nanoTime();
    }

    @Override
    public String toString() {
        return "HierarchicalKvCache{hotFree=" + hot.freeBlocks()
                + "/" + hot.maxBlocks()
                + ", coldFree=" + cold.freeBlocks()
                + "/" + cold.maxBlocks()
                + ", seqs=" + liveSequences() + "}";
    }
}
