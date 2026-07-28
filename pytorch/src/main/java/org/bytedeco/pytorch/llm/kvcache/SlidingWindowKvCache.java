package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Sliding-window / sink KV cache (Mistral SWA, StreamingLLM-style attention sink).
 *
 * <p>Each sequence keeps:
 * <ul>
 *   <li>{@code sinkTokens} leading tokens permanently (attention sink).</li>
 *   <li>A trailing ring of at most {@code windowTokens} recent tokens, with a
 *       {@code windowHead} offset into the first window block so drops are O(1).</li>
 * </ul>
 *
 * <p>{@link #gather} returns the concatenated {@code sink || window} stream
 * (no middle gap) — what SWA attention expects.
 *
 * <p>Built on {@link PagedBlockManager}.
 */
public class SlidingWindowKvCache implements AutoCloseable {

    private final PagedBlockManager pool;
    private final int sinkTokens;
    private final int windowTokens;
    private final ReentrantLock lock = new ReentrantLock();
    private final Map<Long, Seq> sequences = new HashMap<>();
    private long nextSeqId = 1L;
    private boolean closed = false;

    public final LongAdder appendCount = new LongAdder();
    public final LongAdder dropTokens = new LongAdder();
    public final LongAdder reclaimBlocks = new LongAdder();

    public SlidingWindowKvCache(int maxBlocks, int numLayers, int blockSize,
                                int numHeads, int headDim,
                                int sinkTokens, int windowTokens) {
        this(maxBlocks, numLayers, blockSize, numHeads, headDim, sinkTokens, windowTokens, null);
    }

    public SlidingWindowKvCache(int maxBlocks, int numLayers, int blockSize,
                                int numHeads, int headDim,
                                int sinkTokens, int windowTokens, Device device) {
        if (sinkTokens < 0 || windowTokens <= 0) {
            throw new IllegalArgumentException("sinkTokens >= 0 and windowTokens > 0 required");
        }
        this.pool = new PagedBlockManager(maxBlocks, numLayers, blockSize, numHeads, headDim,
                device, torch.kFloat(), 0.10, 0.20);
        this.sinkTokens = sinkTokens;
        this.windowTokens = windowTokens;
        this.nextSeqId = 1L;
        this.closed = false;
    }

    public int sinkTokens() { return sinkTokens; }
    public int windowTokens() { return windowTokens; }
    public int blockSize() { return pool.blockSize(); }
    public int numLayers() { return pool.numLayers(); }
    public int freeBlocks() { return pool.freeBlocks(); }
    public PagedBlockManager pool() { return pool; }

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
            sequences.put(id, new Seq());
            return id;
        } finally { lock.unlock(); }
    }

    public void releaseSequence(long seqId) {
        lock.lock();
        try {
            Seq st = sequences.remove(seqId);
            if (st == null) return;
            releaseSeq(st);
        } finally { lock.unlock(); }
    }

    /**
     * Append one token for all layers. Once past the sink, tokens enter the
     * window ring; when the ring is full the oldest window token is dropped.
     */
    public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
        Objects.requireNonNull(kLayers);
        Objects.requireNonNull(vLayers);
        if (kLayers.length != pool.numLayers() || vLayers.length != pool.numLayers()) {
            throw new IllegalArgumentException("layer count mismatch");
        }
        lock.lock();
        try {
            ensureOpen();
            Seq st = require(seqId);
            int B = pool.blockSize();

            if (st.length < sinkTokens) {
                int local = st.sinkLen;
                int needBlocks = (local + 1 + B - 1) / B;
                while (st.sinkBlocks.size() < needBlocks) {
                    st.sinkBlocks.add(pool.allocateBlock());
                }
                int blockId = st.sinkBlocks.get(local / B);
                writeAll(blockId, local % B, kLayers, vLayers);
                st.sinkLen++;
            } else {
                // Evict from window head until there is room for one more token.
                while (st.windowLen >= windowTokens) {
                    dropOldestWindowToken(st);
                }
                // Physical write position in the window ring storage:
                // tokens occupy [windowHead, windowHead + windowLen).
                int phys = st.windowHead + st.windowLen;
                int blockIdx = phys / B;
                int pos = phys % B;
                while (st.windowBlocks.size() <= blockIdx) {
                    st.windowBlocks.addLast(pool.allocateBlock());
                }
                // Deque has no random access — materialize by index
                int blockId = windowBlockAt(st, blockIdx);
                writeAll(blockId, pos, kLayers, vLayers);
                st.windowLen++;
            }

            st.length++;
            appendCount.increment();
        } finally { lock.unlock(); }
    }

    /**
     * Gather concatenated {@code sink || window} for one layer:
     * {@code {K,V}} each {@code [retainedLength, H, D]}.
     */
    public Tensor[] gather(long seqId, int layer) {
        lock.lock();
        try {
            ensureOpen();
            Seq st = require(seqId);
            if (layer < 0 || layer >= pool.numLayers()) {
                throw new IllegalArgumentException("layer");
            }
            int retained = st.sinkLen + st.windowLen;
            if (retained == 0) {
                Tensor ek = torch.zeros(new long[]{0, pool.numHeads(), pool.headDim()}, pool.options());
                Tensor ev = torch.zeros(new long[]{0, pool.numHeads(), pool.headDim()}, pool.options());
                return new Tensor[]{ek, ev};
            }

            List<Tensor> ks = new ArrayList<>();
            List<Tensor> vs = new ArrayList<>();
            gatherRange(ks, vs, st.sinkBlocks, layer, 0, st.sinkLen);
            gatherWindow(ks, vs, st, layer);

            Tensor k = torch.cat(new TensorVector(ks.toArray(new Tensor[0])), 0);
            Tensor v = torch.cat(new TensorVector(vs.toArray(new Tensor[0])), 0);
            return new Tensor[]{k, v};
        } finally { lock.unlock(); }
    }

    public int sequenceLength(long seqId) {
        lock.lock();
        try { return require(seqId).length; }
        finally { lock.unlock(); }
    }

    /** Tokens still resident (sink + window). */
    public int retainedLength(long seqId) {
        lock.lock();
        try {
            Seq st = require(seqId);
            return st.sinkLen + st.windowLen;
        } finally { lock.unlock(); }
    }

    public int maxRetainedTokens() {
        return sinkTokens + windowTokens;
    }

    @Override
    public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            for (Seq st : sequences.values()) releaseSeq(st);
            sequences.clear();
            pool.close();
        } finally { lock.unlock(); }
    }

    // ---- internals --------------------------------------------------------

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("SlidingWindowKvCache closed");
    }

    private Seq require(long id) {
        Seq st = sequences.get(id);
        if (st == null) throw new IllegalArgumentException("unknown sequence " + id);
        return st;
    }

    private void releaseSeq(Seq st) {
        for (int b : st.sinkBlocks) pool.release(b);
        for (int b : st.windowBlocks) pool.release(b);
        st.sinkBlocks.clear();
        st.windowBlocks.clear();
        st.sinkLen = 0;
        st.windowLen = 0;
        st.windowHead = 0;
    }

    private void writeAll(int blockId, int pos, Tensor[] kLayers, Tensor[] vLayers) {
        for (int layer = 0; layer < pool.numLayers(); layer++) {
            pool.writeToken(blockId, layer, pos, kLayers[layer], vLayers[layer]);
        }
    }

    private void dropOldestWindowToken(Seq st) {
        if (st.windowLen <= 0) return;
        int B = pool.blockSize();
        st.windowHead++;
        st.windowLen--;
        dropTokens.increment();
        if (st.windowHead >= B && !st.windowBlocks.isEmpty()) {
            int old = st.windowBlocks.removeFirst();
            pool.release(old);
            reclaimBlocks.increment();
            st.windowHead -= B;
        }
    }

    private static int windowBlockAt(Seq st, int index) {
        int i = 0;
        for (int b : st.windowBlocks) {
            if (i == index) return b;
            i++;
        }
        throw new IllegalStateException("window block index out of range: " + index);
    }

    /** Gather {@code length} tokens starting at offset 0 from a plain block list. */
    private void gatherRange(List<Tensor> ks, List<Tensor> vs,
                             List<Integer> blocks, int layer, int offset, int length) {
        if (length <= 0 || blocks.isEmpty()) return;
        int B = pool.blockSize();
        int remaining = length;
        int pos = offset;
        int blockIdx = pos / B;
        int from = pos % B;
        while (remaining > 0 && blockIdx < blocks.size()) {
            int take = Math.min(B - from, remaining);
            Tensor slot = pool.getBlockLayer(blocks.get(blockIdx), layer);
            ks.add(slot.select(0, 0).narrow(0, from, take));
            vs.add(slot.select(0, 1).narrow(0, from, take));
            remaining -= take;
            blockIdx++;
            from = 0;
        }
    }

    private void gatherWindow(List<Tensor> ks, List<Tensor> vs, Seq st, int layer) {
        if (st.windowLen <= 0 || st.windowBlocks.isEmpty()) return;
        int B = pool.blockSize();
        int remaining = st.windowLen;
        int phys = st.windowHead;
        List<Integer> blocks = new ArrayList<>(st.windowBlocks);
        while (remaining > 0) {
            int blockIdx = phys / B;
            int from = phys % B;
            if (blockIdx >= blocks.size()) break;
            int take = Math.min(B - from, remaining);
            Tensor slot = pool.getBlockLayer(blocks.get(blockIdx), layer);
            ks.add(slot.select(0, 0).narrow(0, from, take));
            vs.add(slot.select(0, 1).narrow(0, from, take));
            remaining -= take;
            phys += take;
        }
    }

    private static final class Seq {
        final List<Integer> sinkBlocks = new ArrayList<>();
        final Deque<Integer> windowBlocks = new ArrayDeque<>();
        int sinkLen = 0;
        int windowLen = 0;
        /** Offset into the first window block; physical ring index of the oldest live token. */
        int windowHead = 0;
        int length = 0;
    }

    @Override
    public String toString() {
        return "SlidingWindowKvCache{sink=" + sinkTokens
                + ", window=" + windowTokens
                + ", free=" + pool.freeBlocks()
                + ", seqs=" + liveSequences() + "}";
    }
}
