package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Production-oriented paged KV cache for autoregressive transformers
 * (vLLM / PagedAttention style).
 *
 * <ul>
 *   <li>Fixed-size physical blocks: {@code [2, blockSize, numHeads, headDim]} (K/V).</li>
 *   <li>Per-sequence block tables with copy-on-write on shared writes.</li>
 *   <li>Prefix radix tree over token ids; tree holds its own refcount so
 *       completed-block prefixes survive sequence release until LRU eviction.</li>
 *   <li>Watermark-driven prune of unreferenced prefix leaves.</li>
 *   <li>Optional device placement for block storage.</li>
 * </ul>
 *
 * <p>Canonical type lives here; also re-exported as
 * {@link org.bytedeco.pytorch.llm.PagedKvCache}.
 */
public class PagedKvCache implements AutoCloseable {

    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int blockSize;
    private final int maxBlocks;
    private final TensorOptions blockOptions;

    private final Tensor[][] pool;       // [layer][blockId] -> [2, blockSize, H, D]
    private final int[] refCount;        // sequence + prefix-tree references
    private final boolean[] inFreeList;
    private final long[] lastUseNs;
    private final ArrayDeque<Integer> freeList = new ArrayDeque<>();
    private final Map<Long, Seq> sequences = new HashMap<>();
    private final PrefixNode prefixRoot = new PrefixNode(-1);
    private long nextSeqId = 1;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition freeNotEmpty;
    private boolean closed = false;

    private final double lowWatermark;
    private final double highWatermark;

    public final LongAdder allocCount = new LongAdder();
    public final LongAdder evictCount = new LongAdder();
    public final LongAdder cowCount = new LongAdder();
    public final LongAdder prefixHitTokens = new LongAdder();
    public final LongAdder appendCount = new LongAdder();

    public PagedKvCache(int numLayers, int numHeads, int headDim, int blockSize, int maxBlocks) {
        this(numLayers, numHeads, headDim, blockSize, maxBlocks, null);
    }

    public PagedKvCache(int numLayers, int numHeads, int headDim, int blockSize, int maxBlocks,
                        Device device) {
        this(numLayers, numHeads, headDim, blockSize, maxBlocks, device, 0.10, 0.20);
    }

    public PagedKvCache(int numLayers, int numHeads, int headDim, int blockSize, int maxBlocks,
                        Device device, double lowWatermark, double highWatermark) {
        if (numLayers <= 0 || numHeads <= 0 || headDim <= 0 || blockSize <= 0 || maxBlocks <= 0) {
            throw new IllegalArgumentException("all size params must be > 0");
        }
        if (lowWatermark < 0 || highWatermark < lowWatermark || highWatermark > 1) {
            throw new IllegalArgumentException("invalid watermarks");
        }
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.blockSize = blockSize;
        this.maxBlocks = maxBlocks;
        this.lowWatermark = lowWatermark;
        this.highWatermark = highWatermark;

        TensorOptions opts = new TensorOptions(torch.kFloat());
        if (device != null) {
            opts = opts.device(new org.bytedeco.pytorch.DeviceOptional(device));
        }
        this.blockOptions = opts;
        this.pool = new Tensor[numLayers][maxBlocks];
        this.refCount = new int[maxBlocks];
        this.inFreeList = new boolean[maxBlocks];
        this.lastUseNs = new long[maxBlocks];
        this.freeNotEmpty = lock.newCondition();
        this.closed = false;
        this.nextSeqId = 1L;
        long now = System.nanoTime();
        for (int i = 0; i < maxBlocks; i++) {
            freeList.addLast(i);
            inFreeList[i] = true;
            refCount[i] = 0;
            lastUseNs[i] = now;
            for (int layer = 0; layer < numLayers; layer++) {
                pool[layer][i] = null; // allocated on first use via ensureBlockStorage
            }
        }
    }

    public int blockSize() { return blockSize; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public int headDim() { return headDim; }
    public int maxBlocks() { return maxBlocks; }

    public int freeBlocks() {
        lock.lock();
        try { return freeList.size(); }
        finally { lock.unlock(); }
    }

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
            sequences.put(id, new Seq(numLayers));
            return id;
        } finally { lock.unlock(); }
    }

    /** Copy-on-write fork: shares blocks until either side writes. */
    public long fork(long seqId) {
        lock.lock();
        try {
            ensureOpen();
            Seq src = require(seqId);
            Seq dst = new Seq(numLayers);
            dst.tokens.addAll(src.tokens);
            dst.length = src.length;
            for (int layer = 0; layer < numLayers; layer++) {
                for (int b : src.blocks[layer]) {
                    refCount[b]++;
                    touch(b);
                    dst.blocks[layer].add(b);
                }
            }
            long id = nextSeqId++;
            sequences.put(id, dst);
            return id;
        } finally { lock.unlock(); }
    }

    public void releaseSequence(long seqId) {
        lock.lock();
        try {
            Seq st = sequences.remove(seqId);
            if (st == null) return;
            for (int layer = 0; layer < numLayers; layer++) {
                for (int b : st.blocks[layer]) releaseRef(b);
            }
            maybePrunePrefix();
        } finally { lock.unlock(); }
    }

    /**
     * Append one token K/V for all layers.
     * Each {@code kLayers[i]} / {@code vLayers[i]} is {@code [numHeads, headDim]}
     * or {@code [1, numHeads, headDim]}.
     */
    public void append(long seqId, int tokenId, Tensor[] kLayers, Tensor[] vLayers) {
        Objects.requireNonNull(kLayers);
        Objects.requireNonNull(vLayers);
        if (kLayers.length != numLayers || vLayers.length != numLayers) {
            throw new IllegalArgumentException("layer count mismatch");
        }
        lock.lock();
        try {
            ensureOpen();
            Seq st = require(seqId);
            int pos = st.length % blockSize;
            if (pos == 0) {
                for (int layer = 0; layer < numLayers; layer++) {
                    st.blocks[layer].add(allocBlock(layer));
                }
            }
            for (int layer = 0; layer < numLayers; layer++) {
                List<Integer> table = st.blocks[layer];
                int idx = table.size() - 1;
                int b = cowIfNeeded(layer, table.get(idx), table, idx);
                Tensor slot = pool[layer][b];
                Tensor k = squeezeKV(kLayers[layer]);
                Tensor v = squeezeKV(vLayers[layer]);
                slot.select(0, 0).select(0, pos).copy_(k);
                slot.select(0, 1).select(0, pos).copy_(v);
                touch(b);
            }
            st.tokens.add(tokenId);
            st.length++;
            indexPrefix(st);
            appendCount.increment();
        } finally { lock.unlock(); }
    }

    /** Returns {@code {K, V}} each shaped {@code [T, numHeads, headDim]}. */
    public Tensor[] gather(long seqId, int layer) {
        lock.lock();
        try {
            ensureOpen();
            Seq st = require(seqId);
            if (layer < 0 || layer >= numLayers) throw new IllegalArgumentException("layer");
            if (st.length == 0) {
                // Prefer long[] overload — LongArrayRef(long[], long) does not reliably
                // pin the shape for torch.zeros and can yield garbage sizes / overflow.
                // Also allocate K and V separately so they are independent storages.
                Tensor ek = torch.zeros(new long[]{0, numHeads, headDim}, blockOptions);
                Tensor ev = torch.zeros(new long[]{0, numHeads, headDim}, blockOptions);
                return new Tensor[]{ek, ev};
            }
            List<Tensor> ks = new ArrayList<>();
            List<Tensor> vs = new ArrayList<>();
            int remaining = st.length;
            for (int b : st.blocks[layer]) {
                int take = Math.min(blockSize, remaining);
                Tensor slot = pool[layer][b];
                ks.add(slot.select(0, 0).narrow(0, 0, take));
                vs.add(slot.select(0, 1).narrow(0, 0, take));
                touch(b);
                remaining -= take;
            }
            Tensor k = torch.cat(new TensorVector(ks.toArray(new Tensor[0])), 0);
            Tensor v = torch.cat(new TensorVector(vs.toArray(new Tensor[0])), 0);
            return new Tensor[]{k, v};
        } finally { lock.unlock(); }
    }

    /**
     * Match a token prefix against the radix tree. On hit, creates a new
     * sequence that reuses shared physical blocks for all layers when available.
     */
    public PrefixHit matchPrefix(int[] tokens) {
        lock.lock();
        try {
            ensureOpen();
            PrefixNode node = prefixRoot;
            int matched = 0;
            PrefixNode best = null;
            int bestMatched = 0;
            for (int tok : tokens) {
                PrefixNode n = node.children.get(tok);
                if (n == null) break;
                node = n;
                matched++;
                if (node.sharedBlocks != null) {
                    best = node;
                    bestMatched = matched;
                }
            }
            if (best == null || best.sharedBlocks == null) {
                return new PrefixHit(0, -1L);
            }

            Seq st = new Seq(numLayers);
            for (int i = 0; i < bestMatched; i++) st.tokens.add(tokens[i]);
            st.length = bestMatched;

            if (best.sharedBlocksAllLayers != null
                    && best.sharedBlocksAllLayers.size() == numLayers) {
                for (int layer = 0; layer < numLayers; layer++) {
                    for (int b : best.sharedBlocksAllLayers.get(layer)) {
                        retainFromFree(b);
                        st.blocks[layer].add(b);
                    }
                }
            } else {
                for (int b : best.sharedBlocks) {
                    retainFromFree(b);
                    st.blocks[0].add(b);
                }
                int nBlocks = best.sharedBlocks.size();
                for (int layer = 1; layer < numLayers; layer++) {
                    for (int i = 0; i < nBlocks; i++) {
                        st.blocks[layer].add(allocBlock(layer));
                    }
                }
            }

            long id = nextSeqId++;
            sequences.put(id, st);
            prefixHitTokens.add(bestMatched);
            return new PrefixHit(bestMatched, id);
        } finally { lock.unlock(); }
    }

    public int sequenceLength(long seqId) {
        lock.lock();
        try { return require(seqId).length; }
        finally { lock.unlock(); }
    }

    public List<Integer> blockTable(long seqId, int layer) {
        lock.lock();
        try {
            return new ArrayList<>(require(seqId).blocks[layer]);
        } finally { lock.unlock(); }
    }

    /** Force prune until free ratio reaches high watermark. */
    public int prune() {
        lock.lock();
        try {
            return pruneUnlocked(targetFreeBlocks());
        } finally { lock.unlock(); }
    }

    @Override
    public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            sequences.clear();
            clearPrefix(prefixRoot);
            for (int layer = 0; layer < numLayers; layer++) {
                for (int b = 0; b < maxBlocks; b++) {
                    if (pool[layer][b] != null) {
                        try { pool[layer][b].close(); } catch (Throwable ignored) {}
                        pool[layer][b] = null;
                    }
                    refCount[b] = 0;
                    inFreeList[b] = true;
                }
            }
            freeList.clear();
            for (int i = 0; i < maxBlocks; i++) freeList.addLast(i);
            freeNotEmpty.signalAll();
        } finally { lock.unlock(); }
    }

    // ---- internals ----------------------------------------------------------

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("cache closed");
    }

    private Seq require(long id) {
        Seq st = sequences.get(id);
        if (st == null) throw new IllegalArgumentException("unknown sequence " + id);
        return st;
    }

    private void touch(int b) {
        lastUseNs[b] = System.nanoTime();
    }

    private int targetFreeBlocks() {
        return (int) Math.ceil(highWatermark * maxBlocks);
    }

    private int allocBlock(int layer) {
        if (freeList.isEmpty()) {
            pruneUnlocked(Math.max(1, targetFreeBlocks()));
        }
        if (freeList.isEmpty()) {
            throw new IllegalStateException("KV cache OOM: no free blocks after eviction");
        }
        int b = freeList.removeFirst();
        inFreeList[b] = false;
        refCount[b] = 1;
        touch(b);
        ensureBlockStorage(layer, b);
        for (int l = 0; l < numLayers; l++) {
            if (l != layer) ensureBlockStorage(l, b);
            if (l != layer && pool[l][b] != null) pool[l][b].zero_();
        }
        pool[layer][b].zero_();
        allocCount.increment();
        return b;
    }

    private void ensureBlockStorage(int layer, int b) {
        if (pool[layer][b] == null) {
            pool[layer][b] = torch.zeros(new long[]{2, blockSize, numHeads, headDim}, blockOptions);
        }
    }

    /** Drop one sequence/tree ref; return to free list at zero. */
    private void releaseRef(int b) {
        if (b < 0 || b >= maxBlocks) return;
        refCount[b]--;
        if (refCount[b] <= 0) {
            refCount[b] = 0;
            if (!inFreeList[b]) {
                freeList.addLast(b);
                inFreeList[b] = true;
                freeNotEmpty.signal();
            }
        }
    }

    /** Take a block that may currently be only tree-held (refCount >= 1 from tree). */
    private void retainFromFree(int b) {
        if (inFreeList[b]) {
            // should not happen if tree holds a ref — defensive
            freeList.remove((Integer) b);
            inFreeList[b] = false;
            refCount[b] = 1;
        } else {
            refCount[b]++;
        }
        touch(b);
    }

    private int cowIfNeeded(int layer, int blockId, List<Integer> table, int index) {
        if (refCount[blockId] <= 1) return blockId;
        int nb = allocBlock(layer);
        pool[layer][nb].copy_(pool[layer][blockId]);
        releaseRef(blockId);
        table.set(index, nb);
        cowCount.increment();
        return nb;
    }

    private static Tensor squeezeKV(Tensor t) {
        if (t.dim() == 3) return t.squeeze(0);
        if (t.dim() == 2) return t;
        throw new IllegalArgumentException("K/V must be [H,D] or [1,H,D]");
    }

    /**
     * When a sequence completes a full block, publish it into the prefix tree
     * and give the tree its own refcount so the data survives sequence release.
     */
    private void indexPrefix(Seq st) {
        if (st.length == 0 || st.length % blockSize != 0) return;
        PrefixNode node = prefixRoot;
        for (int tok : st.tokens) {
            PrefixNode n = node.children.get(tok);
            if (n == null) {
                n = new PrefixNode(tok);
                node.children.put(tok, n);
            }
            node = n;
        }
        if (st.blocks[0].isEmpty()) return;

        // If node already has shared blocks, drop old tree refs first
        dropTreeRefs(node);

        node.sharedBlocks = new ArrayList<>(st.blocks[0]);
        node.sharedBlocksAllLayers = new ArrayList<>(numLayers);
        for (int layer = 0; layer < numLayers; layer++) {
            List<Integer> bl = new ArrayList<>(st.blocks[layer]);
            node.sharedBlocksAllLayers.add(bl);
            for (int b : bl) {
                refCount[b]++; // tree holds a ref
                touch(b);
            }
        }
        node.lruKey = System.nanoTime();
    }

    private void dropTreeRefs(PrefixNode node) {
        if (node.sharedBlocksAllLayers != null) {
            for (List<Integer> layerBlocks : node.sharedBlocksAllLayers) {
                for (int b : layerBlocks) releaseRef(b);
            }
        } else if (node.sharedBlocks != null) {
            for (int b : node.sharedBlocks) releaseRef(b);
        }
        node.sharedBlocks = null;
        node.sharedBlocksAllLayers = null;
    }

    private void clearPrefix(PrefixNode node) {
        for (PrefixNode child : node.children.values()) {
            clearPrefix(child);
        }
        dropTreeRefs(node);
        node.children.clear();
    }

    private void maybePrunePrefix() {
        double freeRatio = freeList.size() / (double) maxBlocks;
        if (freeRatio < lowWatermark) {
            pruneUnlocked(targetFreeBlocks());
        }
    }

    /**
     * Evict prefix-tree-only blocks (refCount will drop to free when tree refs released)
     * preferring LRU nodes. A node is evictable when no live sequence holds its blocks
     * beyond the tree's own refs — approximated as: after releasing tree refs, blocks
     * would be free. We only evict nodes where every block has refCount == treeRefs.
     */
    private int pruneUnlocked(int wantFree) {
        if (freeList.size() >= wantFree) return 0;
        List<PrefixNode> candidates = new ArrayList<>();
        collectEvictable(prefixRoot, candidates);
        candidates.sort((a, b) -> Long.compare(a.lruKey, b.lruKey));

        int before = freeList.size();
        for (PrefixNode node : candidates) {
            if (freeList.size() >= wantFree) break;
            if (!isTreeOnly(node)) continue;
            dropTreeRefs(node);
            evictCount.increment();
        }
        int freed = freeList.size() - before;
        if (freed > 0) freeNotEmpty.signalAll();
        return freed;
    }

    private boolean isTreeOnly(PrefixNode node) {
        // Tree holds exactly one ref per block listed; if refCount == 1 for all, only tree holds them.
        if (node.sharedBlocksAllLayers != null) {
            for (List<Integer> layerBlocks : node.sharedBlocksAllLayers) {
                for (int b : layerBlocks) {
                    if (refCount[b] != 1) return false;
                }
            }
            return true;
        }
        if (node.sharedBlocks != null) {
            for (int b : node.sharedBlocks) {
                if (refCount[b] != 1) return false;
            }
            return true;
        }
        return false;
    }

    private void collectEvictable(PrefixNode node, List<PrefixNode> out) {
        if (node.sharedBlocks != null || node.sharedBlocksAllLayers != null) {
            out.add(node);
        }
        for (PrefixNode child : node.children.values()) {
            collectEvictable(child, out);
        }
    }

    private static final class Seq {
        final List<Integer> tokens = new ArrayList<>();
        final List<Integer>[] blocks;
        int length = 0;

        @SuppressWarnings("unchecked")
        Seq(int numLayers) {
            blocks = (List<Integer>[]) new List<?>[numLayers];
            for (int i = 0; i < numLayers; i++) blocks[i] = new ArrayList<>();
            this.length = 0;
        }
    }

    private static final class PrefixNode {
        final int token;
        final Map<Integer, PrefixNode> children = new HashMap<>();
        List<Integer> sharedBlocks = null;
        List<List<Integer>> sharedBlocksAllLayers = null;
        long lruKey = 0L;

        PrefixNode(int token) {
            this.token = token;
            this.sharedBlocks = null;
            this.sharedBlocksAllLayers = null;
            this.lruKey = 0L;
        }
    }

    public static final class PrefixHit {
        public final int matchedTokens;
        public final long sequenceId;

        public PrefixHit(int matchedTokens, long sequenceId) {
            this.matchedTokens = matchedTokens;
            this.sequenceId = sequenceId;
        }

        @Override
        public String toString() {
            return "PrefixHit{matched=" + matchedTokens + ", seq=" + sequenceId + "}";
        }
    }
}
