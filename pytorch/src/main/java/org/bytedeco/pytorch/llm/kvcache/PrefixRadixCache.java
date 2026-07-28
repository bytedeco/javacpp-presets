package org.bytedeco.pytorch.llm.kvcache;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

/**
 * SGLang / vLLM-style prefix radix tree over token sequences.
 *
 * <p>Nodes store optional shared physical block ids (one list covering all layers
 * when used with multi-layer blocks). Tree holds its own refcount via a
 * {@link RefCountedBlockStore} so completed prefixes survive sequence release
 * until watermark-driven LRU eviction.
 *
 * <p>This is a pure metadata structure: it does not own tensors. Pair it with
 * {@link PagedBlockManager} (or {@link PagedKvCache} which embeds an equivalent tree).
 */
public class PrefixRadixCache implements AutoCloseable {

    /** Minimal refcount ops the tree needs from a block store. */
    public interface RefCountedBlockStore {
        void retain(int blockId);
        void release(int blockId);
        int refCount(int blockId);
    }

    private final PrefixNode root = new PrefixNode(-1);
    private final ReentrantLock lock = new ReentrantLock();
    private final int blockSize;
    private final RefCountedBlockStore store;
    private final double lowWatermark;
    private final double highWatermark;
    private final int maxBlocks; // for watermark math; 0 = disable auto prune
    private boolean closed = false;

    public final LongAdder insertCount = new LongAdder();
    public final LongAdder hitTokens = new LongAdder();
    public final LongAdder missCount = new LongAdder();
    public final LongAdder evictCount = new LongAdder();

    public PrefixRadixCache(int blockSize, RefCountedBlockStore store) {
        this(blockSize, store, 0, 0.10, 0.20);
    }

    public PrefixRadixCache(int blockSize, RefCountedBlockStore store,
                            int maxBlocks, double lowWatermark, double highWatermark) {
        if (blockSize <= 0) throw new IllegalArgumentException("blockSize must be > 0");
        this.blockSize = blockSize;
        this.store = Objects.requireNonNull(store, "store");
        this.maxBlocks = Math.max(0, maxBlocks);
        if (lowWatermark < 0 || highWatermark < lowWatermark || highWatermark > 1.0) {
            throw new IllegalArgumentException("invalid watermarks");
        }
        this.lowWatermark = lowWatermark;
        this.highWatermark = highWatermark;
        this.closed = false;
    }

    public int blockSize() { return blockSize; }

    /**
     * Longest prefix match. Returns matched token count and shared block ids
     * (caller must {@link RefCountedBlockStore#retain} if it will hold them —
     * this method already retains once for the returned list).
     */
    public Match match(int[] tokens) {
        Objects.requireNonNull(tokens);
        lock.lock();
        try {
            ensureOpen();
            PrefixNode node = root;
            int matched = 0;
            PrefixNode best = null;
            int bestMatched = 0;
            for (int tok : tokens) {
                PrefixNode n = node.children.get(tok);
                if (n == null) break;
                node = n;
                matched++;
                if (node.sharedBlocks != null && !node.sharedBlocks.isEmpty()) {
                    best = node;
                    bestMatched = matched;
                }
            }
            if (best == null) {
                missCount.increment();
                return Match.miss();
            }
            List<Integer> blocks = new ArrayList<>(best.sharedBlocks);
            for (int b : blocks) store.retain(b);
            best.lruKey = System.nanoTime();
            hitTokens.add(bestMatched);
            return new Match(bestMatched, blocks);
        } finally { lock.unlock(); }
    }

    /**
     * Publish a completed prefix (token length must be a multiple of blockSize
     * for block-aligned sharing; partial tails are ignored).
     *
     * @param tokens full token id sequence
     * @param blocks physical block ids covering {@code tokens} (aligned)
     */
    public void insert(int[] tokens, List<Integer> blocks) {
        Objects.requireNonNull(tokens);
        Objects.requireNonNull(blocks);
        if (tokens.length == 0 || blocks.isEmpty()) return;
        // Only index full-block prefixes
        int aligned = (tokens.length / blockSize) * blockSize;
        if (aligned == 0) return;
        int nBlocks = aligned / blockSize;
        if (blocks.size() < nBlocks) {
            throw new IllegalArgumentException("not enough blocks for aligned prefix");
        }

        lock.lock();
        try {
            ensureOpen();
            PrefixNode node = root;
            int blockIdx = 0;
            for (int i = 0; i < aligned; i++) {
                int tok = tokens[i];
                PrefixNode n = node.children.get(tok);
                if (n == null) {
                    n = new PrefixNode(tok);
                    node.children.put(tok, n);
                }
                node = n;
                // Publish shared blocks at every block boundary so partial
                // matches (e.g. first of two blocks) can hit.
                if ((i + 1) % blockSize == 0) {
                    blockIdx++;
                    dropTreeRefs(node);
                    node.sharedBlocks = new ArrayList<>(blocks.subList(0, blockIdx));
                    for (int b : node.sharedBlocks) {
                        store.retain(b);
                    }
                    node.lruKey = System.nanoTime();
                }
            }
            insertCount.increment();
        } finally { lock.unlock(); }
    }

    /** Convenience: insert from {@link List} token ids. */
    public void insert(List<Integer> tokens, List<Integer> blocks) {
        int[] arr = new int[tokens.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = tokens.get(i);
        insert(arr, blocks);
    }

    /**
     * Evict LRU tree-only nodes until {@code wantFreeHint} is satisfied or no
     * more candidates. {@code freeBlocksNow} / {@code maxBlocks} drive the stop
     * condition when maxBlocks &gt; 0.
     *
     * @return number of nodes evicted
     */
    public int evictToFreeRatio(int freeBlocksNow, int poolMaxBlocks) {
        if (poolMaxBlocks <= 0) return 0;
        int wantFree = (int) Math.ceil(highWatermark * poolMaxBlocks);
        if (freeBlocksNow >= wantFree) return 0;
        lock.lock();
        try {
            return evictUnlocked(wantFree, freeBlocksNow, poolMaxBlocks);
        } finally { lock.unlock(); }
    }

    /** Force prune using configured maxBlocks (no-op if maxBlocks == 0). */
    public int prune(int freeBlocksNow) {
        if (maxBlocks <= 0) return 0;
        return evictToFreeRatio(freeBlocksNow, maxBlocks);
    }

    public boolean shouldPrune(int freeBlocksNow) {
        if (maxBlocks <= 0) return false;
        return freeBlocksNow / (double) maxBlocks < lowWatermark;
    }

    public int nodeCount() {
        lock.lock();
        try { return countNodes(root) - 1; } // exclude root
        finally { lock.unlock(); }
    }

    @Override
    public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            clearNode(root);
        } finally { lock.unlock(); }
    }

    // ---- internals --------------------------------------------------------

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("PrefixRadixCache closed");
    }

    private int evictUnlocked(int wantFree, int freeNow, int poolMax) {
        List<PrefixNode> candidates = new ArrayList<>();
        collectEvictable(root, candidates);
        candidates.sort((a, b) -> Long.compare(a.lruKey, b.lruKey));
        int freedNodes = 0;
        int simulatedFree = freeNow;
        for (PrefixNode node : candidates) {
            if (simulatedFree >= wantFree) break;
            if (!isTreeOnly(node)) continue;
            int n = node.sharedBlocks == null ? 0 : node.sharedBlocks.size();
            dropTreeRefs(node);
            evictCount.increment();
            freedNodes++;
            // best-effort: each tree-only block becomes free
            simulatedFree += n;
        }
        return freedNodes;
    }

    private boolean isTreeOnly(PrefixNode node) {
        if (node.sharedBlocks == null || node.sharedBlocks.isEmpty()) return false;
        for (int b : node.sharedBlocks) {
            // tree holds exactly one ref
            if (store.refCount(b) != 1) return false;
        }
        return true;
    }

    private void collectEvictable(PrefixNode node, List<PrefixNode> out) {
        if (node.sharedBlocks != null && !node.sharedBlocks.isEmpty()) {
            out.add(node);
        }
        for (PrefixNode c : node.children.values()) collectEvictable(c, out);
    }

    private void dropTreeRefs(PrefixNode node) {
        if (node.sharedBlocks != null) {
            for (int b : node.sharedBlocks) store.release(b);
            node.sharedBlocks = null;
        }
    }

    private void clearNode(PrefixNode node) {
        for (PrefixNode c : node.children.values()) clearNode(c);
        dropTreeRefs(node);
        node.children.clear();
    }

    private int countNodes(PrefixNode node) {
        int n = 1;
        for (PrefixNode c : node.children.values()) n += countNodes(c);
        return n;
    }

    private static final class PrefixNode {
        final int token;
        final Map<Integer, PrefixNode> children = new HashMap<>();
        List<Integer> sharedBlocks = null;
        long lruKey = 0L;

        PrefixNode(int token) {
            this.token = token;
            this.sharedBlocks = null;
            this.lruKey = 0L;
        }
    }

    /** Result of a prefix match. */
    public static final class Match {
        public final int matchedTokens;
        public final List<Integer> blockIds;

        public Match(int matchedTokens, List<Integer> blockIds) {
            this.matchedTokens = matchedTokens;
            this.blockIds = blockIds == null ? List.of() : List.copyOf(blockIds);
        }

        public static Match miss() { return new Match(0, List.of()); }

        public boolean hit() { return matchedTokens > 0; }

        @Override
        public String toString() {
            return "Match{tokens=" + matchedTokens + ", blocks=" + blockIds.size() + "}";
        }
    }
}
