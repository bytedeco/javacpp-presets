package org.bytedeco.pytorch.llm.kvcache;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Content-addressed block index (TensorRT-LLM / vLLM automatic prefix caching).
 *
 * <p>Each completed KV block is keyed by a 64-bit hash of its token span (and
 * optionally the parent block hash, forming a chain). Lookups enable block-level
 * reuse across sequences without a full token-by-token radix walk.
 *
 * <p>Does not own tensors — only maps {@code hash → physical block id} with
 * refcounts via {@link PrefixRadixCache.RefCountedBlockStore}.
 */
public class BlockHashIndex implements AutoCloseable {

    /** FNV-1a 64-bit seed. */
    private static final long FNV_OFFSET = 0xcbf29ce484222325L;
    private static final long FNV_PRIME = 0x100000001b3L;

    private final PrefixRadixCache.RefCountedBlockStore store;
    private final int blockSize;
    private final int maxEntries;
    private final ReentrantLock lock = new ReentrantLock();

    /** hash → entry; access-order LRU for eviction of unreferenced entries. */
    private final Map<Long, Entry> index = new LinkedHashMap<>(256, 0.75f, true);

    public final LongAdder hitCount = new LongAdder();
    public final LongAdder missCount = new LongAdder();
    public final LongAdder insertCount = new LongAdder();
    public final LongAdder evictCount = new LongAdder();

    public BlockHashIndex(int blockSize, PrefixRadixCache.RefCountedBlockStore store) {
        this(blockSize, store, 100_000);
    }

    public BlockHashIndex(int blockSize, PrefixRadixCache.RefCountedBlockStore store, int maxEntries) {
        if (blockSize <= 0) throw new IllegalArgumentException("blockSize must be > 0");
        if (maxEntries <= 0) throw new IllegalArgumentException("maxEntries must be > 0");
        this.blockSize = blockSize;
        this.store = Objects.requireNonNull(store, "store");
        this.maxEntries = maxEntries;
    }

    public int blockSize() { return blockSize; }
    public int size() {
        lock.lock();
        try { return index.size(); }
        finally { lock.unlock(); }
    }

    /**
     * Hash one block's tokens, chained with {@code parentHash} (use 0 for the first block).
     */
    public static long hashBlock(long parentHash, int[] tokens, int offset, int len) {
        long h = FNV_OFFSET ^ parentHash;
        h = (h ^ (len & 0xffffffffL)) * FNV_PRIME;
        int end = offset + len;
        for (int i = offset; i < end; i++) {
            h = (h ^ (tokens[i] & 0xffffffffL)) * FNV_PRIME;
        }
        return h;
    }

    public static long hashBlock(long parentHash, List<Integer> tokens, int offset, int len) {
        long h = FNV_OFFSET ^ parentHash;
        h = (h ^ (len & 0xffffffffL)) * FNV_PRIME;
        for (int i = 0; i < len; i++) {
            h = (h ^ (tokens.get(offset + i) & 0xffffffffL)) * FNV_PRIME;
        }
        return h;
    }

    /**
     * Lookup a block by hash. On hit, retains the physical block and returns its id.
     * @return physical block id, or -1 on miss
     */
    public int lookup(long hash) {
        lock.lock();
        try {
            Entry e = index.get(hash);
            if (e == null) {
                missCount.increment();
                return -1;
            }
            store.retain(e.blockId);
            e.hits++;
            hitCount.increment();
            return e.blockId;
        } finally { lock.unlock(); }
    }

    /**
     * Insert or refresh a hash → block mapping. Retains one tree/index ref on first insert.
     * Evicts LRU unreferenced entries if over capacity.
     */
    public void insert(long hash, int blockId) {
        lock.lock();
        try {
            Entry existing = index.get(hash);
            if (existing != null) {
                if (existing.blockId == blockId) {
                    // touch LRU
                    index.get(hash);
                    return;
                }
                // replace
                store.release(existing.blockId);
                existing.blockId = blockId;
                store.retain(blockId);
                insertCount.increment();
                return;
            }
            maybeEvictUnlocked();
            store.retain(blockId);
            index.put(hash, new Entry(blockId));
            insertCount.increment();
        } finally { lock.unlock(); }
    }

    /**
     * Match a token sequence block-by-block. Returns physical block ids for the
     * longest hash-chain prefix hit. Each returned id is already retained once.
     */
    public List<Integer> matchPrefix(int[] tokens) {
        Objects.requireNonNull(tokens);
        List<Integer> out = new ArrayList<>();
        lock.lock();
        try {
            long parent = 0L;
            int nBlocks = tokens.length / blockSize;
            for (int b = 0; b < nBlocks; b++) {
                long h = hashBlock(parent, tokens, b * blockSize, blockSize);
                Entry e = index.get(h);
                if (e == null) {
                    missCount.increment();
                    break;
                }
                store.retain(e.blockId);
                e.hits++;
                hitCount.increment();
                out.add(e.blockId);
                parent = h;
            }
            return out;
        } finally { lock.unlock(); }
    }

    /**
     * Index all full blocks of a token sequence.
     * {@code blocks.get(i)} is the physical id for token span
     * {@code [i*blockSize, (i+1)*blockSize)}.
     */
    public void indexSequence(int[] tokens, List<Integer> blocks) {
        Objects.requireNonNull(tokens);
        Objects.requireNonNull(blocks);
        int nBlocks = Math.min(tokens.length / blockSize, blocks.size());
        long parent = 0L;
        lock.lock();
        try {
            for (int b = 0; b < nBlocks; b++) {
                long h = hashBlock(parent, tokens, b * blockSize, blockSize);
                int blockId = blocks.get(b);
                Entry existing = index.get(h);
                if (existing == null) {
                    maybeEvictUnlocked();
                    store.retain(blockId);
                    index.put(h, new Entry(blockId));
                    insertCount.increment();
                } else if (existing.blockId != blockId) {
                    store.release(existing.blockId);
                    existing.blockId = blockId;
                    store.retain(blockId);
                    insertCount.increment();
                } else {
                    index.get(h); // LRU touch
                }
                parent = h;
            }
        } finally { lock.unlock(); }
    }

    public int evictUnreferenced(int maxEvict) {
        lock.lock();
        try {
            int n = 0;
            Iterator<Map.Entry<Long, Entry>> it = index.entrySet().iterator();
            while (it.hasNext() && n < maxEvict) {
                Map.Entry<Long, Entry> e = it.next();
                // Index holds 1 ref; if refCount==1, only we hold it
                if (store.refCount(e.getValue().blockId) == 1) {
                    store.release(e.getValue().blockId);
                    it.remove();
                    evictCount.increment();
                    n++;
                }
            }
            return n;
        } finally { lock.unlock(); }
    }

    @Override
    public void close() {
        lock.lock();
        try {
            for (Entry e : index.values()) {
                try { store.release(e.blockId); } catch (Throwable ignored) {}
            }
            index.clear();
        } finally { lock.unlock(); }
    }

    private void maybeEvictUnlocked() {
        if (index.size() < maxEntries) return;
        // Evict from eldest (LRU) unreferenced first; if none, evict eldest anyway
        Iterator<Map.Entry<Long, Entry>> it = index.entrySet().iterator();
        int budget = Math.max(1, maxEntries / 10);
        int n = 0;
        while (it.hasNext() && n < budget) {
            Map.Entry<Long, Entry> e = it.next();
            if (store.refCount(e.getValue().blockId) == 1) {
                store.release(e.getValue().blockId);
                it.remove();
                evictCount.increment();
                n++;
            }
        }
        // still over capacity? force-evict eldest
        it = index.entrySet().iterator();
        while (it.hasNext() && index.size() >= maxEntries) {
            Map.Entry<Long, Entry> e = it.next();
            store.release(e.getValue().blockId);
            it.remove();
            evictCount.increment();
        }
    }

    private static final class Entry {
        int blockId;
        long hits;

        Entry(int blockId) {
            this.blockId = blockId;
            this.hits = 0L;
        }
    }

    @Override
    public String toString() {
        return "BlockHashIndex{entries=" + size()
                + ", blockSize=" + blockSize
                + ", hits=" + hitCount.sum()
                + ", misses=" + missCount.sum() + "}";
    }
}
