package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.global.torch;

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
 * Session-oriented copy-on-write block manager (TensorRT-LLM / vLLM style).
 *
 * <p>Owns a {@link PagedBlockManager} physical pool and tracks live sessions
 * with access-order LRU. When the free list is empty, the oldest session
 * (other than the requester) is preempted and its blocks recycled.
 *
 * <p>Also supports:
 * <ul>
 *   <li>Per-block CoW: shared blocks are forked on write so forked / prefix-shared
 *       sequences remain isolated after mutation.</li>
 *   <li>Content-addressed block reuse via an embedded {@link BlockHashIndex}
 *       ({@link #matchAndAllocatePath}, {@link #getOrAllocateBlock}).</li>
 * </ul>
 *
 * <p>Thread-safe. Prefer {@link PagedKvCache} for full sequence+prefix APIs;
 * this class is the lower-level building block used by {@link PagedKvBuffer}
 * and hierarchical / multi-tenant schedulers.
 */
public class CoWBlockManager implements AutoCloseable {

    private final PagedBlockManager pool;
    private final BlockHashIndex hashIndex;
    private final ReentrantLock lock = new ReentrantLock();

    /** Access-order LRU of live sessions. Value holds the buffer that owns blocks. */
    private final Map<String, PagedKvBuffer> activeSessions =
            new LinkedHashMap<>(16, 0.75f, true);

    /** Blocks currently charged to each session (union of K/V block maps). */
    private final Map<String, List<Integer>> sessionBlocks = new HashMap<>();

    public final LongAdder allocRequests = new LongAdder();
    public final LongAdder evictCount = new LongAdder();
    public final LongAdder cowCount = new LongAdder();
    public final LongAdder preemptedSessions = new LongAdder();

    public CoWBlockManager(int totalBlocks, int numLayers, int blockSize, int numHeads, int headDim) {
        this(totalBlocks, numLayers, blockSize, numHeads, headDim, null, torch.kFloat());
    }

    public CoWBlockManager(int totalBlocks, int numLayers, int blockSize, int numHeads, int headDim,
                           Device device) {
        this(totalBlocks, numLayers, blockSize, numHeads, headDim, device, torch.kFloat());
    }

    public CoWBlockManager(int totalBlocks, int numLayers, int blockSize, int numHeads, int headDim,
                           Device device, torch.ScalarType dtype) {
        this.pool = new PagedBlockManager(
                totalBlocks, numLayers, blockSize, numHeads, headDim,
                device, dtype, 0.10, 0.20);
        this.hashIndex = new BlockHashIndex(blockSize, adapt(pool));
    }

    /**
     * Legacy demo shape {@code (totalBlocks, layers, blockSize, headDim, dtypeValue)}
     * with {@code numHeads = 1}. Named factory avoids clashing with the 5-int
     * {@code (…, numHeads, headDim)} constructor erasure.
     */
    public static CoWBlockManager withDtypeValue(int totalBlocks, int layers, int blockSize,
                                                int headDim, int dtype) {
        return new CoWBlockManager(totalBlocks, layers, blockSize, 1, headDim,
                null, resolveDtype(dtype));
    }

    private static torch.ScalarType resolveDtype(int value) {
        for (torch.ScalarType e : torch.ScalarType.values()) {
            if (e.value == value) return e;
        }
        return torch.kFloat();
    }

    public PagedBlockManager pool() { return pool; }
    public BlockHashIndex hashIndex() { return hashIndex; }
    public int totalBlocks() { return pool.maxBlocks(); }
    public int getBlockSize() { return pool.blockSize(); }
    public int numLayers() { return pool.numLayers(); }
    public int numHeads() { return pool.numHeads(); }
    public int headDim() { return pool.headDim(); }
    public int getFreeBlockCount() { return pool.freeBlocks(); }
    public int getActiveBlockCount() { return pool.usedBlocks(); }

    public int activeSessionCount() {
        lock.lock();
        try { return activeSessions.size(); }
        finally { lock.unlock(); }
    }

    /**
     * Register or touch a session in the LRU without allocating blocks.
     */
    public void registerSession(String sessionId, PagedKvBuffer buffer) {
        Objects.requireNonNull(sessionId, "sessionId");
        lock.lock();
        try {
            ensureOpen();
            activeSessions.put(sessionId, buffer);
            sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>());
        } finally { lock.unlock(); }
    }

    /**
     * Allocate {@code count} exclusive blocks for {@code sessionId}.
     * Updates LRU position of the session. May preempt other sessions.
     */
    public List<Integer> allocateBlocks(int count, String sessionId, PagedKvBuffer currentBuffer) {
        Objects.requireNonNull(sessionId, "sessionId");
        if (count < 0) throw new IllegalArgumentException("count must be >= 0");
        if (count == 0) {
            registerSession(sessionId, currentBuffer);
            return new ArrayList<>();
        }
        allocRequests.increment();

        lock.lock();
        try {
            ensureOpen();
            // Touch / register session in LRU
            if (currentBuffer != null) {
                activeSessions.put(sessionId, currentBuffer);
            } else if (!activeSessions.containsKey(sessionId)) {
                activeSessions.put(sessionId, null);
            } else {
                activeSessions.get(sessionId); // access-order touch
            }

            List<Integer> allocated = new ArrayList<>(count);
            while (allocated.size() < count) {
                int need = count - allocated.size();
                try {
                    List<Integer> got = pool.allocateBlocks(need);
                    allocated.addAll(got);
                } catch (IllegalStateException oom) {
                    if (!evictOldestSessionUnlocked(sessionId)) {
                        // roll back partial
                        if (!allocated.isEmpty()) pool.releaseAll(allocated);
                        throw new IllegalStateException(
                                "GPU/host KV memory exhausted: need " + count
                                        + " blocks, free=" + pool.freeBlocks()
                                        + ", sessions=" + activeSessions.size(), oom);
                    }
                    evictCount.increment();
                }
            }
            sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>()).addAll(allocated);
            return allocated;
        } finally { lock.unlock(); }
    }

    /**
     * Copy-on-write a single block for a session. Returns the (possibly new) block id
     * that is exclusive to the caller.
     */
    public int cowBlock(int blockId, String sessionId) {
        lock.lock();
        try {
            ensureOpen();
            int nb = pool.cowIfNeeded(blockId);
            if (nb != blockId) {
                cowCount.increment();
                List<Integer> held = sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>());
                // replace tracking: drop old if present, add new
                held.remove((Integer) blockId);
                held.add(nb);
            }
            return nb;
        } finally { lock.unlock(); }
    }

    /**
     * Fork all blocks of {@code srcSession} into {@code dstSession} with shared refs
     * (true CoW). Caller is responsible for building the destination buffer's maps.
     *
     * @return list of shared block ids (same physical ids as source)
     */
    public List<Integer> forkSessionBlocks(String srcSession, String dstSession) {
        Objects.requireNonNull(srcSession);
        Objects.requireNonNull(dstSession);
        lock.lock();
        try {
            ensureOpen();
            List<Integer> src = sessionBlocks.get(srcSession);
            if (src == null || src.isEmpty()) return new ArrayList<>();
            List<Integer> shared = new ArrayList<>(src.size());
            for (int b : src) {
                pool.retain(b);
                shared.add(b);
            }
            sessionBlocks.put(dstSession, new ArrayList<>(shared));
            // touch both in LRU
            if (activeSessions.containsKey(srcSession)) activeSessions.get(srcSession);
            activeSessions.putIfAbsent(dstSession, activeSessions.get(srcSession));
            return shared;
        } finally { lock.unlock(); }
    }

    /**
     * Release all blocks tracked for a session and remove it from the LRU.
     * Does <em>not</em> close the buffer itself.
     */
    public void releaseSession(String sessionId) {
        if (sessionId == null) return;
        lock.lock();
        try {
            activeSessions.remove(sessionId);
            List<Integer> held = sessionBlocks.remove(sessionId);
            if (held != null && !held.isEmpty()) {
                pool.releaseAll(held);
            }
        } finally { lock.unlock(); }
    }

    /**
     * Track already-retained block ids against a session without allocating.
     * Does not call {@link PagedBlockManager#retain}; caller must hold the refs.
     */
    public void trackBlocks(String sessionId, List<Integer> blocks) {
        if (sessionId == null || blocks == null || blocks.isEmpty()) return;
        lock.lock();
        try {
            ensureOpen();
            List<Integer> held = sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>());
            for (int b : blocks) {
                if (!held.contains(b)) held.add(b);
            }
        } finally { lock.unlock(); }
    }

    /**
     * Return specific blocks to the pool and un-charge them from the session.
     */
    public void releaseBlocks(String sessionId, List<Integer> blocks) {
        if (blocks == null || blocks.isEmpty()) return;
        lock.lock();
        try {
            pool.releaseAll(blocks);
            List<Integer> held = sessionBlocks.get(sessionId);
            if (held != null) {
                for (int b : blocks) held.remove((Integer) b);
            }
        } finally { lock.unlock(); }
    }

    /**
     * Force-evict the LRU session other than {@code excludeSessionId}.
     * @return true if a victim was found and reclaimed
     */
    public boolean evictOldestSession(String excludeSessionId) {
        lock.lock();
        try {
            return evictOldestSessionUnlocked(excludeSessionId);
        } finally { lock.unlock(); }
    }

    /**
     * Content-addressed path match + allocate (TRT-LLM / vLLM automatic prefix cache).
     * Hits: {@link BlockHashIndex#lookup} retains once; block is tracked on the session.
     * Misses: fresh allocation via {@link #allocateBlocks}, then inserted into the index.
     */
    public List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBuffer buffer) {
        if (pathHashes == null || pathHashes.isEmpty()) return new ArrayList<>();
        registerSession(sessionId, buffer);

        List<Integer> result = new ArrayList<>(pathHashes.size());
        List<Integer> hitBlocks = new ArrayList<>();
        int missCount = 0;

        for (long h : pathHashes) {
            int b = hashIndex.lookup(h); // retains on hit
            if (b >= 0) {
                result.add(b);
                hitBlocks.add(b);
            } else {
                result.add(-1);
                missCount++;
            }
        }

        if (missCount > 0) {
            List<Integer> fresh = allocateBlocks(missCount, sessionId, buffer);
            int fi = 0;
            for (int i = 0; i < pathHashes.size(); i++) {
                if (result.get(i) < 0) {
                    int b = fresh.get(fi++);
                    result.set(i, b);
                    hashIndex.insert(pathHashes.get(i), b);
                }
            }
        }

        // Hits need session tracking (misses already charged by allocateBlocks).
        trackBlocks(sessionId, hitBlocks);
        return result;
    }

    /** Single content-hash get-or-allocate. */
    public int getOrAllocateBlock(long currentHash, String sessionId, PagedKvBuffer buffer) {
        registerSession(sessionId, buffer);
        int b = hashIndex.lookup(currentHash);
        if (b >= 0) {
            trackBlocks(sessionId, List.of(b));
            return b;
        }
        List<Integer> got = allocateBlocks(1, sessionId, buffer);
        int nb = got.get(0);
        hashIndex.insert(currentHash, nb);
        return nb;
    }

    /**
     * Release a session and opportunistically drop hash-index entries that became
     * unreferenced (tree/index-only).
     */
    public void releaseSessionAndSweep(String sessionId) {
        releaseSession(sessionId);
        hashIndex.evictUnreferenced(64);
    }

    private boolean evictOldestSessionUnlocked(String excludeSessionId) {
        Iterator<Map.Entry<String, PagedKvBuffer>> it = activeSessions.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, PagedKvBuffer> entry = it.next();
            String victimId = entry.getKey();
            if (excludeSessionId != null && excludeSessionId.equals(victimId)) continue;

            PagedKvBuffer victim = entry.getValue();
            it.remove();

            List<Integer> released;
            if (victim != null) {
                released = victim.getAndInvalidateBlocks();
            } else {
                released = sessionBlocks.getOrDefault(victimId, new ArrayList<>());
            }
            sessionBlocks.remove(victimId);
            if (released != null && !released.isEmpty()) {
                pool.releaseAll(released);
            }
            preemptedSessions.increment();
            return true;
        }
        return false;
    }

    private void ensureOpen() {
        if (pool.isClosed()) throw new IllegalStateException("CoWBlockManager closed");
    }

    private static PrefixRadixCache.RefCountedBlockStore adapt(PagedBlockManager pool) {
        return new PrefixRadixCache.RefCountedBlockStore() {
            @Override public void retain(int blockId) { pool.retain(blockId); }
            @Override public void release(int blockId) { pool.release(blockId); }
            @Override public int refCount(int blockId) { return pool.refCount(blockId); }
        };
    }

    @Override
    public void close() {
        lock.lock();
        try {
            for (PagedKvBuffer buf : activeSessions.values()) {
                if (buf != null) {
                    try { buf.getAndInvalidateBlocks(); } catch (Throwable ignored) {}
                }
            }
            activeSessions.clear();
            sessionBlocks.clear();
            try { hashIndex.close(); } catch (Throwable ignored) {}
            pool.close();
        } finally { lock.unlock(); }
    }

    @Override
    public String toString() {
        return "CoWBlockManager{pool=" + pool
                + ", sessions=" + activeSessionCount()
                + ", free=" + getFreeBlockCount()
                + ", hashEntries=" + hashIndex.size() + "}";
    }
}
