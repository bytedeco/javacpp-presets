package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Per-session paged KV buffer backed by a {@link CoWBlockManager}.
 *
 * <p>Physical blocks are multi-layer ({@code [numLayers, 2, blockSize, H, D]}), so a
 * single shared block table covers every layer. Supports:
 * <ul>
 *   <li>Prefill of a contiguous token span into newly allocated blocks.</li>
 *   <li>Decode append of a single token (allocates a new block on boundary).</li>
 *   <li>Gather of full K/V history for attention.</li>
 *   <li>Preemption-safe invalidation via {@link #getAndInvalidateBlocks()}.</li>
 *   <li>Copy-on-write fork from another buffer (shares physical blocks).</li>
 * </ul>
 *
 * <p>Thread-safe for concurrent readers vs a single writer / invalidator.
 */
public class PagedKvBuffer implements AutoCloseable {

    private final String sessionId;
    private final CoWBlockManager manager;
    private final int numLayers;
    private final int blockSize;
    private final int numHeads;
    private final int headDim;

    /** Shared physical block ids (each stores all layers' K and V). */
    private final List<Integer> blockTable = new ArrayList<>();

    private final ReentrantReadWriteLock stateLock = new ReentrantReadWriteLock();
    private final AtomicBoolean invalidated = new AtomicBoolean(false);
    private final AtomicInteger length = new AtomicInteger(0);

    public final LongAdder prefillTokens = new LongAdder();
    public final LongAdder decodeTokens = new LongAdder();
    public final LongAdder cowEvents = new LongAdder();

    public PagedKvBuffer(String sessionId, CoWBlockManager manager, int numLayers) {
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId");
        this.manager = Objects.requireNonNull(manager, "manager");
        if (numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
        if (numLayers != manager.numLayers()) {
            throw new IllegalArgumentException("numLayers mismatch: buffer=" + numLayers
                    + " manager=" + manager.numLayers());
        }
        this.numLayers = numLayers;
        this.blockSize = manager.getBlockSize();
        this.numHeads = manager.numHeads();
        this.headDim = manager.headDim();
        manager.registerSession(sessionId, this);
    }

    public String sessionId() { return sessionId; }
    public int numLayers() { return numLayers; }
    public int blockSize() { return blockSize; }
    public int numHeads() { return numHeads; }
    public int headDim() { return headDim; }
    public int length() { return length.get(); }
    public boolean isInvalidated() { return invalidated.get(); }

    public int getKBlockCount(int layer) {
        checkLayer(layer);
        stateLock.readLock().lock();
        try { return blockTable.size(); }
        finally { stateLock.readLock().unlock(); }
    }

    /** Alias: K and V share physical blocks in the dual-slot pool layout. */
    public int getVBlockCount(int layer) { return getKBlockCount(layer); }

    public List<Integer> blockTable() {
        stateLock.readLock().lock();
        try { return new ArrayList<>(blockTable); }
        finally { stateLock.readLock().unlock(); }
    }

    /** Per-layer view (same physical ids for every layer). */
    public List<Integer> blockTable(int layer) {
        checkLayer(layer);
        return blockTable();
    }

    /**
     * Prefill one layer: write {@code T} consecutive K/V rows.
     * {@code k}/{@code v} shaped {@code [T, numHeads, headDim]}.
     */
    public void prefill(int layer, Tensor k, Tensor v) {
        Objects.requireNonNull(k);
        Objects.requireNonNull(v);
        stateLock.writeLock().lock();
        try {
            ensureLive();
            checkLayer(layer);
            int t = (int) k.size(0);
            if (t == 0) return;
            if (v.size(0) != t) throw new IllegalArgumentException("K/V token count mismatch");

            int start = length.get();
            ensureCapacityUnlocked(start + t);
            PagedBlockManager pool = manager.pool();
            for (int i = 0; i < t; i++) {
                int abs = start + i;
                int blockId = exclusiveBlockUnlocked(abs / blockSize);
                pool.writeToken(blockId, layer, abs % blockSize, k.select(0, i), v.select(0, i));
            }
            length.updateAndGet(cur -> Math.max(cur, start + t));
            prefillTokens.add(t);
        } finally { stateLock.writeLock().unlock(); }
    }

    /**
     * Prefill all layers at once. {@code kLayers[i]}/{@code vLayers[i]} are
     * {@code [T, H, D]}. Advances sequence length by T exactly once.
     */
    public void prefillAll(Tensor[] kLayers, Tensor[] vLayers) {
        Objects.requireNonNull(kLayers);
        Objects.requireNonNull(vLayers);
        if (kLayers.length != numLayers || vLayers.length != numLayers) {
            throw new IllegalArgumentException("layer count mismatch");
        }
        stateLock.writeLock().lock();
        try {
            ensureLive();
            int t = (int) kLayers[0].size(0);
            if (t == 0) return;
            for (int layer = 0; layer < numLayers; layer++) {
                if (kLayers[layer].size(0) != t || vLayers[layer].size(0) != t) {
                    throw new IllegalArgumentException("inconsistent T across layers");
                }
            }
            int start = length.get();
            ensureCapacityUnlocked(start + t);
            PagedBlockManager pool = manager.pool();
            for (int i = 0; i < t; i++) {
                int abs = start + i;
                int blockId = exclusiveBlockUnlocked(abs / blockSize);
                int pos = abs % blockSize;
                for (int layer = 0; layer < numLayers; layer++) {
                    pool.writeToken(blockId, layer, pos,
                            kLayers[layer].select(0, i),
                            vLayers[layer].select(0, i));
                }
            }
            length.set(start + t);
            prefillTokens.add(t);
        } finally { stateLock.writeLock().unlock(); }
    }

    /**
     * Decode step: append one token for all layers.
     * Each {@code kLayers[i]}/{@code vLayers[i]} is {@code [H,D]} or {@code [1,H,D]}.
     */
    public void append(Tensor[] kLayers, Tensor[] vLayers) {
        Objects.requireNonNull(kLayers);
        Objects.requireNonNull(vLayers);
        if (kLayers.length != numLayers || vLayers.length != numLayers) {
            throw new IllegalArgumentException("layer count mismatch");
        }
        stateLock.writeLock().lock();
        try {
            ensureLive();
            int posInSeq = length.get();
            ensureCapacityUnlocked(posInSeq + 1);
            int blockId = exclusiveBlockUnlocked(posInSeq / blockSize);
            int pos = posInSeq % blockSize;
            PagedBlockManager pool = manager.pool();
            for (int layer = 0; layer < numLayers; layer++) {
                pool.writeToken(blockId, layer, pos, kLayers[layer], vLayers[layer]);
            }
            length.incrementAndGet();
            decodeTokens.increment();
        } finally { stateLock.writeLock().unlock(); }
    }

    /** Gather full history for one layer: {@code {K,V}} each {@code [T,H,D]}. */
    public Tensor[] gather(int layer) {
        stateLock.readLock().lock();
        try {
            ensureLive();
            checkLayer(layer);
            return manager.pool().gather(blockTable, layer, length.get());
        } finally { stateLock.readLock().unlock(); }
    }

    /**
     * Fork this buffer into a new session: shares physical blocks (CoW).
     * Subsequent writes on either side diverge via copy-on-write.
     */
    public PagedKvBuffer fork(String newSessionId) {
        Objects.requireNonNull(newSessionId);
        stateLock.readLock().lock();
        try {
            ensureLive();
            PagedKvBuffer child = new PagedKvBuffer(newSessionId, manager, numLayers);
            child.stateLock.writeLock().lock();
            try {
                child.blockTable.addAll(this.blockTable);
                child.length.set(this.length.get());
            } finally {
                child.stateLock.writeLock().unlock();
            }
            if (!blockTable.isEmpty()) {
                // forkSessionBlocks retains once per physical id and charges dst session.
                manager.forkSessionBlocks(sessionId, newSessionId);
            }
            return child;
        } finally { stateLock.readLock().unlock(); }
    }

    /**
     * Called by the manager on preemption: freeze the buffer and return every
     * physical block id it holds so the pool can reclaim them.
     */
    public List<Integer> getAndInvalidateBlocks() {
        stateLock.writeLock().lock();
        try {
            invalidated.set(true);
            List<Integer> all = new ArrayList<>(blockTable);
            blockTable.clear();
            length.set(0);
            return all;
        } finally { stateLock.writeLock().unlock(); }
    }

    @Override
    public void close() {
        stateLock.writeLock().lock();
        try {
            if (!invalidated.getAndSet(true)) {
                blockTable.clear();
                length.set(0);
            }
            manager.releaseSession(sessionId);
        } finally { stateLock.writeLock().unlock(); }
    }

    // ---- internals --------------------------------------------------------

    private void ensureLive() {
        if (invalidated.get()) {
            throw new IllegalStateException("session " + sessionId + " has been evicted/closed");
        }
    }

    private void checkLayer(int layer) {
        if (layer < 0 || layer >= numLayers) {
            throw new IllegalArgumentException("layer out of range: " + layer);
        }
    }

    /** Ensure shared block table covers absolute positions {@code [0, absLen)}. */
    private void ensureCapacityUnlocked(int absLen) {
        int needBlocks = absLen <= 0 ? 0 : (absLen + blockSize - 1) / blockSize;
        int have = blockTable.size();
        if (have >= needBlocks) return;
        List<Integer> neu = manager.allocateBlocks(needBlocks - have, sessionId, this);
        blockTable.addAll(neu);
    }

    /** Return block id at index, CoW-ing if shared. */
    private int exclusiveBlockUnlocked(int blockIdx) {
        int b = blockTable.get(blockIdx);
        int nb = manager.cowBlock(b, sessionId);
        if (nb != b) {
            blockTable.set(blockIdx, nb);
            cowEvents.increment();
        }
        return nb;
    }

    // ---- compatibility shims ----------------------------------------------

    /**
     * Legacy single-half prefill. Writes {@code input} ({@code [T,H,D]}) into both
     * K and V slots so gather stays well-defined. Prefer {@link #prefill} / {@link #prefillAll}.
     */
    @Deprecated
    public void prefillUltra(int layer, int kvType, Tensor input) {
        stateLock.writeLock().lock();
        try {
            ensureLive();
            checkLayer(layer);
            int t = (int) input.size(0);
            if (t == 0) return;
            int start = length.get();
            ensureCapacityUnlocked(start + t);
            PagedBlockManager pool = manager.pool();
            for (int i = 0; i < t; i++) {
                int abs = start + i;
                int blockId = exclusiveBlockUnlocked(abs / blockSize);
                Tensor row = input.select(0, i);
                pool.writeToken(blockId, layer, abs % blockSize, row, row);
            }
            length.updateAndGet(cur -> Math.max(cur, start + t));
            prefillTokens.add(t);
        } finally { stateLock.writeLock().unlock(); }
    }

    /** Compatibility: older code used CharSequence. */
    public CharSequence getSessionId() { return sessionId; }

    @Override
    public String toString() {
        return "PagedKvBuffer{session=" + sessionId
                + ", len=" + length.get()
                + ", blocks=" + blockTable.size()
                + ", layers=" + numLayers
                + ", invalidated=" + invalidated.get() + "}";
    }
}
