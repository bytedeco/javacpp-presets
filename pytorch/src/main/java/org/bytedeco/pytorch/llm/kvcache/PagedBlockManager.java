package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Industrial physical block pool for paged KV cache (vLLM / TensorRT-LLM style).
 *
 * <p>Layout of the backing tensor:
 * {@code [maxBlocks, numLayers, 2, blockSize, numHeads, headDim]} where axis-2 is K/V.
 *
 * <ul>
 *   <li>O(1) free-list allocation / release with per-block reference counts.</li>
 *   <li>Optional device placement (CPU / CUDA / MPS).</li>
 *   <li>Watermark-driven free-space accounting (callers decide what to evict).</li>
 *   <li>Thread-safe under a single reentrant lock; waiters can block on free space.</li>
 *   <li>Pool tensor is allocated in the constructor; per-block zeroing is lazy on free→alloc.</li>
 * </ul>
 */
public class PagedBlockManager implements AutoCloseable {

    private final int maxBlocks;
    private final int numLayers;
    private final int blockSize;
    private final int numHeads;
    private final int headDim;
    private final TensorOptions options;

    /**
     * Shared physical pool {@code [maxBlocks, numLayers, 2, blockSize, numHeads, headDim]}.
     * Always non-null after successful construction; set null only after {@link #close()}.
     */
    private Tensor blockPool;

    private final int[] refCount;
    private final boolean[] inFreeList;
    private final boolean[] dirty; // true => needs zero_ before next exclusive use
    private final long[] lastUseNs;
    private final ArrayDeque<Integer> freeList = new ArrayDeque<>();

    private final ReentrantLock lock = new ReentrantLock();
    private final Condition freeNotEmpty;
    private boolean closed = false;

    private final double lowWatermark;
    private final double highWatermark;

    public final LongAdder allocCount = new LongAdder();
    public final LongAdder freeCount = new LongAdder();
    public final LongAdder waitCount = new LongAdder();
    public final LongAdder retainCount = new LongAdder();

    public PagedBlockManager(int maxBlocks, int numLayers, int blockSize, int numHeads, int headDim) {
        this(maxBlocks, numLayers, blockSize, numHeads, headDim, null, torch.kFloat(), 0.10, 0.20);
    }

    public PagedBlockManager(int maxBlocks, int numLayers, int blockSize, int numHeads, int headDim,
                             Device device) {
        this(maxBlocks, numLayers, blockSize, numHeads, headDim, device, torch.kFloat(), 0.10, 0.20);
    }

    /**
     * @param scalarType torch dtype for block storage (e.g. {@code torch.kFloat()}, {@code torch.kHalf()})
     */
    public PagedBlockManager(int maxBlocks, int numLayers, int blockSize, int numHeads, int headDim,
                             Device device, torch.ScalarType scalarType,
                             double lowWatermark, double highWatermark) {
        if (maxBlocks <= 0 || numLayers <= 0 || blockSize <= 0 || numHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("all size params must be > 0");
        }
        if (lowWatermark < 0 || highWatermark < lowWatermark || highWatermark > 1.0) {
            throw new IllegalArgumentException("invalid watermarks: low=" + lowWatermark
                    + " high=" + highWatermark);
        }
        Objects.requireNonNull(scalarType, "scalarType");

        this.maxBlocks = maxBlocks;
        this.numLayers = numLayers;
        this.blockSize = blockSize;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.lowWatermark = lowWatermark;
        this.highWatermark = highWatermark;
        this.closed = false;

        TensorOptions opts = new TensorOptions(scalarType);
        if (device != null) {
            opts = opts.device(new DeviceOptional(device));
        }
        this.options = opts;

        this.refCount = new int[maxBlocks];
        this.inFreeList = new boolean[maxBlocks];
        this.dirty = new boolean[maxBlocks];
        this.lastUseNs = new long[maxBlocks];
        this.freeNotEmpty = lock.newCondition();

        // Eager pool allocation — all structural fields fully initialized after ctor.
        this.blockPool = torch.zeros(
                new long[]{maxBlocks, numLayers, 2L, blockSize, numHeads, headDim},
                this.options);

        long now = System.nanoTime();
        for (int i = 0; i < maxBlocks; i++) {
            freeList.addLast(i);
            inFreeList[i] = true;
            dirty[i] = false; // freshly zeroed pool
            refCount[i] = 0;
            lastUseNs[i] = now;
        }
    }

    /**
     * Legacy demo shape {@code (maxBlocks, numLayers, blockSize, headDim, dtypeValue)} with
     * {@code numHeads = 1}. Named factory avoids clashing with the 5-int
     * {@code (…, numHeads, headDim)} constructor erasure.
     */
    public static PagedBlockManager withDtypeValue(int maxBlocks, int numLayers, int blockSize,
                                                   int headDim, int scalarTypeValue) {
        return new PagedBlockManager(maxBlocks, numLayers, blockSize, 1, headDim,
                null, resolveScalarType(scalarTypeValue), 0.10, 0.20);
    }

    private static torch.ScalarType resolveScalarType(int value) {
        for (torch.ScalarType e : torch.ScalarType.values()) {
            if (e.value == value) return e;
        }
        return torch.kFloat();
    }

    // ---- accessors --------------------------------------------------------

    public int maxBlocks() { return maxBlocks; }
    public int numLayers() { return numLayers; }
    public int blockSize() { return blockSize; }
    /** Demo alias. */
    public int getBlockSize() { return blockSize; }
    public int numHeads() { return numHeads; }
    public int headDim() { return headDim; }
    public double lowWatermark() { return lowWatermark; }
    public double highWatermark() { return highWatermark; }
    public TensorOptions options() { return options; }

    public int freeBlocks() {
        lock.lock();
        try { return freeList.size(); }
        finally { lock.unlock(); }
    }

    public int usedBlocks() {
        lock.lock();
        try { return maxBlocks - freeList.size(); }
        finally { lock.unlock(); }
    }

    public double freeRatio() {
        lock.lock();
        try { return freeList.size() / (double) maxBlocks; }
        finally { lock.unlock(); }
    }

    public boolean belowLowWatermark() {
        return freeRatio() < lowWatermark;
    }

    public int targetFreeBlocks() {
        return (int) Math.ceil(highWatermark * maxBlocks);
    }

    public int refCount(int blockId) {
        lock.lock();
        try {
            checkBlockId(blockId);
            return refCount[blockId];
        } finally { lock.unlock(); }
    }

    // ---- allocation -------------------------------------------------------

    /**
     * Allocate one exclusive block (refCount = 1). Throws if pool is exhausted.
     * Does not wait; use {@link #allocateBlockBlocking(long)} to wait.
     */
    public int allocateBlock() {
        lock.lock();
        try {
            ensureOpen();
            if (freeList.isEmpty()) {
                throw new IllegalStateException("KV block pool OOM: no free blocks");
            }
            return takeFreeUnlocked();
        } finally { lock.unlock(); }
    }

    /** Allocate {@code count} exclusive blocks. All-or-nothing: rolls back on failure. */
    public List<Integer> allocateBlocks(int count) {
        if (count <= 0) return new ArrayList<>();
        lock.lock();
        try {
            ensureOpen();
            if (freeList.size() < count) {
                throw new IllegalStateException("KV block pool OOM: need " + count
                        + " free, have " + freeList.size());
            }
            List<Integer> out = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                out.add(takeFreeUnlocked());
            }
            return out;
        } finally { lock.unlock(); }
    }

    /**
     * Allocate one block, waiting up to {@code timeoutNs} nanoseconds for free space.
     * @return block id, or -1 on timeout
     */
    public int allocateBlockBlocking(long timeoutNs) throws InterruptedException {
        lock.lock();
        try {
            ensureOpen();
            long deadline = timeoutNs <= 0 ? 0 : System.nanoTime() + timeoutNs;
            while (freeList.isEmpty()) {
                waitCount.increment();
                if (timeoutNs <= 0) {
                    freeNotEmpty.await();
                } else {
                    long remaining = deadline - System.nanoTime();
                    if (remaining <= 0) return -1;
                    freeNotEmpty.awaitNanos(remaining);
                }
                ensureOpen();
            }
            return takeFreeUnlocked();
        } finally { lock.unlock(); }
    }

    /** Bump refCount of an already-allocated (or tree-held) block. */
    public void retain(int blockId) {
        lock.lock();
        try {
            ensureOpen();
            checkBlockId(blockId);
            if (inFreeList[blockId]) {
                freeList.remove((Integer) blockId);
                inFreeList[blockId] = false;
                refCount[blockId] = 1;
            } else {
                refCount[blockId]++;
            }
            touch(blockId);
            retainCount.increment();
        } finally { lock.unlock(); }
    }

    public void retainAll(Iterable<Integer> blockIds) {
        lock.lock();
        try {
            ensureOpen();
            for (int b : blockIds) {
                checkBlockId(b);
                if (inFreeList[b]) {
                    freeList.remove((Integer) b);
                    inFreeList[b] = false;
                    refCount[b] = 1;
                } else {
                    refCount[b]++;
                }
                touch(b);
                retainCount.increment();
            }
        } finally { lock.unlock(); }
    }

    /**
     * Drop one reference. When refCount hits 0 the block returns to the free list
     * (storage is kept; marked dirty for lazy zero on next alloc).
     */
    /** Demo alias for {@link #release(int)}. */
    public void freeBlock(int blockId) { release(blockId); }

    public void release(int blockId) {
        lock.lock();
        try {
            if (closed) return;
            checkBlockId(blockId);
            releaseUnlocked(blockId);
        } finally { lock.unlock(); }
    }

    public void releaseAll(Iterable<Integer> blockIds) {
        lock.lock();
        try {
            if (closed) return;
            for (int b : blockIds) {
                checkBlockId(b);
                releaseUnlocked(b);
            }
        } finally { lock.unlock(); }
    }

    /**
     * Copy-on-write: if {@code blockId} is shared (refCount &gt; 1), allocate a fresh
     * block, deep-copy all layers of KV data, drop one ref on the source, and return
     * the new id. If exclusive, return {@code blockId} unchanged.
     */
    public int cowIfNeeded(int blockId) {
        lock.lock();
        try {
            ensureOpen();
            checkBlockId(blockId);
            if (refCount[blockId] <= 1) {
                touch(blockId);
                return blockId;
            }
            int nb = takeFreeUnlocked();
            Tensor pool = requirePool();
            // copy full multi-layer slot: pool[block] -> pool[nb]
            pool.select(0, nb).copy_(pool.select(0, blockId));
            dirty[nb] = false;
            releaseUnlocked(blockId);
            return nb;
        } finally { lock.unlock(); }
    }

    // ---- tensor views -----------------------------------------------------

    /**
     * Ensure the backing pool tensor is ready. No-op after construction
     * (pool is allocated eagerly); still validates the manager is open.
     */
    public void ensureAllocated() {
        lock.lock();
        try {
            ensureOpen();
            requirePool();
        } finally { lock.unlock(); }
    }

    /**
     * View of one physical block across all layers:
     * {@code [numLayers, 2, blockSize, numHeads, headDim]}.
     */
    public Tensor getBlock(int blockId) {
        lock.lock();
        try {
            ensureOpen();
            checkBlockId(blockId);
            touch(blockId);
            return requirePool().select(0, blockId);
        } finally { lock.unlock(); }
    }

    /**
     * View of one layer of one block: {@code [2, blockSize, numHeads, headDim]}.
     */
    public Tensor getBlockLayer(int blockId, int layer) {
        lock.lock();
        try {
            ensureOpen();
            checkBlockId(blockId);
            if (layer < 0 || layer >= numLayers) {
                throw new IllegalArgumentException("layer out of range: " + layer);
            }
            touch(blockId);
            return requirePool().select(0, blockId).select(0, layer);
        } finally { lock.unlock(); }
    }

    /**
     * Write one token's K and V into {@code (blockId, layer, pos)}.
     * {@code k}/{@code v} must be {@code [numHeads, headDim]} or {@code [1, numHeads, headDim]}.
     */
    public void writeToken(int blockId, int layer, int pos, Tensor k, Tensor v) {
        lock.lock();
        try {
            ensureOpen();
            checkBlockId(blockId);
            if (layer < 0 || layer >= numLayers) throw new IllegalArgumentException("layer");
            if (pos < 0 || pos >= blockSize) throw new IllegalArgumentException("pos");
            Tensor slot = requirePool().select(0, blockId).select(0, layer); // [2, B, H, D]
            Tensor ks = squeezeKV(k);
            Tensor vs = squeezeKV(v);
            slot.select(0, 0).select(0, pos).copy_(ks);
            slot.select(0, 1).select(0, pos).copy_(vs);
            dirty[blockId] = false;
            touch(blockId);
        } finally { lock.unlock(); }
    }

    /**
     * Gather tokens {@code [0, length)} from a block table into contiguous K/V
     * shaped {@code [length, numHeads, headDim]} for one layer.
     */
    public Tensor[] gather(List<Integer> blockTable, int layer, int length) {
        Objects.requireNonNull(blockTable, "blockTable");
        if (length < 0) throw new IllegalArgumentException("length");
        lock.lock();
        try {
            ensureOpen();
            if (layer < 0 || layer >= numLayers) throw new IllegalArgumentException("layer");
            Tensor pool = requirePool();
            if (length == 0) {
                Tensor ek = torch.zeros(new long[]{0, numHeads, headDim}, options);
                Tensor ev = torch.zeros(new long[]{0, numHeads, headDim}, options);
                return new Tensor[]{ek, ev};
            }
            int needed = (length + blockSize - 1) / blockSize;
            if (blockTable.size() < needed) {
                throw new IllegalArgumentException("blockTable too short: have "
                        + blockTable.size() + " need " + needed);
            }
            List<Tensor> ks = new ArrayList<>(needed);
            List<Tensor> vs = new ArrayList<>(needed);
            int remaining = length;
            for (int i = 0; i < needed; i++) {
                int b = blockTable.get(i);
                checkBlockId(b);
                int take = Math.min(blockSize, remaining);
                Tensor slot = pool.select(0, b).select(0, layer);
                ks.add(slot.select(0, 0).narrow(0, 0, take));
                vs.add(slot.select(0, 1).narrow(0, 0, take));
                touch(b);
                remaining -= take;
            }
            Tensor k = torch.cat(new org.bytedeco.pytorch.TensorVector(ks.toArray(new Tensor[0])), 0);
            Tensor v = torch.cat(new org.bytedeco.pytorch.TensorVector(vs.toArray(new Tensor[0])), 0);
            return new Tensor[]{k, v};
        } finally { lock.unlock(); }
    }

    /** Snapshot of LRU timestamps (copy); index = blockId. */
    public long[] lastUseSnapshot() {
        lock.lock();
        try { return Arrays.copyOf(lastUseNs, lastUseNs.length); }
        finally { lock.unlock(); }
    }

    // ---- lifecycle --------------------------------------------------------

    @Override
    public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            if (blockPool != null) {
                try { blockPool.close(); } catch (Throwable ignored) {}
                blockPool = null;
            }
            Arrays.fill(refCount, 0);
            Arrays.fill(inFreeList, true);
            freeList.clear();
            for (int i = 0; i < maxBlocks; i++) freeList.addLast(i);
            freeNotEmpty.signalAll();
        } finally { lock.unlock(); }
    }

    public boolean isClosed() {
        lock.lock();
        try { return closed; }
        finally { lock.unlock(); }
    }

    // ---- internals --------------------------------------------------------

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("PagedBlockManager closed");
    }

    private void checkBlockId(int blockId) {
        if (blockId < 0 || blockId >= maxBlocks) {
            throw new IllegalArgumentException("blockId out of range: " + blockId);
        }
    }

    private void touch(int b) {
        lastUseNs[b] = System.nanoTime();
    }

    private int takeFreeUnlocked() {
        int b = freeList.removeFirst();
        inFreeList[b] = false;
        refCount[b] = 1;
        touch(b);
        // blockPool is non-null while open (eager ctor alloc); zero only if dirty.
        if (dirty[b]) {
            requirePool().select(0, b).zero_();
            dirty[b] = false;
        }
        allocCount.increment();
        return b;
    }

    private void releaseUnlocked(int b) {
        if (refCount[b] <= 0) return;
        refCount[b]--;
        if (refCount[b] == 0) {
            if (!inFreeList[b]) {
                freeList.addLast(b);
                inFreeList[b] = true;
                dirty[b] = true;
                freeCount.increment();
                freeNotEmpty.signal();
            }
        }
    }

    private Tensor requirePool() {
        Tensor p = blockPool;
        if (p == null) {
            throw new IllegalStateException("PagedBlockManager pool not available (closed?)");
        }
        return p;
    }

    static Tensor squeezeKV(Tensor t) {
        if (t.dim() == 3) return t.squeeze(0);
        if (t.dim() == 2) return t;
        throw new IllegalArgumentException("K/V must be [H,D] or [1,H,D], got dim=" + t.dim());
    }

    @Override
    public String toString() {
        return "PagedBlockManager{maxBlocks=" + maxBlocks
                + ", layers=" + numLayers
                + ", blockSize=" + blockSize
                + ", heads=" + numHeads
                + ", headDim=" + headDim
                + ", free=" + freeBlocks()
                + ", closed=" + closed + "}";
    }
}
