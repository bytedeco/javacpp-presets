package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Contiguous (non-paged) per-session KV buffer cache.
 *
 * <p>Layout per session: {@code [numLayers, 2, contextLength, kvWidth]} where
 * axis-1 is K/V and {@code kvWidth} is typically {@code numHeads * headDim} or
 * {@code numKvHeads * headDim} for GQA.
 *
 * <p>This is the simple dense alternative to {@link PagedKvCache} /
 * {@link PagedKvBuffer}: fixed max context, no block tables, O(1) slice for
 * attention. Prefer paged variants under multi-tenant / long-context pressure.
 *
 * <p>Thread-safe: session map is concurrent; each buffer serializes appends.
 */
public class KvBufferCache implements AutoCloseable {

    private final ConcurrentMap<String, KvBuffer> buffers = new ConcurrentHashMap<>();
    private final torch.ScalarType scalarType;
    private final int numLayers;
    private final int contextLength;
    private final int kvWidth;
    private final TensorOptions options;
    private volatile boolean closed = false;

    public final LongAdder sessionCreates = new LongAdder();
    public final LongAdder sessionCloses = new LongAdder();
    public final LongAdder appendCount = new LongAdder();

    public KvBufferCache(int numLayers, int contextLength, int kvWidth) {
        this(numLayers, contextLength, kvWidth, torch.kFloat(), null);
    }

    public KvBufferCache(int numLayers, int contextLength, int kvWidth, torch.ScalarType dtype) {
        this(numLayers, contextLength, kvWidth, dtype, null);
    }

    public KvBufferCache(int numLayers, int contextLength, int kvWidth,
                         torch.ScalarType dtype, Device device) {
        if (numLayers <= 0 || contextLength <= 0 || kvWidth <= 0) {
            throw new IllegalArgumentException("size params must be > 0");
        }
        this.numLayers = numLayers;
        this.contextLength = contextLength;
        this.kvWidth = kvWidth;
        this.scalarType = Objects.requireNonNull(dtype, "dtype");
        TensorOptions opts = new TensorOptions(dtype);
        if (device != null) {
            opts = opts.device(new DeviceOptional(device));
        }
        this.options = opts;
        this.closed = false;
    }

    /**
     * Compatibility ctor: {@code (scalarTypeValue, numLayers, contextLength, kvLength)}.
     */
    public KvBufferCache(int scalarTypeValue, int numLayers, int contextLength, int kvLength) {
        this(numLayers, contextLength, kvLength, resolveDtype(scalarTypeValue), null);
    }

    private static torch.ScalarType resolveDtype(int value) {
        for (torch.ScalarType e : torch.ScalarType.values()) {
            if (e.value == value) return e;
        }
        return torch.kFloat();
    }

    public int numLayers() { return numLayers; }
    public int contextLength() { return contextLength; }
    public int kvWidth() { return kvWidth; }
    public int size() { return buffers.size(); }

    public KvBuffer getKvBuffer(String session) {
        Objects.requireNonNull(session, "session");
        ensureOpen();
        return buffers.computeIfAbsent(session, s -> {
            sessionCreates.increment();
            return new KvBuffer(s);
        });
    }

    public boolean contains(String session) {
        return buffers.containsKey(session);
    }

    public void release(String session) {
        KvBuffer buf = buffers.remove(session);
        if (buf != null) {
            buf.close();
            sessionCloses.increment();
        }
    }

    @Override
    public void close() {
        closed = true;
        for (KvBuffer buf : buffers.values()) {
            try { buf.close(); } catch (Throwable ignored) {}
        }
        buffers.clear();
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("KvBufferCache closed");
    }

    /**
     * Dense KV storage for one session.
     * Shape: {@code [numLayers, 2, contextLength, kvWidth]}.
     */
    public final class KvBuffer implements AutoCloseable {
        private final String session;
        private final AtomicInteger currentPos = new AtomicInteger(0);
        private final ReentrantLock lock = new ReentrantLock();
        /** Dense storage; non-null after construction until {@link #close()}. */
        private Tensor fullCache;
        private boolean closed = false;

        KvBuffer(String session) {
            this.session = Objects.requireNonNull(session, "session");
            long[] shape = {numLayers, 2L, contextLength, kvWidth};
            this.fullCache = torch.zeros(shape, options);
            this.closed = false;
            this.currentPos.set(0);
        }

        public String session() { return session; }
        public int getCurrentPosition() { return currentPos.get(); }
        public int remaining() { return contextLength - currentPos.get(); }

        /**
         * View at {@code (layer, kvIndex, position)} → {@code [kvWidth]}.
         * {@code kvIndex}: 0 = Key, 1 = Value.
         */
        public Tensor getTensorAt(int layerIndex, int position, int kvIndex) {
            checkLayer(layerIndex);
            checkKv(kvIndex);
            checkPos(position);
            lock.lock();
            try {
                ensureOpenBuf();
                return requireCache().select(0, layerIndex)
                        .select(0, kvIndex)
                        .select(0, position);
            } finally { lock.unlock(); }
        }

        /**
         * Contiguous history {@code [0, upperBound)} for attention:
         * {@code [upperBound, kvWidth]}.
         */
        public Tensor getTensorsUpTo(int layerIndex, int kvIndex, int upperBound) {
            checkLayer(layerIndex);
            checkKv(kvIndex);
            if (upperBound < 0 || upperBound > contextLength) {
                throw new IllegalArgumentException("upperBound out of range: " + upperBound);
            }
            lock.lock();
            try {
                ensureOpenBuf();
                return requireCache().select(0, layerIndex)
                        .select(0, kvIndex)
                        .slice(0, new LongOptional(0), new LongOptional(upperBound), 1);
            } finally { lock.unlock(); }
        }

        /** History up to current write position. */
        public Tensor getTensorsUpToCurrent(int layerIndex, int kvIndex) {
            return getTensorsUpTo(layerIndex, kvIndex, currentPos.get());
        }

        /**
         * Write one token's K or V at the current position without advancing.
         * Call {@link #incrementPosition()} after both K and V (or all layers) are written.
         */
        public void append(int layerIndex, int kvIndex, Tensor newData) {
            Objects.requireNonNull(newData, "newData");
            lock.lock();
            try {
                ensureOpenBuf();
                int pos = currentPos.get();
                if (pos >= contextLength) {
                    throw new IllegalStateException("KV buffer full: contextLength=" + contextLength);
                }
                requireCache().select(0, layerIndex)
                        .select(0, kvIndex)
                        .select(0, pos)
                        .copy_(squeezeRow(newData));
                appendCount.increment();
            } finally { lock.unlock(); }
        }

        /**
         * Write K and V for one layer at the current position without advancing.
         * {@code k}/{@code v} shaped {@code [kvWidth]} or {@code [1, kvWidth]}.
         */
        public void writeKv(int layerIndex, Tensor k, Tensor v) {
            Objects.requireNonNull(k);
            Objects.requireNonNull(v);
            lock.lock();
            try {
                ensureOpenBuf();
                int pos = currentPos.get();
                if (pos >= contextLength) {
                    throw new IllegalStateException("KV buffer full: contextLength=" + contextLength);
                }
                Tensor row = requireCache().select(0, layerIndex); // [2, ctx, W]
                row.select(0, 0).select(0, pos).copy_(squeezeRow(k));
                row.select(0, 1).select(0, pos).copy_(squeezeRow(v));
                appendCount.increment();
            } finally { lock.unlock(); }
        }

        /**
         * Write all layers' K/V for one token and advance the position.
         * {@code kLayers[i]}/{@code vLayers[i]} shaped {@code [kvWidth]} or {@code [1,kvWidth]}.
         */
        public void appendToken(Tensor[] kLayers, Tensor[] vLayers) {
            Objects.requireNonNull(kLayers);
            Objects.requireNonNull(vLayers);
            if (kLayers.length != numLayers || vLayers.length != numLayers) {
                throw new IllegalArgumentException("layer count mismatch");
            }
            lock.lock();
            try {
                ensureOpenBuf();
                int pos = currentPos.get();
                if (pos >= contextLength) {
                    throw new IllegalStateException("KV buffer full: contextLength=" + contextLength);
                }
                Tensor cache = requireCache();
                for (int layer = 0; layer < numLayers; layer++) {
                    Tensor row = cache.select(0, layer);
                    row.select(0, 0).select(0, pos).copy_(squeezeRow(kLayers[layer]));
                    row.select(0, 1).select(0, pos).copy_(squeezeRow(vLayers[layer]));
                }
                currentPos.incrementAndGet();
                appendCount.increment();
            } finally { lock.unlock(); }
        }

        /** @deprecated buggy arg order in older demos; use {@link #append(int, int, Tensor)}. */
        @Deprecated
        public void append2(int layerIndex, int kvIndex, Tensor newData) {
            // Old code swapped position/kvIndex — keep name but correct behavior.
            append(layerIndex, kvIndex, newData);
        }

        public void incrementPosition() {
            lock.lock();
            try {
                int pos = currentPos.get();
                if (pos >= contextLength) {
                    throw new IllegalStateException("KV buffer full");
                }
                currentPos.incrementAndGet();
            } finally { lock.unlock(); }
        }

        /** Reset write cursor to 0 without reallocating (contents become logical garbage). */
        public void rewind() {
            lock.lock();
            try { currentPos.set(0); }
            finally { lock.unlock(); }
        }

        /** Zero storage and reset cursor. */
        public void clear() {
            lock.lock();
            try {
                ensureOpenBuf();
                requireCache().zero_();
                currentPos.set(0);
            } finally { lock.unlock(); }
        }

        public Tensor raw() {
            lock.lock();
            try {
                ensureOpenBuf();
                return requireCache();
            } finally { lock.unlock(); }
        }

        @Override
        public void close() {
            lock.lock();
            try {
                if (closed) return;
                closed = true;
                if (fullCache != null) {
                    try { fullCache.close(); } catch (Throwable ignored) {}
                    fullCache = null;
                }
            } finally { lock.unlock(); }
        }

        private void ensureOpenBuf() {
            if (closed) {
                throw new IllegalStateException("KvBuffer closed: " + session);
            }
            requireCache();
        }

        private Tensor requireCache() {
            Tensor c = fullCache;
            if (c == null) {
                throw new IllegalStateException("KvBuffer storage missing: " + session);
            }
            return c;
        }

        private void checkLayer(int layerIndex) {
            if (layerIndex < 0 || layerIndex >= numLayers) {
                throw new IllegalArgumentException("layer out of range: " + layerIndex);
            }
        }

        private void checkKv(int kvIndex) {
            if (kvIndex != 0 && kvIndex != 1) {
                throw new IllegalArgumentException("kvIndex must be 0 (K) or 1 (V)");
            }
        }

        private void checkPos(int position) {
            if (position < 0 || position >= contextLength) {
                throw new IllegalArgumentException("position out of range: " + position);
            }
        }

        private static Tensor squeezeRow(Tensor t) {
            if (t.dim() == 1) return t;
            if (t.dim() == 2 && t.size(0) == 1) return t.squeeze(0);
            // Flatten [H,D] -> [H*D] for dense layout convenience
            if (t.dim() == 2) return t.reshape(new long[]{t.numel()});
            if (t.dim() == 3 && t.size(0) == 1) {
                return t.squeeze(0).reshape(new long[]{t.numel() / t.size(0)});
            }
            throw new IllegalArgumentException("expected [W], [1,W], or [H,D]; got dim=" + t.dim());
        }

        @Override
        public String toString() {
            return "KvBuffer{session=" + session
                    + ", pos=" + currentPos.get()
                    + "/" + contextLength
                    + ", closed=" + closed + "}";
        }
    }

    @Override
    public String toString() {
        return "KvBufferCache{layers=" + numLayers
                + ", ctx=" + contextLength
                + ", width=" + kvWidth
                + ", sessions=" + buffers.size()
                + ", dtype=" + scalarType + "}";
    }
}
