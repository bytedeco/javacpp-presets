package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Flat (brute-force) index — mirrors {@code faiss.IndexFlat}, {@code IndexFlatL2},
 * {@code IndexFlatIP}.
 *
 * <pre>
 *   IndexFlat index = new IndexFlatL2(d);
 *   // or: new IndexFlat(d, MetricType.METRIC_INNER_PRODUCT);
 *   index.add(base_vecs);
 *   SearchResult r = index.search(query_vecs, k);
 * </pre>
 *
 * <p>Search uses {@link DistanceBackend} selected by {@link DeviceSelector}
 * (CUDA torch matmul when available, else parallel CPU kernels).
 */
public class IndexFlat extends Index {
    private static final long serialVersionUID = 1L;

    /** Row-major storage [capacity * d]. */
    protected float[] xb;
    protected int capacity;
    /** Optional per-vector ids (null → positional). Used only when subclass sets them. */
    protected long[] ids;
    protected boolean hasIds;

    /** Cached squared norms for L2 acceleration (lazy). */
    protected transient float[] cachedSqNorms;
    protected transient boolean normsValid;

    /** Optional resident GPU tensor (semantic GpuIndexFlat). */
    protected transient Object gpuTensor; // org.bytedeco.pytorch.Tensor, avoid hard fail if missing

    public IndexFlat(int d) {
        this(d, MetricType.METRIC_L2);
    }

    public IndexFlat(int d, MetricType metric) {
        super(d, metric);
        this.capacity = 0;
        this.xb = new float[0];
        this.is_trained = true; // Flat needs no train
        this.hasIds = false;
    }

    /** Factory matching {@code faiss.IndexFlatL2(d)}. */
    public static IndexFlatL2 IndexFlatL2(int d) {
        return new IndexFlatL2(d);
    }

    /** Factory matching {@code faiss.IndexFlatIP(d)}. */
    public static IndexFlatIP IndexFlatIP(int d) {
        return new IndexFlatIP(d);
    }

    @Override
    public String indexType() {
        return metric_type == MetricType.METRIC_INNER_PRODUCT ? "FlatIP" : "FlatL2";
    }

    @Override
    public synchronized void add(float[] x, int n) {
        if (n <= 0) return;
        checkDim(x, n);
        ensureCapacity((int) ntotal + n);
        System.arraycopy(x, 0, xb, (int) ntotal * d, n * d);
        ntotal += n;
        normsValid = false;
        // keep GPU cache in sync lazily — invalidate
        releaseGpuTensor();
    }

    @Override
    public synchronized void add_with_ids(float[] x, int n, long[] externalIds) {
        if (n <= 0) return;
        checkDim(x, n);
        if (externalIds == null || externalIds.length < n)
            throw new IllegalArgumentException("ids length < n");
        ensureCapacity((int) ntotal + n);
        if (ids == null) {
            ids = new long[capacity];
            // backfill positional for existing
            for (int i = 0; i < ntotal; i++) ids[i] = i;
            hasIds = true;
        }
        System.arraycopy(x, 0, xb, (int) ntotal * d, n * d);
        for (int i = 0; i < n; i++) ids[(int) ntotal + i] = externalIds[i];
        ntotal += n;
        normsValid = false;
        releaseGpuTensor();
    }

    @Override
    public SearchResult search(float[] xq, int nq, int k) {
        if (nq <= 0 || k <= 0)
            return emptyResult(nq, k);
        checkDim(xq, nq);
        if (ntotal == 0) return emptyResult(nq, k);
        k = (int) Math.min(k, ntotal);
        DistanceBackend backend = selectBackend();
        return backend.knn(xb, (int) ntotal, xq, nq, d, k, metric_type, hasIds ? ids : null);
    }

    @Override
    public RangeSearchResult range_search(float[] xq, int nq, float radius) {
        if (nq <= 0) return new RangeSearchResult(new long[]{0}, new float[0], new long[0]);
        checkDim(xq, nq);
        if (ntotal == 0)
            return new RangeSearchResult(new long[nq + 1], new float[0], new long[0]);
        DistanceBackend backend = selectBackend();
        return backend.range(xb, (int) ntotal, xq, nq, d, radius, metric_type, hasIds ? ids : null);
    }

    @Override
    public synchronized long remove_ids(IDSelector sel) {
        if (sel == null || ntotal == 0) return 0;
        int n = (int) ntotal;
        int w = 0;
        for (int r = 0; r < n; r++) {
            long id = hasIds && ids != null ? ids[r] : r;
            if (sel.is_member(id)) continue; // drop
            if (w != r) {
                System.arraycopy(xb, r * d, xb, w * d, d);
                if (hasIds && ids != null) ids[w] = ids[r];
            }
            w++;
        }
        long removed = n - w;
        ntotal = w;
        normsValid = false;
        releaseGpuTensor();
        return removed;
    }

    @Override
    public void reconstruct(long key, float[] recons) {
        if (key < 0 || key >= ntotal)
            throw new IllegalArgumentException("reconstruct key out of range: " + key);
        if (recons == null || recons.length < d)
            throw new IllegalArgumentException("recons too small");
        System.arraycopy(xb, (int) key * d, recons, 0, d);
    }

    @Override
    public synchronized void reset() {
        ntotal = 0;
        normsValid = false;
        releaseGpuTensor();
    }

    /** Raw storage access (row-major). Length may exceed ntotal*d (capacity). */
    public float[] getXb() { return xb; }

    public int capacity() { return capacity; }

    // ---- GPU residency ----

    @Override
    public synchronized void to_gpu_storage(int device) {
        if (!DeviceSelector.isCudaAvailable()) {
            markCpu();
            return;
        }
        try {
            org.bytedeco.pytorch.Tensor t = org.bytedeco.pytorch.global.torch
                .tensor(java.util.Arrays.copyOf(xb, (int) ntotal * d))
                .reshape(new long[]{ntotal, d});
            java.lang.reflect.Method cuda = t.getClass().getMethod("cuda");
            Object g = cuda.invoke(t);
            try { t.close(); } catch (Throwable ignored) {}
            releaseGpuTensor();
            gpuTensor = g;
            markGpu(device);
        } catch (Throwable e) {
            markCpu();
        }
    }

    @Override
    public synchronized void to_cpu_storage() {
        // If GPU tensor holds newer data we would pull back; Flat always mutates xb on CPU first.
        releaseGpuTensor();
        markCpu();
    }

    private void releaseGpuTensor() {
        if (gpuTensor instanceof AutoCloseable ac) {
            try { ac.close(); } catch (Exception ignored) {}
        }
        gpuTensor = null;
    }

    private DistanceBackend selectBackend() {
        if (onGpu || DeviceSelector.resolve() == DeviceSelector.Device.CUDA) {
            return CudaDistanceBackend.INSTANCE;
        }
        return CpuDistanceBackend.INSTANCE;
    }

    private void ensureCapacity(int need) {
        if (need <= capacity) return;
        int nc = Math.max(Math.max(16, capacity * 2), need);
        xb = java.util.Arrays.copyOf(xb, nc * d);
        if (ids != null) ids = java.util.Arrays.copyOf(ids, nc);
        capacity = nc;
    }

    private SearchResult emptyResult(int nq, int k) {
        nq = Math.max(0, nq);
        k = Math.max(0, k);
        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];
        float fill = metric_type.lowerIsBetter() ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        for (int q = 0; q < nq; q++) {
            for (int j = 0; j < k; j++) {
                D[q][j] = fill;
                I[q][j] = -1;
            }
        }
        return new SearchResult(D, I);
    }
}
