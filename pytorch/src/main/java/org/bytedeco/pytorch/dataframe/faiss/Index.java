package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Abstract FAISS-like index — mirrors {@code faiss.Index}.
 *
 * <p>API surface follows Python FAISS used in {@code org/lance/ipc/faiss.md}:
 * {@code train}/{@code add}/{@code add_with_ids}/{@code search}/{@code range_search}/
 * {@code remove_ids}/{@code reconstruct}.
 */
public abstract class Index implements java.io.Serializable {
    private static final long serialVersionUID = 1L;

    /** Vector dimension. */
    public final int d;
    /** Number of vectors currently stored. */
    protected long ntotal;
    /** Whether {@link #train(float[], int)} has been called (or not required). */
    protected boolean is_trained;
    /** Distance metric. */
    public MetricType metric_type;
    /** Verbose flag (FAISS parity). */
    public boolean verbose;

    /** Whether this index holds data on a CUDA device (semantic GPU index). */
    protected transient boolean onGpu;
    protected transient int gpuDevice = -1;

    protected Index(int d, MetricType metric) {
        if (d <= 0) throw new IllegalArgumentException("d must be > 0");
        this.d = d;
        this.metric_type = metric == null ? MetricType.METRIC_L2 : metric;
        this.ntotal = 0;
        this.is_trained = false;
        this.verbose = false;
        this.onGpu = false;
    }

    public long ntotal() { return ntotal; }
    public boolean is_trained() { return is_trained; }
    public boolean is_gpu() { return onGpu; }
    public int gpu_device() { return gpuDevice; }

    /** Mark trained (for indexes that need no train, call in ctor / IO loaders). */
    public void setTrained(boolean v) { this.is_trained = v; }

    /** Package/IO helper: set ntotal after bulk load. */
    void setNtotal(long n) { this.ntotal = n; }

    // ---- train / add ----

    /**
     * Train quantizer / codebooks. No-op for Flat / HNSW.
     * @param x row-major {@code [n * d]} float32
     * @param n number of training vectors
     */
    public void train(float[] x, int n) {
        // default: nothing
        is_trained = true;
    }

    public void train(float[][] rows) {
        if (rows == null || rows.length == 0) { is_trained = true; return; }
        train(pack(rows), rows.length);
    }

    /** Add vectors without ids (sequential ids 0..n-1 relative to current ntotal). */
    public abstract void add(float[] x, int n);

    public void add(float[][] rows) {
        if (rows == null || rows.length == 0) return;
        add(pack(rows), rows.length);
    }

    /**
     * Add with external ids. Default throws — subclasses that support it override,
     * or wrap with {@link IndexIDMap}.
     */
    public void add_with_ids(float[] x, int n, long[] ids) {
        throw new UnsupportedOperationException(
            getClass().getSimpleName() + " does not support add_with_ids; wrap with IndexIDMap");
    }

    public void add_with_ids(float[][] rows, long[] ids) {
        if (rows == null || rows.length == 0) return;
        add_with_ids(pack(rows), rows.length, ids);
    }

    // ---- search ----

    /**
     * k-NN search.
     * @param xq row-major queries {@code [nq * d]}
     * @param nq number of queries
     * @param k top-k
     * @return {@link SearchResult} with {@code D[nq][k]}, {@code I[nq][k]}
     */
    public abstract SearchResult search(float[] xq, int nq, int k);

    /** Single query convenience. */
    public SearchResult search(float[] query, int k) {
        if (query == null || query.length != d)
            throw new IllegalArgumentException("query dim mismatch");
        return search(query, 1, k);
    }

    /** 2-D queries convenience. */
    public SearchResult search(float[][] queries, int k) {
        if (queries == null || queries.length == 0)
            return new SearchResult(new float[0][k], new long[0][k]);
        return search(pack(queries), queries.length, k);
    }

    /**
     * Range search — return all neighbors within radius.
     * For L2: distance &lt;= radius; for IP: score &gt;= radius.
     */
    public RangeSearchResult range_search(float[] xq, int nq, float radius) {
        throw new UnsupportedOperationException(
            getClass().getSimpleName() + " does not support range_search");
    }

    public RangeSearchResult range_search(float[] query, float radius) {
        return range_search(query, 1, radius);
    }

    // ---- mutate / reconstruct ----

    /**
     * Remove vectors whose id is selected. Returns number removed.
     * Default: unsupported.
     */
    public long remove_ids(IDSelector sel) {
        throw new UnsupportedOperationException(
            getClass().getSimpleName() + " does not support remove_ids");
    }

    /** Reconstruct vector at storage position {@code key} into {@code recons[0..d)}. */
    public void reconstruct(long key, float[] recons) {
        throw new UnsupportedOperationException(
            getClass().getSimpleName() + " does not support reconstruct");
    }

    public float[] reconstruct(long key) {
        float[] out = new float[d];
        reconstruct(key, out);
        return out;
    }

    /** Reset to empty. */
    public void reset() {
        ntotal = 0;
    }

    // ---- GPU residency (semantic) ----

    protected void markGpu(int device) {
        this.onGpu = true;
        this.gpuDevice = device;
    }

    protected void markCpu() {
        this.onGpu = false;
        this.gpuDevice = -1;
    }

    /** Subclasses may release GPU caches. */
    public void to_cpu_storage() { markCpu(); }

    /** Subclasses may upload vectors to CUDA tensors. */
    public void to_gpu_storage(int device) { markGpu(device); }

    // ---- helpers ----

    public static float[] pack(float[][] rows) {
        if (rows == null || rows.length == 0) return new float[0];
        int n = rows.length;
        int dim = rows[0].length;
        float[] m = new float[n * dim];
        for (int i = 0; i < n; i++) {
            if (rows[i] == null || rows[i].length != dim)
                throw new IllegalArgumentException("ragged row at " + i);
            System.arraycopy(rows[i], 0, m, i * dim, dim);
        }
        return m;
    }

    protected void requireTrained() {
        if (!is_trained)
            throw new IllegalStateException(getClass().getSimpleName() + " is not trained");
    }

    protected void checkDim(float[] x, int n) {
        if (n < 0) throw new IllegalArgumentException("n < 0");
        if (x == null || x.length < (long) n * d)
            throw new IllegalArgumentException("x too small for n=" + n + " d=" + d);
    }

    /** Type tag for IO. */
    public abstract String indexType();
}
