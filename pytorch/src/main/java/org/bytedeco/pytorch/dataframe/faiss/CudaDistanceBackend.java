package org.bytedeco.pytorch.dataframe.faiss;

/**
 * CUDA distance backend via javacpp-pytorch Tensor matmul.
 * Falls back to {@link CpuDistanceBackend} if CUDA is unavailable or ops fail.
 */
public final class CudaDistanceBackend implements DistanceBackend {
    public static final CudaDistanceBackend INSTANCE = new CudaDistanceBackend();

    private CudaDistanceBackend() {}

    @Override
    public String name() {
        return DeviceSelector.isCudaAvailable() ? "cuda-torch" : "cuda-fallback-cpu";
    }

    @Override
    public SearchResult knn(float[] base, int nb, float[] queries, int nq,
                            int d, int k, MetricType metric, long[] ids) {
        if (!DeviceSelector.isCudaAvailable()) {
            return CpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids);
        }
        SearchResult r = CpuDistanceBackend.knnTorch(base, nb, queries, nq, d, k, metric, ids, true);
        if (r != null) return r;
        return CpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids);
    }

    @Override
    public RangeSearchResult range(float[] base, int nb, float[] queries, int nq,
                                   int d, float radius, MetricType metric, long[] ids) {
        // Range search stays on CPU (variable-length output; GEMM still possible but top-filter is Java)
        return CpuDistanceBackend.INSTANCE.range(base, nb, queries, nq, d, radius, metric, ids);
    }
}
