package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Pluggable bulk distance / top-k backend (CPU pure-Java, torch CPU, torch CUDA).
 */
public interface DistanceBackend {

    /** Human-readable name for benchmarks. */
    String name();

    /**
     * Brute-force k-NN of queries against a row-major base matrix.
     *
     * @param base   row-major [nb * d]
     * @param nb     number of base vectors
     * @param queries row-major [nq * d]
     * @param nq     number of queries
     * @param d      dimension
     * @param k      top-k
     * @param metric L2 or IP
     * @param ids    optional external ids length nb; null → use 0..nb-1
     */
    SearchResult knn(float[] base, int nb,
                     float[] queries, int nq,
                     int d, int k,
                     MetricType metric,
                     long[] ids);

    /**
     * Range search against base.
     * L2: keep dist &lt;= radius; IP: keep score &gt;= radius.
     */
    RangeSearchResult range(float[] base, int nb,
                            float[] queries, int nq,
                            int d, float radius,
                            MetricType metric,
                            long[] ids);
}
