package org.bytedeco.pytorch.dataframe.faiss;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Multi-shard index — mirrors {@code faiss.IndexShards}.
 *
 * <pre>
 *   IndexShards shards = new IndexShards(d);
 *   shards.add_shard(shard1);
 *   shards.add_shard(shard2);
 *   shards.set_nthreads(4);
 *   SearchResult r = shards.search(queries, k);
 * </pre>
 *
 * <p>Each shard is searched (optionally in parallel); results are merged per query
 * to global top-k. Metric is taken from the first shard.
 */
public class IndexShards extends Index {
    private static final long serialVersionUID = 1L;

    private final List<Index> shards = new ArrayList<>();
    private int nthreads = 1;

    public IndexShards(int d) {
        super(d, MetricType.METRIC_L2);
        this.is_trained = true;
    }

    public IndexShards(int d, MetricType metric) {
        super(d, metric);
        this.is_trained = true;
    }

    @Override
    public String indexType() {
        return "Shards(" + shards.size() + ")";
    }

    public void add_shard(Index shard) {
        if (shard == null) throw new IllegalArgumentException("null shard");
        if (shard.d != d) throw new IllegalArgumentException("shard dim mismatch");
        if (shards.isEmpty()) {
            this.metric_type = shard.metric_type;
        }
        shards.add(shard);
        recomputeNtotal();
    }

    public int nshard() { return shards.size(); }

    public Index shard(int i) { return shards.get(i); }

    /** FAISS {@code set_threads} / thread count for parallel shard search. */
    public void set_nthreads(int n) {
        this.nthreads = Math.max(1, n);
    }

    public void set_threads(int n) { set_nthreads(n); }

    public int get_nthreads() { return nthreads; }

    private void recomputeNtotal() {
        long t = 0;
        for (Index s : shards) t += s.ntotal();
        this.ntotal = t;
    }

    @Override
    public void train(float[] x, int n) {
        for (Index s : shards) s.train(x, n);
        is_trained = true;
    }

    @Override
    public void add(float[] x, int n) {
        throw new UnsupportedOperationException(
            "IndexShards.add not supported; add to individual shards or use add_shard with prebuilt indexes");
    }

    @Override
    public SearchResult search(float[] xq, int nq, int k) {
        if (shards.isEmpty() || nq <= 0 || k <= 0) {
            return empty(nq, k);
        }
        checkDim(xq, nq);
        recomputeNtotal();

        final int ns = shards.size();
        SearchResult[] partial = new SearchResult[ns];

        if (nthreads <= 1 || ns == 1) {
            for (int i = 0; i < ns; i++) {
                partial[i] = shards.get(i).search(xq, nq, k);
            }
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(Math.min(nthreads, ns));
            try {
                List<Future<SearchResult>> futures = new ArrayList<>(ns);
                for (int i = 0; i < ns; i++) {
                    final int si = i;
                    futures.add(pool.submit(() -> shards.get(si).search(xq, nq, k)));
                }
                for (int i = 0; i < ns; i++) {
                    partial[i] = futures.get(i).get();
                }
            } catch (Exception e) {
                throw new RuntimeException("IndexShards parallel search failed", e);
            } finally {
                pool.shutdown();
            }
        }

        return merge(partial, nq, k, metric_type.lowerIsBetter());
    }

    /** Manual multi-shard merge (also used by benchmark example 4). */
    public static SearchResult merge(SearchResult[] partial, int nq, int k, boolean lowerIsBetter) {
        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];
        float fill = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        for (int q = 0; q < nq; q++) {
            TopK heap = new TopK(k, lowerIsBetter);
            for (SearchResult sr : partial) {
                if (sr == null || q >= sr.nq()) continue;
                int kk = sr.k();
                for (int j = 0; j < kk; j++) {
                    if (sr.I[q][j] < 0) continue;
                    heap.offer(sr.I[q][j], sr.D[q][j]);
                }
            }
            heap.export(D[q], I[q]);
            // ensure fill for empty
            for (int j = 0; j < k; j++) {
                if (I[q][j] < 0) D[q][j] = fill;
            }
        }
        return new SearchResult(D, I);
    }

    @Override
    public synchronized void reset() {
        for (Index s : shards) s.reset();
        ntotal = 0;
    }

    @Override
    public void to_gpu_storage(int device) {
        for (Index s : shards) s.to_gpu_storage(device);
        markGpu(device);
    }

    @Override
    public void to_cpu_storage() {
        for (Index s : shards) s.to_cpu_storage();
        markCpu();
    }

    private SearchResult empty(int nq, int k) {
        nq = Math.max(0, nq); k = Math.max(0, k);
        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];
        float fill = metric_type.lowerIsBetter() ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        for (int q = 0; q < nq; q++) {
            for (int j = 0; j < k; j++) { D[q][j] = fill; I[q][j] = -1; }
        }
        return new SearchResult(D, I);
    }
}
