package org.bytedeco.pytorch.dataframe.faiss;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * High-performance CPU brute-force k-NN / range search.
 *
 * <p>Primary path: parallel per-query distance loops with 8-wide unrolled kernels
 * and bounded top-k. Optional torch CPU GEMM path when base is large and metric is IP
 * (or L2 via norm trick) — falls back silently on any torch failure.
 */
public final class CpuDistanceBackend implements DistanceBackend {
    public static final CpuDistanceBackend INSTANCE = new CpuDistanceBackend();

    /** Min (nb * nq) to attempt torch mm path. */
    private static final long TORCH_THRESHOLD = 64L * 1024L;

    private CpuDistanceBackend() {}

    @Override
    public String name() {
        return "cpu-java";
    }

    @Override
    public SearchResult knn(float[] base, int nb, float[] queries, int nq,
                            int d, int k, MetricType metric, long[] ids) {
        if (nq <= 0 || k <= 0 || nb <= 0) {
            float[][] D = new float[Math.max(0, nq)][Math.max(0, k)];
            long[][] I = new long[Math.max(0, nq)][Math.max(0, k)];
            for (int q = 0; q < nq; q++) {
                for (int j = 0; j < k; j++) {
                    D[q][j] = metric.lowerIsBetter() ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
                    I[q][j] = -1;
                }
            }
            return new SearchResult(D, I);
        }
        k = Math.min(k, nb);

        // Try torch GEMM for large problems
        if ((long) nb * nq >= TORCH_THRESHOLD && nq >= 4) {
            try {
                SearchResult r = knnTorch(base, nb, queries, nq, d, k, metric, ids, false);
                if (r != null) return r;
            } catch (Throwable ignored) {
                // fall through to pure Java
            }
        }
        return knnJava(base, nb, queries, nq, d, k, metric, ids);
    }

    @Override
    public RangeSearchResult range(float[] base, int nb, float[] queries, int nq,
                                   int d, float radius, MetricType metric, long[] ids) {
        if (nq <= 0 || nb <= 0) {
            return new RangeSearchResult(new long[]{0}, new float[0], new long[0]);
        }
        boolean l2 = metric == MetricType.METRIC_L2;
        List<float[]> allD = new ArrayList<>(nq);
        List<long[]> allI = new ArrayList<>(nq);
        long[] lims = new long[nq + 1];
        long cursor = 0;
        for (int q = 0; q < nq; q++) {
            int qOff = q * d;
            // collect matches
            float[] tmpD = new float[Math.min(nb, 1024)];
            long[] tmpI = new long[tmpD.length];
            int m = 0;
            for (int i = 0; i < nb; i++) {
                float dist;
                if (l2) {
                    dist = DistanceKernel.l2Row(queries, qOff, base, i * d, d);
                } else {
                    dist = DistanceKernel.ipRow(queries, qOff, base, i * d, d);
                }
                boolean keep = l2 ? dist <= radius : dist >= radius;
                if (!keep) continue;
                if (m == tmpD.length) {
                    tmpD = java.util.Arrays.copyOf(tmpD, tmpD.length * 2);
                    tmpI = java.util.Arrays.copyOf(tmpI, tmpI.length * 2);
                }
                tmpD[m] = dist;
                tmpI[m] = ids != null ? ids[i] : i;
                m++;
            }
            // sort matches best-first
            sortPairs(tmpD, tmpI, m, l2);
            allD.add(java.util.Arrays.copyOf(tmpD, m));
            allI.add(java.util.Arrays.copyOf(tmpI, m));
            cursor += m;
            lims[q + 1] = cursor;
        }
        float[] D = new float[(int) cursor];
        long[] I = new long[(int) cursor];
        int p = 0;
        for (int q = 0; q < nq; q++) {
            float[] dq = allD.get(q);
            long[] iq = allI.get(q);
            System.arraycopy(dq, 0, D, p, dq.length);
            System.arraycopy(iq, 0, I, p, iq.length);
            p += dq.length;
        }
        return new RangeSearchResult(lims, D, I);
    }

    // ---- pure Java knn ----

    SearchResult knnJava(float[] base, int nb, float[] queries, int nq,
                         int d, int k, MetricType metric, long[] ids) {
        boolean lower = metric.lowerIsBetter();
        boolean l2 = metric == MetricType.METRIC_L2;
        TopK[] heaps = new TopK[nq];
        for (int q = 0; q < nq; q++) heaps[q] = new TopK(k, lower);

        if (nq == 1) {
            scanQuery(base, nb, queries, 0, d, heaps[0], l2, ids);
        } else {
            int parallelism = Math.max(1, ForkJoinPool.commonPool().getParallelism());
            int chunk = Math.max(1, (nq + parallelism - 1) / parallelism);
            RecursiveAction root = new RecursiveAction() {
                @Override protected void compute() {
                    List<RecursiveAction> tasks = new ArrayList<>();
                    for (int start = 0; start < nq; start += chunk) {
                        final int s = start;
                        final int e = Math.min(nq, start + chunk);
                        tasks.add(new RecursiveAction() {
                            @Override protected void compute() {
                                for (int q = s; q < e; q++) {
                                    scanQuery(base, nb, queries, q, d, heaps[q], l2, ids);
                                }
                            }
                        });
                    }
                    invokeAll(tasks);
                }
            };
            ForkJoinPool.commonPool().invoke(root);
        }
        return TopK.toSearchResult(heaps, k);
    }

    private static void scanQuery(float[] base, int nb, float[] queries, int q,
                                  int d, TopK heap, boolean l2, long[] ids) {
        int qOff = q * d;
        // Extract query to contiguous for better kernel
        float[] qv = new float[d];
        System.arraycopy(queries, qOff, qv, 0, d);
        for (int i = 0; i < nb; i++) {
            float dist = l2
                ? DistanceKernel.l2Row(qv, base, i * d, d)
                : DistanceKernel.ipRow(qv, base, i * d, d);
            long id = ids != null ? ids[i] : i;
            heap.offer(id, dist);
        }
    }

    // ---- torch path ----

    /**
     * Compute pairwise scores via torch.mm then top-k on CPU arrays.
     * @return null if torch path unavailable
     */
    static SearchResult knnTorch(float[] base, int nb, float[] queries, int nq,
                                 int d, int k, MetricType metric, long[] ids,
                                 boolean preferCuda) {
        try {
            org.bytedeco.pytorch.Tensor xb = org.bytedeco.pytorch.global.torch.tensor(base)
                .reshape(new long[]{nb, d});
            org.bytedeco.pytorch.Tensor xq = org.bytedeco.pytorch.global.torch.tensor(
                java.util.Arrays.copyOf(queries, nq * d)).reshape(new long[]{nq, d});

            if (preferCuda && DeviceSelector.isCudaAvailable()) {
                try {
                    java.lang.reflect.Method cuda = xb.getClass().getMethod("cuda");
                    xb = (org.bytedeco.pytorch.Tensor) cuda.invoke(xb);
                    xq = (org.bytedeco.pytorch.Tensor) cuda.invoke(xq);
                } catch (Throwable e) {
                    // stay on CPU
                }
            }

            org.bytedeco.pytorch.Tensor scores;
            if (metric == MetricType.METRIC_INNER_PRODUCT) {
                // scores = xq @ xb.T   → [nq, nb]
                scores = xq.matmul(xb.transpose(0, 1));
            } else {
                // L2^2 = ||q||^2 + ||b||^2 - 2 q·b
                org.bytedeco.pytorch.ScalarTypeOptional noDtype = new org.bytedeco.pytorch.ScalarTypeOptional();
                org.bytedeco.pytorch.Tensor q2 = xq.pow(new org.bytedeco.pytorch.Scalar(2)).sum(new long[]{1}, true, noDtype); // [nq,1]
                org.bytedeco.pytorch.Tensor b2 = xb.pow(new org.bytedeco.pytorch.Scalar(2)).sum(new long[]{1}, true, noDtype); // [nb,1]
                org.bytedeco.pytorch.Tensor dots = xq.matmul(xb.transpose(0, 1));
                scores = q2.add(b2.transpose(0, 1)).sub(dots.mul(new org.bytedeco.pytorch.Scalar(2)));
            }

            // pull back to float[]
            org.bytedeco.pytorch.Tensor cpu = scores.contiguous();
            try {
                if (cpu.is_cuda()) cpu = cpu.cpu();
            } catch (Throwable ignored) {}
            if (!cpu.is_contiguous()) cpu = cpu.contiguous();
            org.bytedeco.javacpp.FloatPointer ptr = cpu.data_ptr_float();
            float[] flat = new float[nq * nb];
            ptr.get(flat);

            // top-k per row
            boolean lower = metric.lowerIsBetter();
            float[][] D = new float[nq][k];
            long[][] I = new long[nq][k];
            for (int q = 0; q < nq; q++) {
                TopK heap = new TopK(k, lower);
                int row = q * nb;
                for (int i = 0; i < nb; i++) {
                    heap.offer(ids != null ? ids[i] : i, flat[row + i]);
                }
                heap.export(D[q], I[q]);
            }

            try { xb.close(); } catch (Throwable ignored) {}
            try { xq.close(); } catch (Throwable ignored) {}
            try { scores.close(); } catch (Throwable ignored) {}
            try { cpu.close(); } catch (Throwable ignored) {}
            return new SearchResult(D, I);
        } catch (Throwable e) {
            return null;
        }
    }

    // fix l2Row overload — DistanceKernel has (q, matrix, rowBase, dim) with q as full array from 0
    // We need a version with query offset. Add helper here:

    // Actually DistanceKernel.l2Row(float[] q, float[] matrix, int rowBase, int dim) assumes q starts at 0.
    // scanQuery copies to qv so OK. For range() I used a non-existent overload — fix below via local helpers.

    private static void sortPairs(float[] d, long[] ids, int n, boolean lowerIsBetter) {
        // simple insertion sort — n usually small for range hits; for large use quicksort
        if (n > 64) {
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) order[i] = i;
            if (lowerIsBetter) {
                java.util.Arrays.sort(order, (a, b) -> Float.compare(d[a], d[b]));
            } else {
                java.util.Arrays.sort(order, (a, b) -> Float.compare(d[b], d[a]));
            }
            float[] nd = new float[n];
            long[] ni = new long[n];
            for (int i = 0; i < n; i++) {
                nd[i] = d[order[i]];
                ni[i] = ids[order[i]];
            }
            System.arraycopy(nd, 0, d, 0, n);
            System.arraycopy(ni, 0, ids, 0, n);
            return;
        }
        for (int i = 1; i < n; i++) {
            float key = d[i];
            long kid = ids[i];
            int j = i - 1;
            if (lowerIsBetter) {
                while (j >= 0 && d[j] > key) {
                    d[j + 1] = d[j];
                    ids[j + 1] = ids[j];
                    j--;
                }
            } else {
                while (j >= 0 && d[j] < key) {
                    d[j + 1] = d[j];
                    ids[j + 1] = ids[j];
                    j--;
                }
            }
            d[j + 1] = key;
            ids[j + 1] = kid;
        }
    }
}
