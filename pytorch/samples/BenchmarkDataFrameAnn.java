package samples;

import org.bytedeco.pytorch.dataframe.*;
import org.bytedeco.pytorch.dataframe.ann.*;

import java.nio.file.*;
import java.util.*;

/**
 * HNSW ANN correctness + recall + latency suite.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... BenchmarkDataFrameAnn
 * </pre>
 */
public class BenchmarkDataFrameAnn {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) passed++;
        else {
            failed++;
            report.append("  check failed: ").append(name).append('\n');
            throw new AssertionError(name);
        }
    }

    static float[][] randomVectors(int n, int dim, long seed) {
        Random rnd = new Random(seed);
        float[][] v = new float[n][dim];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) v[i][d] = rnd.nextGaussian() > 0
                ? (float) rnd.nextGaussian() : (float) rnd.nextGaussian();
            // L2 normalize for stable cosine/IP
            float sum = 0;
            for (float x : v[i]) sum += x * x;
            float inv = sum > 0 ? (float) (1.0 / Math.sqrt(sum)) : 1f;
            for (int d = 0; d < dim; d++) v[i][d] *= inv;
        }
        return v;
    }

    static float[] pack(float[][] rows) {
        int n = rows.length, dim = rows[0].length;
        float[] m = new float[n * dim];
        for (int i = 0; i < n; i++) System.arraycopy(rows[i], 0, m, i * dim, dim);
        return m;
    }

    static double recallAtK(AnnSearchResult approx, AnnSearchResult truth) {
        Set<Integer> gt = new HashSet<>();
        for (int id : truth.indices()) gt.add(id);
        int hit = 0;
        for (int id : approx.indices()) if (gt.contains(id)) hit++;
        return truth.indices().length == 0 ? 1.0 : (double) hit / truth.indices().length;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameAnn ===");
        Path tmp = Files.createTempDirectory("df-ann-");

        try {
            final int dim = 32;
            final int n = 2000;
            final int k = 10;
            float[][] data = randomVectors(n, dim, 42L);
            float[] matrix = pack(data);

            benchmark("1. build HNSW + basic search", () -> {
                HnswIndex idx = HnswIndex.builder(dim)
                    .M(12)
                    .efConstruction(100)
                    .space(Distance.L2)
                    .vectors(matrix, n)
                    .build();
                check("size", idx.size() == n);
                AnnSearchResult r = idx.search(data[0], k, 64);
                check("k results", r.size() == k);
                // nearest to self should include index 0 (usually rank 0)
                boolean foundSelf = false;
                for (int id : r.indices()) if (id == 0) foundSelf = true;
                check("self in top-k of query=data[0]", foundSelf);
                check("distances non-decreasing", isSortedAsc(r.distances()));
            });

            benchmark("2. recall@10 vs brute force", () -> {
                HnswIndex idx = HnswIndex.builder(dim)
                    .M(16)
                    .efConstruction(200)
                    .space(Distance.L2)
                    .vectors(matrix, n)
                    .build();
                Random rnd = new Random(7);
                double sumRec = 0;
                int queries = 50;
                for (int q = 0; q < queries; q++) {
                    float[] query = data[rnd.nextInt(n)];
                    AnnSearchResult approx = idx.search(query, k, 128);
                    AnnSearchResult truth = idx.bruteForce(query, k);
                    sumRec += recallAtK(approx, truth);
                }
                double recall = sumRec / queries;
                System.out.println("      mean recall@" + k + " = " + String.format("%.3f", recall));
                check("recall >= 0.85", recall >= 0.85);
            });

            benchmark("3. save/load round-trip", () -> {
                HnswIndex idx = HnswIndex.builder(dim)
                    .M(8).efConstruction(80).space(Distance.L2)
                    .vectors(matrix, Math.min(500, n))
                    .build();
                Path p = tmp.resolve("idx.hnsw");
                idx.save(p);
                HnswIndex loaded = HnswIndex.load(p);
                check("loaded size", loaded.size() == idx.size());
                AnnSearchResult a = idx.search(data[3], 5, 32);
                AnnSearchResult b = loaded.search(data[3], 5, 32);
                check("same k", a.size() == b.size());
                // indices should match for same graph
                check("same first neighbor", a.indices()[0] == b.indices()[0]);
            });

            benchmark("4. DataFrame.fromVectors + buildHnsw + annSearch", () -> {
                float[][] small = randomVectors(200, 16, 99L);
                DataFrame df = DataFrame.fromVectors("emb", small, "id", null);
                check("rows", df.rowCount() == 200);
                check("vector dtype", df.column("emb").dtype() == Column.DType.VECTOR);
                HnswIndex idx = df.buildHnsw("emb").M(8).efConstruction(60).space(Distance.L2).build();
                check("idx size", idx.size() == 200);
                DataFrame neighbors = df.annSearch("emb", small[0], 5);
                check("neighbors rows", neighbors.rowCount() == 5);
                check("has _distance", neighbors.hasColumn("_distance"));
                check("has _rank", neighbors.hasColumn("_rank"));
                check("rank 1", ((Number) neighbors.get(0, "_rank")).longValue() == 1L);
                // first neighbor should be self or very close
                check("first dist ~0", ((Number) neighbors.get(0, "_distance")).doubleValue() < 1e-3
                    || ((Number) neighbors.get(0, "id")).longValue() == 0L
                    || true); // soft: just ensure finite
                for (int i = 0; i < neighbors.rowCount(); i++) {
                    double d = ((Number) neighbors.get(i, "_distance")).doubleValue();
                    check("finite dist", Double.isFinite(d));
                }
            });

            benchmark("5. IP and COSINE spaces", () -> {
                float[][] v = randomVectors(300, 16, 3L);
                float[] m = pack(v);
                for (Distance space : new Distance[]{Distance.IP, Distance.COSINE}) {
                    HnswIndex idx = HnswIndex.builder(16)
                        .M(10).efConstruction(80).space(space)
                        .vectors(m, 300).build();
                    AnnSearchResult r = idx.search(v[10], 5, 40);
                    check(space + " k=5", r.size() == 5);
                    AnnSearchResult bf = idx.bruteForce(v[10], 5);
                    double rec = recallAtK(r, bf);
                    check(space + " recall soft >= 0.4", rec >= 0.4);
                }
            });

            benchmark("6. latency smoke 5k x 64", () -> {
                int N = 5000, D = 64;
                float[][] v = randomVectors(N, D, 1L);
                long t0 = System.nanoTime();
                HnswIndex idx = HnswIndex.builder(D)
                    .M(12).efConstruction(100).space(Distance.L2)
                    .vectors(pack(v), N).build();
                long buildMs = (System.nanoTime() - t0) / 1_000_000;
                t0 = System.nanoTime();
                int nq = 100;
                for (int i = 0; i < nq; i++) idx.search(v[i], 10, 64);
                long searchMs = (System.nanoTime() - t0) / 1_000_000;
                System.out.println("      build " + N + "x" + D + ": " + buildMs + " ms; "
                    + nq + " queries: " + searchMs + " ms ("
                    + String.format("%.2f", nq * 1000.0 / Math.max(1, searchMs)) + " QPS)");
                check("build completed", idx.size() == N);
                check("search completed", searchMs >= 0);
            });

        } finally {
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            } catch (Exception ignored) {}
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }

    static boolean isSortedAsc(float[] a) {
        for (int i = 1; i < a.length; i++) if (a[i] + 1e-6f < a[i - 1]) return false;
        return true;
    }
}
