package dataframe;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.ann.AnnSearchResult;
import org.bytedeco.pytorch.dataframe.ann.Distance;
import org.bytedeco.pytorch.dataframe.ann.HnswIndex;
import org.bytedeco.pytorch.dataframe.faiss.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Multi-dimensional FAISS suite — API + correctness + latency vs faiss.md.
 *
 * <p>Mirrors the 5 engineering examples in {@code org/lance/ipc/faiss.md}:
 * <ol>
 *   <li>HNSW + IndexIDMap (add_with_ids, incremental, remove, persist, efSearch hot-update)</li>
 *   <li>IVF_PQ (train, nprobe, recall vs Flat GT, persist)</li>
 *   <li>CPU↔GPU semantic migration (SKIP if no CUDA)</li>
 *   <li>IndexShards parallel merge</li>
 *   <li>Flat range_search + blacklist filter + reconstruct</li>
 * </ol>
 *
 * <p>Also prints a comparison table: faiss HNSW / Flat vs legacy {@code ann.HnswIndex}.
 *
 * <pre>
 *   java ... dataframe.BenchmarkDataFrameFaiss           # default N=10k gate
 *   java ... dataframe.BenchmarkDataFrameFaiss --full    # faiss.md scale 100k
 *   java ... dataframe.BenchmarkDataFrameFaiss --n 5000 --nq 200 --dim 128
 * </pre>
 */
public class BenchmarkDataFrameFaiss {
    static int passed = 0, failed = 0, skipped = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<String> summary = new ArrayList<>();
    static final List<String> perfTable = new ArrayList<>();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static final class Skip extends RuntimeException {
        Skip(String m) { super(m); }
    }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
            summary.add(String.format("OK    %-56s %6d ms", name, ms));
        } catch (Skip s) {
            skipped++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" SKIP " + name + " (" + ms + " ms): " + s.getMessage());
            summary.add(String.format("SKIP  %-56s %s", name, s.getMessage()));
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
            summary.add(String.format("FAIL  %-56s %s", name, e.toString()));
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

    static void skip(String reason) { throw new Skip(reason); }

    static float[] createVectors(int n, int dim, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n * dim];
        for (int i = 0; i < n * dim; i++) v[i] = (float) rnd.nextGaussian();
        Faiss.normalize_L2(v, n, dim);
        return v;
    }

    static float[][] rowsOf(float[] packed, int n, int dim) {
        float[][] r = new float[n][dim];
        for (int i = 0; i < n; i++) System.arraycopy(packed, i * dim, r[i], 0, dim);
        return r;
    }

    static double calcRecall(long[][] pred, long[][] gt) {
        int nq = pred.length;
        int k = pred[0].length;
        int hit = 0;
        for (int q = 0; q < nq; q++) {
            Set<Long> g = new HashSet<>();
            for (long id : gt[q]) if (id >= 0) g.add(id);
            for (long id : pred[q]) if (id >= 0 && g.contains(id)) hit++;
        }
        return (double) hit / (nq * k);
    }

    static void addPerf(String backend, long buildMs, long searchMs, int nq, double recall) {
        double qps = searchMs <= 0 ? Double.POSITIVE_INFINITY : nq * 1000.0 / searchMs;
        String line = String.format(Locale.ROOT,
            "%-28s  build=%6d ms  search=%6d ms  QPS=%8.1f  recall@K=%5.3f",
            backend, buildMs, searchMs, qps, recall);
        perfTable.add(line);
        System.out.println("      " + line);
    }

    public static void main(String[] args) throws Exception {
        int DIM = 128, N_TOTAL = 10_000, N_QUERY = 200, K = 10;
        boolean full = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--full" -> { full = true; N_TOTAL = 100_000; N_QUERY = 2000; }
                case "--n" -> N_TOTAL = Integer.parseInt(args[++i]);
                case "--nq" -> N_QUERY = Integer.parseInt(args[++i]);
                case "--dim" -> DIM = Integer.parseInt(args[++i]);
                case "--k" -> K = Integer.parseInt(args[++i]);
            }
        }
        // scale nlist with N
        final int dim = DIM, nTotal = N_TOTAL, nQuery = N_QUERY, k = K;
        final int nlist = Math.min(1024, Math.max(16, nTotal / 40));

        System.out.println("=== BenchmarkDataFrameFaiss ===");
        System.out.println("dim=" + dim + " N=" + nTotal + " nq=" + nQuery + " k=" + k
            + (full ? " [full faiss.md scale]" : " [gate]"));
        System.out.println("device: " + Faiss.device_describe());

        Path tmp = Files.createTempDirectory("df-faiss-");
        float[] base = createVectors(nTotal, dim, 42L);
        float[] queries = createVectors(nQuery, dim, 7L);
        long[] ids = new long[nTotal];
        for (int i = 0; i < nTotal; i++) ids[i] = i;

        try {
            // -------- Example 1: HNSW + IDMap --------
            benchmark("1. HNSW + IndexIDMap add/search/inc/remove/persist/ef", () -> {
                IndexHNSWFlat raw = new IndexHNSWFlat(dim, 32);
                raw.hnsw.efConstruction = 128;
                raw.hnsw.efSearch = 64;
                IndexIDMap index = new IndexIDMap(raw);

                long t0 = System.nanoTime();
                index.add_with_ids(base, nTotal, ids);
                long buildMs = (System.nanoTime() - t0) / 1_000_000;
                check("ntotal", index.ntotal() == nTotal);

                t0 = System.nanoTime();
                SearchResult r = index.search(queries, nQuery, k);
                long searchMs = (System.nanoTime() - t0) / 1_000_000;
                check("D shape", r.D.length == nQuery && r.D[0].length == k);
                check("I shape", r.I.length == nQuery && r.I[0].length == k);
                // self-search on first base vector as query → id 0 often in top
                SearchResult self = index.search(Arrays.copyOf(base, dim), k);
                boolean found0 = false;
                for (long id : self.ids()) if (id == 0) found0 = true;
                check("self id 0 in top-k (soft)", found0 || self.distances()[0] < 1e-3);

                // incremental
                float[] neu = createVectors(50, dim, 99L);
                long[] newIds = new long[50];
                for (int i = 0; i < 50; i++) newIds[i] = nTotal + i;
                index.add_with_ids(neu, 50, newIds);
                check("after inc", index.ntotal() == nTotal + 50);

                // remove
                long rem = index.remove_ids(new IDSelectorArray(new long[]{10, 20, 30}));
                check("removed 3", rem == 3);
                check("ntotal after rem", index.ntotal() == nTotal + 50 - 3);

                // persist
                Path p = tmp.resolve("hnsw_id.jfaiss");
                Faiss.write_index(index, p);
                Index loaded = Faiss.read_index(p);
                check("loaded type", loaded instanceof IndexIDMap);
                // hot-update efSearch on inner HNSW
                IndexIDMap lim = (IndexIDMap) loaded;
                if (lim.index instanceof IndexHNSWFlat h) {
                    h.hnsw.efSearch = 96;
                    check("efSearch hot", h.hnsw.efSearch == 96);
                }
                SearchResult r2 = loaded.search(queries, Math.min(20, nQuery), k);
                check("loaded search k", r2.k() == k);

                // GT recall
                IndexFlatIP gt = new IndexFlatIP(dim);
                gt.add(base, nTotal);
                SearchResult gtr = gt.search(queries, nQuery, k);
                // rebuild clean HNSW for fair recall (post-remove index is rebuilt)
                IndexHNSWFlat h2 = new IndexHNSWFlat(dim, 32, MetricType.METRIC_INNER_PRODUCT);
                h2.hnsw.efConstruction = 128;
                h2.hnsw.efSearch = 64;
                long tb = System.nanoTime();
                h2.add(base, nTotal);
                long bms = (System.nanoTime() - tb) / 1_000_000;
                long ts = System.nanoTime();
                SearchResult hr = h2.search(queries, nQuery, k);
                long sms = (System.nanoTime() - ts) / 1_000_000;
                double rec = calcRecall(hr.I, gtr.I);
                System.out.println("      HNSW recall@" + k + " vs FlatIP = "
                    + String.format(Locale.ROOT, "%.3f", rec));
                check("HNSW recall >= 0.70", rec >= 0.70);
                addPerf("faiss.HNSWFlat M=32", bms, sms, nQuery, rec);
                addPerf("faiss.HNSW+IDMap (init)", buildMs, searchMs, nQuery, Double.NaN);
            });

            // -------- Example 2: IVF_PQ --------
            benchmark("2. IVF_PQ train/add/nprobe/recall/persist", () -> {
                int m = 16;
                check("dim % m == 0", dim % m == 0);
                IndexFlatIP quant = new IndexFlatIP(dim);
                IndexIVFPQ ivf = new IndexIVFPQ(quant, dim, nlist, m, 8);
                ivf.metric_type = MetricType.METRIC_INNER_PRODUCT;
                ivf.verbose = nTotal >= 50_000;

                long t0 = System.nanoTime();
                ivf.train(base, nTotal);
                long trainMs = (System.nanoTime() - t0) / 1_000_000;
                check("trained", ivf.is_trained());
                t0 = System.nanoTime();
                ivf.add(base, nTotal);
                long addMs = (System.nanoTime() - t0) / 1_000_000;
                check("ntotal", ivf.ntotal() == nTotal);
                ivf.nprobe = Math.min(32, nlist);

                t0 = System.nanoTime();
                SearchResult r = ivf.search(queries, nQuery, k);
                long searchMs = (System.nanoTime() - t0) / 1_000_000;
                check("k results", r.k() == k);

                IndexFlatIP gt = new IndexFlatIP(dim);
                gt.add(base, nTotal);
                SearchResult gtr = gt.search(queries, nQuery, k);
                double rec = calcRecall(r.I, gtr.I);
                System.out.println("      IVFPQ nlist=" + nlist + " nprobe=" + ivf.nprobe
                    + " train=" + trainMs + "ms add=" + addMs + "ms recall@" + k + "="
                    + String.format(Locale.ROOT, "%.3f", rec));
                // PQ is lossy — soft gate
                check("IVFPQ recall >= 0.20", rec >= 0.20);

                Path p = tmp.resolve("ivfpq.jfaiss");
                Faiss.write_index(ivf, p);
                Index loaded = Faiss.read_index(p);
                check("loaded IVFPQ", loaded instanceof IndexIVFPQ);
                ((IndexIVFPQ) loaded).nprobe = ivf.nprobe; // FAISS: must reset nprobe after load
                SearchResult r2 = loaded.search(queries, Math.min(10, nQuery), k);
                check("loaded search", r2.nq() > 0);

                // reconstruct
                float[] recon = ivf.reconstruct(0);
                check("reconstruct dim", recon.length == dim);
                addPerf("faiss.IVFPQ nprobe=" + ivf.nprobe, trainMs + addMs, searchMs, nQuery, rec);
            });

            // -------- Example 3: CPU↔GPU --------
            benchmark("3. CPU↔GPU Flat migration + search", () -> {
                System.out.println("      " + Faiss.device_describe());
                if (!Faiss.cuda_available()) {
                    skip("CUDA not available (" + DeviceSelector.lastProbeDetail() + ")");
                }
                IndexFlatIP cpu = new IndexFlatIP(dim);
                cpu.add(base, nTotal);
                StandardGpuResources res = new StandardGpuResources(0);
                long t0 = System.nanoTime();
                Index gpu = Faiss.index_cpu_to_gpu(res, 0, cpu);
                long migMs = (System.nanoTime() - t0) / 1_000_000;
                check("marked gpu or still searchable", gpu != null);

                t0 = System.nanoTime();
                SearchResult r = gpu.search(queries, nQuery, k);
                long searchMs = (System.nanoTime() - t0) / 1_000_000;
                check("gpu search k", r.k() == k);

                // CPU baseline
                DeviceSelector.setPreferred(DeviceSelector.Device.CPU);
                IndexFlatIP cpu2 = new IndexFlatIP(dim);
                cpu2.add(base, nTotal);
                t0 = System.nanoTime();
                SearchResult rc = cpu2.search(queries, nQuery, k);
                long cpuMs = (System.nanoTime() - t0) / 1_000_000;
                DeviceSelector.setPreferred(null);

                double rec = calcRecall(r.I, rc.I);
                System.out.println("      migrate=" + migMs + "ms gpuSearch=" + searchMs
                    + "ms cpuSearch=" + cpuMs + "ms agreement=" + String.format(Locale.ROOT, "%.3f", rec));
                check("GPU vs CPU recall agreement >= 0.99", rec >= 0.99);

                Index back = Faiss.index_gpu_to_cpu(gpu);
                Path p = tmp.resolve("gpu_exported.jfaiss");
                Faiss.write_index(back, p);
                check("exported exists", Files.exists(p));
                addPerf("faiss.FlatIP CUDA", migMs, searchMs, nQuery, 1.0);
                addPerf("faiss.FlatIP CPU", 0, cpuMs, nQuery, 1.0);
            });

            // -------- Example 4: IndexShards --------
            benchmark("4. IndexShards parallel + manual merge", () -> {
                int split = nTotal / 2;
                float[] s1 = Arrays.copyOfRange(base, 0, split * dim);
                float[] s2 = Arrays.copyOfRange(base, split * dim, nTotal * dim);
                long[] id1 = Arrays.copyOfRange(ids, 0, split);
                long[] id2 = Arrays.copyOfRange(ids, split, nTotal);

                IndexIDMap shard1 = new IndexIDMap(new IndexHNSWFlat(dim, 16, MetricType.METRIC_INNER_PRODUCT));
                ((IndexHNSWFlat) shard1.index).hnsw.efConstruction = 64;
                ((IndexHNSWFlat) shard1.index).hnsw.efSearch = 32;
                shard1.add_with_ids(s1, split, id1);

                IndexIDMap shard2 = new IndexIDMap(new IndexHNSWFlat(dim, 16, MetricType.METRIC_INNER_PRODUCT));
                ((IndexHNSWFlat) shard2.index).hnsw.efConstruction = 64;
                ((IndexHNSWFlat) shard2.index).hnsw.efSearch = 32;
                shard2.add_with_ids(s2, nTotal - split, id2);

                IndexShards shards = new IndexShards(dim, MetricType.METRIC_INNER_PRODUCT);
                shards.add_shard(shard1);
                shards.add_shard(shard2);
                shards.set_nthreads(4);

                long t0 = System.nanoTime();
                SearchResult r = shards.search(queries, nQuery, k);
                long ms = (System.nanoTime() - t0) / 1_000_000;
                check("shard k", r.k() == k);

                // manual merge
                SearchResult a = shard1.search(queries, nQuery, k);
                SearchResult b = shard2.search(queries, nQuery, k);
                SearchResult merged = IndexShards.merge(
                    new SearchResult[]{a, b}, nQuery, k, false /* IP higher better */);
                // agreement on first neighbor id rate
                int agree = 0;
                for (int q = 0; q < nQuery; q++) {
                    if (r.I[q][0] == merged.I[q][0]) agree++;
                }
                double rate = (double) agree / nQuery;
                System.out.println("      shards search " + ms + " ms; top1 agree with manual merge="
                    + String.format(Locale.ROOT, "%.3f", rate));
                check("top1 agree >= 0.90", rate >= 0.90);
                addPerf("faiss.IndexShards x2", 0, ms, nQuery, Double.NaN);
            });

            // -------- Example 5: Flat + range + filter + reconstruct --------
            benchmark("5. FlatIP range_search + blacklist + reconstruct", () -> {
                IndexFlatIP flat = new IndexFlatIP(dim);
                IndexIDMap index = new IndexIDMap(flat);
                index.add_with_ids(base, nTotal, ids);

                SearchResult top = index.search(queries, nQuery, k * 2);
                check("topk", top.k() == k * 2);

                // Random unit vectors in high-d have typical top IP ~0.2–0.4; pick adaptive threshold
                double thr = 0.15;
                if (top.D.length > 0 && top.D[0].length > 0) {
                    // median of top1 scores * 0.5 as soft radius
                    double s = 0;
                    int m = Math.min(nQuery, 50);
                    for (int q = 0; q < m; q++) s += top.D[q][0];
                    thr = Math.max(0.05, (s / m) * 0.5);
                }
                final double threshold = thr;
                Set<Long> black = Set.of(12L, 34L, 56L, 78L);
                int kept = 0;
                for (int q = 0; q < Math.min(nQuery, 50); q++) {
                    int c = 0;
                    for (int j = 0; j < top.k(); j++) {
                        if (top.D[q][j] >= threshold && !black.contains(top.I[q][j])) c++;
                    }
                    kept += c;
                }
                System.out.println("      filtered hits (50q, thr="
                    + String.format(Locale.ROOT, "%.3f", threshold) + "): " + kept);
                check("some filtered hits", kept > 0);

                // native range_search on inner flat (by position) — use bare flat for radius
                IndexFlatIP bare = new IndexFlatIP(dim);
                bare.add(base, nTotal);
                RangeSearchResult rr = bare.range_search(queries, Math.min(20, nQuery), (float) threshold);
                check("range lims len", rr.lims.length == Math.min(20, nQuery) + 1);
                System.out.println("      range_search total hits=" + rr.D.length);
                check("range_search has hits", rr.D.length > 0);

                float[] recon = bare.reconstruct(100 % nTotal);
                check("recon dim", recon.length == dim);
                // L2 switch
                IndexFlatL2 l2 = new IndexFlatL2(dim);
                // use non-normalized for L2 demo? still fine with normalized
                l2.add(base, Math.min(1000, nTotal));
                SearchResult lr = l2.search(queries, 5, 5);
                check("l2 k", lr.k() == 5);
                check("l2 dist >= 0", lr.D[0][0] >= 0f);
            });

            // -------- Comparison: legacy ann.HnswIndex --------
            benchmark("6. compare ann.HnswIndex vs faiss.IndexHNSWFlat", () -> {
                int n = Math.min(nTotal, 20_000);
                int nq = Math.min(nQuery, 200);
                float[] xb = Arrays.copyOf(base, n * dim);
                float[] xq = Arrays.copyOf(queries, nq * dim);

                // GT
                IndexFlatIP gt = new IndexFlatIP(dim);
                gt.add(xb, n);
                SearchResult gtr = gt.search(xq, nq, k);

                // legacy ann
                long t0 = System.nanoTime();
                HnswIndex legacy = HnswIndex.builder(dim)
                    .M(16).efConstruction(100).space(Distance.IP)
                    .vectors(xb, n).build();
                long legBuild = (System.nanoTime() - t0) / 1_000_000;
                t0 = System.nanoTime();
                // batch via loop
                long[][] legI = new long[nq][k];
                for (int q = 0; q < nq; q++) {
                    float[] qv = Arrays.copyOfRange(xq, q * dim, (q + 1) * dim);
                    AnnSearchResult ar = legacy.search(qv, k, 64);
                    for (int j = 0; j < k && j < ar.size(); j++) legI[q][j] = ar.indices()[j];
                }
                long legSearch = (System.nanoTime() - t0) / 1_000_000;
                double legRec = calcRecall(legI, gtr.I);

                // faiss HNSW
                IndexHNSWFlat fh = new IndexHNSWFlat(dim, 16, MetricType.METRIC_INNER_PRODUCT);
                fh.hnsw.efConstruction = 100;
                fh.hnsw.efSearch = 64;
                t0 = System.nanoTime();
                fh.add(xb, n);
                long fBuild = (System.nanoTime() - t0) / 1_000_000;
                t0 = System.nanoTime();
                SearchResult fr = fh.search(xq, nq, k);
                long fSearch = (System.nanoTime() - t0) / 1_000_000;
                double fRec = calcRecall(fr.I, gtr.I);

                addPerf("ann.HnswIndex (legacy)", legBuild, legSearch, nq, legRec);
                addPerf("faiss.HNSWFlat (new)", fBuild, fSearch, nq, fRec);

                double speedup = legSearch > 0 ? (double) legSearch / Math.max(1, fSearch) : 1.0;
                System.out.println("      search speedup faiss/legacy = "
                    + String.format(Locale.ROOT, "%.2fx", speedup)
                    + "  recall leg=" + String.format(Locale.ROOT, "%.3f", legRec)
                    + " faiss=" + String.format(Locale.ROOT, "%.3f", fRec));
                // Don't hard-fail on speedup (machine variance); require faiss recall competitive
                check("faiss HNSW recall >= legacy - 0.05", fRec + 0.05 >= legRec);
            });

            // -------- DataFrame hooks --------
            benchmark("7. DataFrame.buildFaiss + faissSearch", () -> {
                int n = Math.min(500, nTotal);
                float[][] rows = rowsOf(base, n, dim);
                long[] rowIds = Arrays.copyOf(ids, n);
                DataFrame df = DataFrame.fromVectors("emb", rows, "id", rowIds);
                check("rows", df.rowCount() == n);

                Index idx = df.buildFaiss("emb")
                    .hnsw(12)
                    .efConstruction(80)
                    .metric(MetricType.METRIC_INNER_PRODUCT)
                    .normalize(false)
                    .ids("id")
                    .build();
                check("idx ntotal", idx.ntotal() == n);
                DataFrame hits = df.faissSearch(idx, rows[0], 5);
                check("hits > 0", hits.rowCount() > 0);
                check("has _distance", hits.hasColumn("_distance"));
                check("has _rank", hits.hasColumn("_rank"));
                check("has _id", hits.hasColumn("_id"));

                Index flat = df.buildFaiss("emb").flatIP().build();
                DataFrame hits2 = df.faissSearch(flat, rows[1], 3);
                check("flat hits", hits2.rowCount() == 3);
            });

            // -------- Flat CPU throughput --------
            benchmark("8. IndexFlatIP bulk throughput", () -> {
                IndexFlatIP flat = new IndexFlatIP(dim);
                long t0 = System.nanoTime();
                flat.add(base, nTotal);
                long bms = (System.nanoTime() - t0) / 1_000_000;
                t0 = System.nanoTime();
                SearchResult r = flat.search(queries, nQuery, k);
                long sms = (System.nanoTime() - t0) / 1_000_000;
                check("flat k", r.k() == k);
                addPerf("faiss.FlatIP " + DeviceSelector.resolve(), bms, sms, nQuery, 1.0);
            });

        } finally {
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            } catch (Exception ignored) {}
        }

        System.out.println();
        System.out.println("======== PERF TABLE ========");
        for (String line : perfTable) System.out.println(line);
        System.out.println();
        System.out.println("======== SUMMARY ========");
        for (String line : summary) System.out.println(line);
        System.out.println("passed=" + passed + " failed=" + failed + " skipped=" + skipped);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL OK");
    }
}
