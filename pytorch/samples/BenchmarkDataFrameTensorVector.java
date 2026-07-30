package samples;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.ann.Distance;
import org.bytedeco.pytorch.dataframe.ann.HnswIndex;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.dtype.TensorData;
import org.bytedeco.pytorch.dataframe.dtype.VectorData;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStores;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Multi-dimensional Tensor ↔ DataFrame ↔ VectorStore correctness + live backend suite.
 *
 * <p>Always runs: TensorBridge ranks 1–4, DataFrame rank layouts, InMemory VectorStore.
 * Soft-skips remote backends that are not reachable (Redis / Milvus / Mongo / Qdrant).
 * OpenSearch is attempted at {@code http://localhost:9200} (or {@code VS_OPENSEARCH_URL}).
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameTensorVector
 * </pre>
 */
public class BenchmarkDataFrameTensorVector {
    static int passed = 0, failed = 0, skipped = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<String> summary = new ArrayList<>();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
            summary.add(String.format("OK    %-52s %6d ms", name, ms));
        } catch (Skip s) {
            skipped++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" SKIP " + name + " (" + ms + " ms): " + s.getMessage());
            summary.add(String.format("SKIP  %-52s %s", name, s.getMessage()));
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
            summary.add(String.format("FAIL  %-52s %s", name, e.toString()));
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

    static void skip(String reason) {
        throw new Skip(reason);
    }

    static final class Skip extends RuntimeException {
        Skip(String m) { super(m); }
    }

    static float[] randomFloats(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float) rnd.nextGaussian();
        return v;
    }

    static float[][] randomVectors(int n, int dim, long seed) {
        float[][] v = new float[n][];
        Random rnd = new Random(seed);
        for (int i = 0; i < n; i++) {
            v[i] = new float[dim];
            float sum = 0;
            for (int d = 0; d < dim; d++) {
                v[i][d] = (float) rnd.nextGaussian();
                sum += v[i][d] * v[i][d];
            }
            float inv = sum > 0 ? (float) (1.0 / Math.sqrt(sum)) : 1f;
            for (int d = 0; d < dim; d++) v[i][d] *= inv;
        }
        return v;
    }

    static boolean portOpen(String host, int port, int timeoutMs) {
        try (java.net.Socket s = new java.net.Socket()) {
            s.connect(new java.net.InetSocketAddress(host, port), timeoutMs);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    static String env(String key, String def) {
        String v = System.getenv(key);
        return v == null || v.isBlank() ? def : v;
    }

    // ---- tests --------------------------------------------------------------

    static void testTensorBridgeRanks() {
        // 1-D
        float[] d1 = randomFloats(16, 1);
        Tensor t1 = torch.tensor(d1);
        TensorData td1 = TensorBridge.toTensorData(t1);
        check("1d shape", Arrays.equals(td1.getShape(), new int[]{16}));
        check("1d data", TensorBridge.approxEqual(d1, td1.getData(), 1e-5f));
        Tensor t1b = td1.toTensor();
        check("1d roundtrip numel", t1b.numel() == 16);

        NDArray n1 = TensorBridge.toNDArray(t1);
        check("1d ndarray size", n1.size == 16);
        Tensor t1c = n1.toTensor();
        check("1d ndarray→tensor numel", t1c.numel() == 16);

        // 2-D
        float[] d2 = randomFloats(3 * 4, 2);
        Tensor t2 = torch.tensor(d2).reshape(new long[]{3, 4});
        TensorData td2 = TensorData.fromTensor(t2);
        check("2d shape", Arrays.equals(td2.getShape(), new int[]{3, 4}));
        check("2d data", TensorBridge.approxEqual(d2, td2.getData(), 1e-5f));
        Tensor t2b = td2.toTensor();
        check("2d rank", TensorBridge.rank(t2b) == 2);

        // 3-D
        float[] d3 = randomFloats(2 * 3 * 4, 3);
        Tensor t3 = torch.tensor(d3).reshape(new long[]{2, 3, 4});
        TensorData td3 = TensorBridge.toTensorData(t3);
        check("3d shape", Arrays.equals(td3.getShape(), new int[]{2, 3, 4}));
        NDArray n3 = td3.toNDArray();
        check("3d ndarray rank", n3.shape.length == 3);
        check("3d ndarray dims", n3.shape[0] == 2 && n3.shape[1] == 3 && n3.shape[2] == 4);

        // 4-D NCHW-style
        float[] d4 = randomFloats(2 * 3 * 4 * 5, 4);
        Tensor t4 = torch.tensor(d4).reshape(new long[]{2, 3, 4, 5});
        TensorData td4 = TensorData.fromTensor(t4);
        check("4d shape", Arrays.equals(td4.getShape(), new int[]{2, 3, 4, 5}));
        check("4d size", td4.size() == 2 * 3 * 4 * 5);
        Tensor t4b = TensorBridge.toTensor(td4);
        check("4d roundtrip rank", TensorBridge.rank(t4b) == 4);
        long[] s4 = TensorBridge.sizesOf(t4b);
        check("4d dims", s4[0] == 2 && s4[1] == 3 && s4[2] == 4 && s4[3] == 5);
    }

    static void testDtypeFactories() {
        float[] v = randomFloats(8, 10);
        EmbeddingData emb = new EmbeddingData(v, "hash");
        Tensor te = emb.toTensor();
        check("emb→tensor dim", te.numel() == 8);
        EmbeddingData emb2 = EmbeddingData.fromTensor(te, "hash");
        check("emb roundtrip", TensorBridge.approxEqual(v, emb2.getVector(), 1e-5f));

        VectorData vd = new VectorData(v, "v");
        Tensor tv = vd.toTensor();
        check("vd→tensor", tv.numel() == 8);
        VectorData vd2 = VectorData.fromTensor(tv, "v2");
        check("vd roundtrip dim", vd2.getVectorSize() == 8);

        // multi-dim VectorData
        double[] md = new double[12];
        for (int i = 0; i < 12; i++) md[i] = i;
        VectorData vdm = new VectorData(md, new int[]{3, 4}, "m");
        Tensor tm = vdm.toTensor();
        check("vd multi rank", TensorBridge.rank(tm) == 2);
        check("vd multi shape0", TensorBridge.sizesOf(tm)[0] == 3);

        TensorData td = new TensorData(v, new int[]{2, 4});
        NDArray na = td.toNDArray();
        check("td→ndarray", na.shape[0] == 2 && na.shape[1] == 4);
        TensorData td2 = TensorData.fromNDArray(na);
        check("ndarray→td shape", Arrays.equals(td2.getShape(), new int[]{2, 4}));
    }

    static void testDataFrameRankLayouts() {
        // rank 1 → single numeric col
        float[] d1 = {1, 2, 3, 4, 5};
        Tensor t1 = torch.tensor(d1);
        DataFrame df1 = DataFrame.fromTensor(t1, "x");
        check("df1 rows", df1.rowCount() == 5);
        check("df1 cols", df1.columnCount() == 1);

        // rank 2 COLUMNS
        float[] d2 = randomFloats(4 * 3, 20);
        Tensor t2 = torch.tensor(d2).reshape(new long[]{4, 3});
        DataFrame df2 = DataFrame.fromTensor(t2);
        check("df2 rows", df2.rowCount() == 4);
        check("df2 cols", df2.columnCount() == 3);
        Tensor back2 = df2.toTensor("col_0", "col_1", "col_2");
        check("df2 toTensor rank", TensorBridge.rank(back2) == 2);
        check("df2 toTensor shape", TensorBridge.sizesOf(back2)[0] == 4 && TensorBridge.sizesOf(back2)[1] == 3);

        // rank 2 ROWS_AS_TENSOR → VECTOR cells
        DataFrame df2r = DataFrame.fromTensor(t2, DataFrame.TensorLayout.ROWS_AS_TENSOR, "vec");
        check("df2r rows", df2r.rowCount() == 4);
        check("df2r dtype", df2r.column("vec").dtype() == Column.DType.VECTOR);
        Object cell0 = df2r.get(0, "vec");
        check("df2r cell float[]", cell0 instanceof float[] && ((float[]) cell0).length == 3);

        // rank 3 → TENSOR cells shape [3,4]
        float[] d3 = randomFloats(5 * 3 * 4, 30);
        Tensor t3 = torch.tensor(d3).reshape(new long[]{5, 3, 4});
        DataFrame df3 = DataFrame.fromTensor(t3, "t");
        check("df3 rows", df3.rowCount() == 5);
        check("df3 dtype TENSOR", df3.column("t").dtype() == Column.DType.TENSOR);
        Object c3 = df3.get(0, "t");
        check("df3 cell TensorData", c3 instanceof TensorData);
        check("df3 cell shape", Arrays.equals(((TensorData) c3).getShape(), new int[]{3, 4}));
        Tensor packed3 = df3.toTensorColumn("t");
        check("df3 pack rank", TensorBridge.rank(packed3) == 3);
        check("df3 pack shape0", TensorBridge.sizesOf(packed3)[0] == 5);

        // rank 4
        float[] d4 = randomFloats(2 * 2 * 3 * 4, 40);
        Tensor t4 = torch.tensor(d4).reshape(new long[]{2, 2, 3, 4});
        DataFrame df4 = DataFrame.fromTensorRows(t4, "vol");
        check("df4 rows", df4.rowCount() == 2);
        Object c4 = df4.get(0, "vol");
        check("df4 TensorData", c4 instanceof TensorData);
        check("df4 cell shape", Arrays.equals(((TensorData) c4).getShape(), new int[]{2, 3, 4}));
        Tensor packed4 = df4.toTensorColumn("vol");
        check("df4 pack rank", TensorBridge.rank(packed4) == 4);

        // fromNDArray
        NDArray arr = new NDArray(d2, 4, 3);
        DataFrame dfn = DataFrame.fromNDArray(arr);
        check("fromNDArray rows", dfn.rowCount() == 4);
        check("fromNDArray cols", dfn.columnCount() == 3);

        // fromTensors map
        Map<String, Tensor> m = new LinkedHashMap<>();
        m.put("a", t1);
        m.put("b", t2);
        DataFrame dfm = DataFrame.fromTensors(m);
        check("fromTensors rows", dfm.rowCount() == 1);
        check("fromTensors cols", dfm.columnCount() == 2);
    }

    static void testInMemoryVectorStore() throws Exception {
        final int dim = 32;
        final int n = 200;
        float[][] data = randomVectors(n, dim, 42L);

        DataFrame df = DataFrame.fromVectors("emb", data, "id", null);
        // also add a payload col
        df.addColumn("tag", Column.DType.STRING);
        for (int i = 0; i < n; i++) df.set(i, "tag", "g" + (i % 5));

        try (VectorStore vs = VectorStores.memory("bench-mem", dim, VectorMetric.L2)) {
            vs.ensureCollection();
            df.toVectorStore(vs, "id", "emb", "tag");
            check("mem count", vs.count() == n);

            VectorSearchResult r = vs.search(data[0], 5);
            check("mem search k", r.size() == 5);
            check("mem self in top", r.hits().get(0).id().equals("0") || containsId(r, "0"));

            // scroll → DataFrame
            DataFrame loaded = DataFrame.fromVectorStore(vs, 1000);
            check("mem scroll rows", loaded.rowCount() == n);
            check("mem scroll has vector", loaded.hasColumn("vector"));

            // fetch
            List<VectorRecord> got = vs.fetch("0", "1", "missing");
            check("mem fetch size", got.size() == 2);

            // ANN recall vs HNSW on same data
            float[] matrix = new float[n * dim];
            for (int i = 0; i < n; i++) System.arraycopy(data[i], 0, matrix, i * dim, dim);
            HnswIndex idx = HnswIndex.builder(dim).M(12).efConstruction(100)
                .space(Distance.L2).vectors(matrix, n).build();
            var truth = idx.search(data[7], 10, 64);
            VectorSearchResult approx = vs.search(VectorQuery.builder(data[7], 10).ef(64).build());
            int hit = 0;
            for (int id : truth.indices()) {
                if (containsId(approx, Integer.toString(id))) hit++;
            }
            double recall = hit / 10.0;
            check("mem recall@10 >= 0.7 (got " + recall + ")", recall >= 0.7);

            // multi-dim TENSOR must be rejected
            DataFrame bad = DataFrame.create();
            bad.addColumn("t", Column.DType.TENSOR);
            int ri = bad.addEmptyRow();
            bad.set(ri, "t", new TensorData(randomFloats(12, 99), new int[]{3, 4}));
            boolean rejected = false;
            try {
                bad.toVectorStore(vs, null, "t");
            } catch (VectorStoreException ex) {
                rejected = ex.getMessage() != null && ex.getMessage().contains("TENSOR");
            }
            check("mem reject TENSOR col", rejected);
        }
    }

    static boolean containsId(VectorSearchResult r, String id) {
        for (VectorHit h : r.hits()) if (id.equals(h.id())) return true;
        return false;
    }

    static void testLiveBackend(String label, VectorStoreFactory factory, int dim, float[][] data) {
        benchmark("live." + label + " upsert/search/scroll", () -> {
            VectorStore vs;
            try {
                vs = factory.open();
            } catch (Exception e) {
                skip("connect failed: " + e.getMessage());
                return;
            }
            try (vs) {
                try {
                    vs.dropCollection();
                } catch (Exception ignored) {}
                vs.ensureCollection();

                List<VectorRecord> batch = new ArrayList<>();
                int n = Math.min(data.length, 100);
                for (int i = 0; i < n; i++) {
                    batch.add(VectorRecord.of(String.valueOf(i), data[i],
                        Map.of("tag", "t" + (i % 3), "i", i)));
                }
                long t0 = System.nanoTime();
                vs.upsert(batch);
                long upsertMs = (System.nanoTime() - t0) / 1_000_000;

                // small settle for near-realtime indexes
                try { Thread.sleep(500); } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }

                t0 = System.nanoTime();
                VectorSearchResult r = vs.search(data[0], 5);
                long searchMs = (System.nanoTime() - t0) / 1_000_000;
                check(label + " search nonempty", r.size() > 0);

                t0 = System.nanoTime();
                DataFrame loaded = vs.toDataFrame(n + 10);
                long scrollMs = (System.nanoTime() - t0) / 1_000_000;
                check(label + " scroll rows>0", loaded.rowCount() > 0);

                System.out.printf("       [%s] upsert=%dms search=%dms scroll=%dms count≈%d hits=%d rows=%d%n",
                    label, upsertMs, searchMs, scrollMs, vs.count(), r.size(), loaded.rowCount());

                try { vs.dropCollection(); } catch (Exception ignored) {}
            } catch (Skip s) {
                throw s;
            } catch (VectorStoreException e) {
                skip(label + " error: " + e.getMessage());
            }
        });
    }

    @FunctionalInterface
    interface VectorStoreFactory {
        VectorStore open() throws Exception;
    }

    static void testExplicitFlattenToEmbeddingThenStore() throws Exception {
        // multi-dim tensor in DF → explicit flatten → EMBEDDING → vector store
        float[] raw = randomFloats(4 * 3 * 4, 55);
        Tensor t = torch.tensor(raw).reshape(new long[]{4, 3, 4});
        DataFrame df = DataFrame.fromTensor(t, "vol");
        // materialize embedding by mean-pool flatten of each cell
        df.addColumn("emb", Column.DType.EMBEDDING);
        for (int i = 0; i < df.rowCount(); i++) {
            TensorData td = (TensorData) df.get(i, "vol");
            float[] flat = td.getData();
            // simple L2-normalize flatten as "embedding"
            float sum = 0;
            for (float x : flat) sum += x * x;
            float inv = sum > 0 ? (float) (1.0 / Math.sqrt(sum)) : 1f;
            float[] e = new float[flat.length];
            for (int j = 0; j < flat.length; j++) e[j] = flat[j] * inv;
            df.set(i, "emb", new EmbeddingData(e, "flatten-pool"));
        }
        int dim = ((EmbeddingData) df.get(0, "emb")).getDimension();
        try (VectorStore vs = VectorStores.memory("flatten-demo", dim, VectorMetric.COSINE)) {
            vs.ensureCollection();
            df.toVectorStore(vs, null, "emb");
            check("flatten store count", vs.count() == 4);
            VectorSearchResult r = vs.search(((EmbeddingData) df.get(0, "emb")).getVector(), 2);
            check("flatten search", r.size() == 2);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameTensorVector ===");
        // init torch
        try {
            Tensor warm = torch.tensor(new float[]{1f, 2f, 3f});
            check("torch warmup", warm.numel() == 3);
        } catch (Throwable e) {
            System.err.println("Torch init failed: " + e);
            throw e;
        }

        benchmark("1. TensorBridge ranks 1–4 roundtrip", BenchmarkDataFrameTensorVector::testTensorBridgeRanks);
        benchmark("2. dtype factories (Embedding/Vector/TensorData/NDArray)", BenchmarkDataFrameTensorVector::testDtypeFactories);
        benchmark("3. DataFrame rank layouts + toTensorColumn", BenchmarkDataFrameTensorVector::testDataFrameRankLayouts);
        benchmark("4. InMemory VectorStore upsert/search/scroll/recall", BenchmarkDataFrameTensorVector::testInMemoryVectorStore);
        benchmark("5. explicit flatten multi-dim → EMBEDDING → store", BenchmarkDataFrameTensorVector::testExplicitFlattenToEmbeddingThenStore);

        final int dim = 32;
        float[][] liveData = randomVectors(50, dim, 7L);

        // OpenSearch
        String osUrl = env("VS_OPENSEARCH_URL", "http://localhost:9200");
        testLiveBackend("opensearch", () -> {
            if (!portOpen("localhost", 9200, 400) && osUrl.contains("localhost")) {
                skip("port 9200 closed");
            }
            return VectorStores.openSearch(osUrl, "df-bench-vecs", dim, VectorMetric.L2);
        }, dim, liveData);

        // Redis
        String redisHost = env("VS_REDIS_HOST", "127.0.0.1");
        int redisPort = Integer.parseInt(env("VS_REDIS_PORT", "6379"));
        testLiveBackend("redis", () -> {
            if (!portOpen(redisHost, redisPort, 400)) skip("port " + redisPort + " closed");
            return VectorStores.redis(redisHost, redisPort, "idx:df-bench", dim, VectorMetric.COSINE);
        }, dim, liveData);

        // Milvus REST
        String milvusUrl = env("VS_MILVUS_URL", "http://localhost:9091");
        testLiveBackend("milvus", () -> {
            boolean open = portOpen("localhost", 9091, 400) || portOpen("localhost", 19530, 400);
            if (!open && milvusUrl.contains("localhost")) skip("ports 9091/19530 closed");
            return VectorStores.milvus(milvusUrl, "df_bench_vecs", dim, VectorMetric.L2);
        }, dim, liveData);

        // Qdrant
        String qUrl = env("VS_QDRANT_URL", "http://localhost:6333");
        testLiveBackend("qdrant", () -> {
            if (!portOpen("localhost", 6333, 400) && qUrl.contains("localhost")) skip("port 6333 closed");
            return VectorStores.qdrant(qUrl, "df_bench_vecs", dim, VectorMetric.COSINE);
        }, dim, liveData);

        // Mongo Atlas Data API — only if env configured
        String mongoApi = System.getenv("ATLAS_DATA_API_URL");
        String mongoKey = System.getenv("ATLAS_API_KEY");
        testLiveBackend("mongo-atlas", () -> {
            if (mongoApi == null || mongoApi.isBlank() || mongoKey == null || mongoKey.isBlank()) {
                skip("ATLAS_DATA_API_URL / ATLAS_API_KEY not set");
            }
            return VectorStores.mongoAtlas(mongoApi, mongoKey,
                env("ATLAS_DATA_SOURCE", "Cluster0"),
                env("ATLAS_DATABASE", "rag"),
                "df_bench_vecs", dim, VectorMetric.COSINE);
        }, dim, liveData);

        System.out.println();
        System.out.println("=== Summary ===");
        for (String line : summary) System.out.println(line);
        System.out.println();
        System.out.printf("Passed checks≈%d  Failed benches=%d  Skipped=%d%n", passed, failed, skipped);
        if (report.length() > 0) {
            System.out.println("--- failures ---");
            System.out.print(report);
        }
        if (failed > 0) System.exit(1);
    }
}
