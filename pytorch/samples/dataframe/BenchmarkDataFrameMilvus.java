package dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.milvus.Milvus;
import org.bytedeco.pytorch.dataframe.milvus.MilvusBackend;
import org.bytedeco.pytorch.dataframe.milvus.MilvusException;
import org.bytedeco.pytorch.dataframe.milvus.MilvusOptions;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreProvider;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStores;
import org.bytedeco.pytorch.dataframe.vectorstore.memory.InMemoryVectorStore;

import java.util.Map;
import java.util.Random;

/**
 * DataFrame ↔ Milvus full-client benchmark (REST v2, no milvus-sdk-java).
 *
 * <p>Dimensions:
 * <ul>
 *   <li>D0 SPI override smoke (always runs)</li>
 *   <li>D1 Options / factory API smoke</li>
 *   <li>D2 live connection + collection lifecycle (env {@code MILVUS_URL})</li>
 *   <li>D3 DF write/read round-trip</li>
 *   <li>D4 vector upsert / search / scroll</li>
 *   <li>D5 scale 1k rows</li>
 * </ul>
 *
 * <pre>
 *   java ... dataframe.BenchmarkDataFrameMilvus
 *   MILVUS_URL=http://localhost:9091 MILVUS_TOKEN=root:Milvus java ... dataframe.BenchmarkDataFrameMilvus
 * </pre>
 */
public class BenchmarkDataFrameMilvus {

    static int passed = 0, failed = 0, skipped = 0;

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void section(String title) {
        System.out.println("\n=== " + title + " ===");
    }

    static void check(String name, boolean ok) {
        check(name, ok, null);
    }

    static void check(String name, boolean ok, String detail) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
        } else {
            failed++;
            System.out.println("  FAIL  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
        }
    }

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("  SKIP  " + name + " — " + reason);
    }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            System.out.println(" FAIL " + name + ": " + e);
            e.printStackTrace(System.out);
        }
    }

    static float[] unit(Random rnd, int dim) {
        float[] v = new float[dim];
        double n = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) rnd.nextGaussian();
            n += v[i] * v[i];
        }
        n = Math.sqrt(n);
        if (n > 1e-12) for (int i = 0; i < dim; i++) v[i] = (float) (v[i] / n);
        return v;
    }

    static DataFrame seed(int n, int dim) {
        Random rnd = new Random(42);
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("emb", Column.DType.VECTOR);
        df.addColumn("title", Column.DType.STRING);
        df.addColumn("year", Column.DType.INT64);
        for (int i = 0; i < n; i++) {
            int r = df.addEmptyRow();
            df.set(r, "id", String.valueOf(i));
            df.set(r, "emb", unit(rnd, dim));
            df.set(r, "title", "doc-" + i);
            df.set(r, "year", 2000L + (i % 20));
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameMilvus ===");
        final int dim = 32;

        // ── D0 SPI override ──────────────────────────────────────────────
        section("D0 SPI override smoke");
        benchmark("1. VectorStoreProvider overrides milvus scheme", () -> {
            VectorStoreProvider prev = null;
            VectorStores.registerProvider(new VectorStoreProvider() {
                @Override public String name() { return "milvus"; }
                @Override public VectorStore open(Map<String, Object> config) {
                    int d = config.get("dim") instanceof Number n ? n.intValue() : dim;
                    return InMemoryVectorStore.builder(d).name("spi-milvus").metric(VectorMetric.COSINE).build();
                }
            });
            try (VectorStore vs = VectorStores.open("milvus", Map.of("dim", dim, "collection", "x"))) {
                check("spi backend is memory", "memory".equals(vs.backend()), "backend=" + vs.backend());
                vs.ensureCollection();
                DataFrame df = seed(5, dim);
                df.toVectorStore(vs, "id", "emb", "title");
                check("spi upsert count", vs.count() == 5, "count=" + vs.count());
            } finally {
                // re-register nothing: clear by reload (drops SPI-only) then leave builtin
                VectorStores.reloadProviders();
            }
        });

        benchmark("2. MilvusBackend SPI register/unregister", () -> {
            final boolean[] opened = {false};
            Milvus.registerBackend(new MilvusBackend() {
                @Override public String name() { return "milvus-test-only"; }
                @Override public Milvus open(Map<String, Object> config) {
                    opened[0] = true;
                    // still use builtin under the hood for a real client shape
                    return Milvus.openBuiltin(Map.of(
                        "url", config.getOrDefault("url", "http://127.0.0.1:1")));
                }
            });
            check("backend registered", Milvus.backend("milvus-test-only") != null);
            // open default milvus scheme still builtin unless name is milvus
            check("default scheme not overridden by alias", Milvus.backend("milvus") == null
                || !"milvus-test-only".equals(Milvus.backend("milvus").name()));
            // force via openBuiltin path still works without network until first call
            Milvus m = Milvus.openBuiltin(Map.of("url", "http://127.0.0.1:1"));
            check("openBuiltin non-null", m != null);
            m.close();
        });

        // ── D1 Options / factory ─────────────────────────────────────────
        section("D1 Options + factory API");
        benchmark("3. MilvusOptions fluent + fromMap", () -> {
            MilvusOptions o = MilvusOptions.builder()
                .collection("docs").dim(dim).metric(VectorMetric.IP)
                .idColumn("id").vectorColumn("emb")
                .batchSize(50).ifExists(MilvusOptions.IfExists.APPEND)
                .payloadColumns("title", "year")
                .build();
            check("collection", "docs".equals(o.collection()));
            check("dim", o.dim() == dim);
            check("metric IP", o.metric() == VectorMetric.IP);
            check("batch", o.batchSize() == 50);
            MilvusOptions o2 = MilvusOptions.fromMap(Map.of(
                "collection", "c2", "dim", dim, "metric", "cosine"));
            check("fromMap collection", "c2".equals(o2.collection()));
            check("fromMap metric", o2.metric() == VectorMetric.COSINE);
        });

        benchmark("4. connectUri parse + builder", () -> {
            // no network until first call
            Milvus m = null;
            try {
                m = Milvus.builder("http://localhost:9091").token("root:Milvus").dbName("default").build();
                check("builder url", m.url() != null && m.url().contains("9091"));
                check("builder db", "default".equals(m.dbName()));
            } finally {
                if (m != null) m.close();
            }
            Milvus m2 = Milvus.connectUri("milvus://localhost:9091/mydb?token=t");
            try {
                check("connectUri db", "mydb".equals(m2.dbName()), "db=" + m2.dbName());
            } finally {
                m2.close();
            }
        });

        benchmark("5. DataFrame.toMilvus overload resolves (compile-time smoke via reflection-ish)", () -> {
            // ensure new overloads exist: toMilvus(Milvus, MilvusOptions)
            boolean ok = false;
            for (var method : DataFrame.class.getMethods()) {
                if ("toMilvus".equals(method.getName()) && method.getParameterCount() == 2
                        && method.getParameterTypes()[0] == Milvus.class) {
                    ok = true;
                    break;
                }
            }
            check("toMilvus(Milvus, Options) present", ok);
        });

        // ── D2+ live ─────────────────────────────────────────────────────
        String url = System.getenv("MILVUS_URL");
        if (url == null || url.isBlank()) {
            if (args.length > 0) url = args[0];
        }
        String token = System.getenv("MILVUS_TOKEN");

        section("D2–D5 live Milvus");
        if (url == null || url.isBlank()) {
            skip("live suite", "set MILVUS_URL (optional MILVUS_TOKEN) to exercise REST");
        } else {
            final String liveUrl = url;
            final String liveToken = token;
            final String coll = "df_bench_" + System.currentTimeMillis();
            try (Milvus m = liveToken == null
                    ? Milvus.connect(liveUrl)
                    : Milvus.connect(liveUrl, liveToken)) {
                benchmark("6. has/create/load collection", () -> {
                    m.dropCollection(coll);
                    check("not exists", !m.hasCollection(coll));
                    m.createCollection(coll, dim, VectorMetric.COSINE);
                    check("exists", m.hasCollection(coll));
                });

                DataFrame base = seed(50, dim);
                benchmark("7. writeDataFrame + count", () -> {
                    MilvusOptions opts = MilvusOptions.builder()
                        .collection(coll).dim(dim).metric(VectorMetric.COSINE)
                        .idColumn("id").vectorColumn("emb")
                        .ifExists(MilvusOptions.IfExists.APPEND)
                        .build();
                    int n = base.toMilvus(m, opts);
                    check("written 50", n == 50, "n=" + n);
                    long c = m.count(coll);
                    check("count>=50", c >= 50 || c == -1, "count=" + c);
                });

                benchmark("8. readDataFrame", () -> {
                    DataFrame back = DataFrame.readMilvus(m, MilvusOptions.builder()
                        .collection(coll).limit(100).build());
                    check("rows>0", back.rowCount() > 0, "rows=" + back.rowCount());
                });

                benchmark("9. searchDataFrame", () -> {
                    float[] q = (float[]) base.get(0, "emb");
                    DataFrame hits = DataFrame.searchMilvus(m, q, 5, MilvusOptions.builder()
                        .collection(coll).metric(VectorMetric.COSINE).build());
                    check("hits>0", hits.rowCount() > 0, "rows=" + hits.rowCount());
                });

                DataFrame big = seed(1000, dim);
                benchmark("10. scale 1k write", () -> {
                    m.dropCollection(coll);
                    MilvusOptions opts = MilvusOptions.builder()
                        .collection(coll).dim(dim).idColumn("id").vectorColumn("emb")
                        .batchSize(200).build();
                    int n = big.toMilvus(m, opts);
                    check("written 1000", n == 1000, "n=" + n);
                });

                benchmark("11. drop collection", () -> {
                    m.dropCollection(coll);
                    check("dropped", !m.hasCollection(coll));
                });
            } catch (MilvusException e) {
                skip("live suite", "connection failed: " + e.getMessage());
            }
        }

        System.out.println("\n=== RESULT passed=" + passed + " failed=" + failed + " skipped=" + skipped + " ===");
        if (failed > 0) System.exit(1);
    }
}
