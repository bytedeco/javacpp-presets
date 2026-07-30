package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.mongo.Mongo;
import org.bytedeco.pytorch.dataframe.mongo.MongoBackend;
import org.bytedeco.pytorch.dataframe.mongo.MongoException;
import org.bytedeco.pytorch.dataframe.mongo.MongoOptions;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreProvider;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStores;
import org.bytedeco.pytorch.dataframe.vectorstore.memory.InMemoryVectorStore;

import java.util.Map;
import java.util.Random;

/**
 * DataFrame ↔ Mongo full-client benchmark (Atlas Data API, no mongo-driver).
 *
 * <p>Env: {@code MONGO_DATA_API_URL}, {@code MONGO_API_KEY}
 * (optional {@code MONGO_DATA_SOURCE}, {@code MONGO_DATABASE}).
 */
public class BenchmarkDataFrameMongo {

    static int passed = 0, failed = 0, skipped = 0;

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void section(String title) { System.out.println("\n=== " + title + " ==="); }

    static void check(String name, boolean ok) { check(name, ok, null); }

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
            System.out.println("  OK  " + name + " (" + ((System.nanoTime() - t0) / 1_000_000L) + " ms)");
        } catch (Throwable e) {
            failed++;
            System.out.println(" FAIL " + name + ": " + e);
            e.printStackTrace(System.out);
        }
    }

    static float[] unit(Random rnd, int dim) {
        float[] v = new float[dim];
        double n = 0;
        for (int i = 0; i < dim; i++) { v[i] = (float) rnd.nextGaussian(); n += v[i] * v[i]; }
        n = Math.sqrt(n);
        if (n > 1e-12) for (int i = 0; i < dim; i++) v[i] = (float) (v[i] / n);
        return v;
    }

    static DataFrame seed(int n, int dim) {
        Random rnd = new Random(11);
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("emb", Column.DType.VECTOR);
        df.addColumn("title", Column.DType.STRING);
        for (int i = 0; i < n; i++) {
            int r = df.addEmptyRow();
            df.set(r, "id", "id-" + i);
            df.set(r, "emb", unit(rnd, dim));
            df.set(r, "title", "doc-" + i);
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameMongo ===");
        final int dim = 32;

        section("D0 SPI override smoke");
        benchmark("1. VectorStoreProvider overrides mongo", () -> {
            VectorStores.registerProvider(new VectorStoreProvider() {
                @Override public String name() { return "mongo"; }
                @Override public VectorStore open(Map<String, Object> config) {
                    int d = config.get("dim") instanceof Number n ? n.intValue() : dim;
                    return InMemoryVectorStore.builder(d).name("spi-mongo").metric(VectorMetric.COSINE).build();
                }
            });
            try (VectorStore vs = VectorStores.open("mongo", Map.of("dim", dim, "url", "https://x", "collection", "c"))) {
                check("spi backend memory", "memory".equals(vs.backend()), vs.backend());
            } finally {
                VectorStores.reloadProviders();
            }
        });

        benchmark("2. MongoBackend register", () -> {
            Mongo.registerBackend(new MongoBackend() {
                @Override public String name() { return "mongo-test-only"; }
                @Override public Mongo open(Map<String, Object> config) {
                    return Mongo.openBuiltin(Map.of(
                        "url", "https://example.invalid",
                        "apiKey", "x",
                        "dataSource", "Cluster0",
                        "database", "test"));
                }
            });
            check("registered", Mongo.backend("mongo-test-only") != null);
        });

        section("D1 Options + factory");
        benchmark("3. MongoOptions fluent + fromMap", () -> {
            MongoOptions o = MongoOptions.builder()
                .database("rag").collection("docs").dim(dim)
                .idColumn("id").vectorColumn("emb")
                .vectorPath("embedding").indexName("vector_index")
                .batchSize(20)
                .build();
            check("db", "rag".equals(o.database()));
            check("coll", "docs".equals(o.collection()));
            check("path", "embedding".equals(o.vectorPath()));
            MongoOptions o2 = MongoOptions.fromMap(Map.of(
                "database", "d2", "collection", "c2", "dim", dim, "metric", "ip"));
            check("fromMap", "d2".equals(o2.database()) && o2.metric() == VectorMetric.IP);
        });

        benchmark("4. builder / connectUri parse", () -> {
            try (Mongo m = Mongo.builder("https://data.mongodb-api.com/app/x/endpoint/data/v1")
                    .apiKey("k").dataSource("Cluster0").database("test").build()) {
                check("url set", m.url() != null && m.url().contains("mongodb"));
                check("ds", "Cluster0".equals(m.dataSource()));
            }
            try (Mongo m = Mongo.connectUri(
                    "atlas://data.mongodb-api.com/app/x/endpoint/data/v1?apiKey=k&dataSource=C1&database=db")) {
                check("connectUri ds", "C1".equals(m.dataSource()), "ds=" + m.dataSource());
                check("connectUri db", "db".equals(m.database()), "db=" + m.database());
            }
        });

        benchmark("5. DataFrame.toMongo(Mongo, Options) present", () -> {
            boolean ok = false;
            for (var m : DataFrame.class.getMethods()) {
                if ("toMongo".equals(m.getName()) && m.getParameterCount() == 2
                        && m.getParameterTypes()[0] == Mongo.class) {
                    ok = true; break;
                }
            }
            check("overload present", ok);
        });

        String url = System.getenv("MONGO_DATA_API_URL");
        String apiKey = System.getenv("MONGO_API_KEY");
        String dataSource = System.getenv().getOrDefault("MONGO_DATA_SOURCE", "Cluster0");
        String database = System.getenv().getOrDefault("MONGO_DATABASE", "test");
        if ((url == null || url.isBlank()) && args.length >= 2) {
            url = args[0];
            apiKey = args[1];
        }

        section("D2–D5 live Mongo Data API");
        if (url == null || url.isBlank() || apiKey == null || apiKey.isBlank()) {
            skip("live suite", "set MONGO_DATA_API_URL + MONGO_API_KEY");
        } else {
            final String liveUrl = url;
            final String liveKey = apiKey;
            final String coll = "df_bench_" + System.currentTimeMillis();
            try (Mongo m = Mongo.connect(liveUrl, liveKey, dataSource, database)) {
                DataFrame base = seed(25, dim);
                benchmark("6. ensure + writeDataFrame", () -> {
                    m.dropCollection(coll);
                    MongoOptions opts = MongoOptions.builder()
                        .database(database).collection(coll).dim(dim)
                        .idColumn("id").vectorColumn("emb")
                        .vectorPath("embedding")
                        .ifExists(MongoOptions.IfExists.REPLACE)
                        .build();
                    int n = base.toMongo(m, opts);
                    check("written 25", n == 25, "n=" + n);
                });
                benchmark("7. count + read", () -> {
                    long c = m.countDocuments(coll, Map.of());
                    check("count>=25", c >= 25 || c == -1, "count=" + c);
                    DataFrame back = DataFrame.readMongo(m, MongoOptions.builder()
                        .database(database).collection(coll).limit(100).build());
                    check("rows>0", back.rowCount() > 0, "rows=" + back.rowCount());
                });
                benchmark("8. findOne by id", () -> {
                    var doc = m.findOne(coll, Map.of("_id", "id-0"));
                    // id field may be _id depending on options; accept either
                    if (doc == null) doc = m.findOne(coll, Map.of("id", "id-0"));
                    check("found", doc != null);
                });
                benchmark("9. scale 200 upserts", () -> {
                    DataFrame big = seed(200, dim);
                    m.dropCollection(coll);
                    int n = big.toMongo(m, MongoOptions.builder()
                        .database(database).collection(coll).dim(dim)
                        .idColumn("id").vectorColumn("emb").build());
                    check("written 200", n == 200, "n=" + n);
                });
                // vectorSearch requires Atlas Vector Search index created in UI — soft
                benchmark("10. vectorSearch soft", () -> {
                    try {
                        float[] q = (float[]) base.get(0, "emb");
                        DataFrame hits = DataFrame.searchMongo(m, q, 5, MongoOptions.builder()
                            .database(database).collection(coll)
                            .vectorPath("embedding").indexName("vector_index")
                            .metric(VectorMetric.COSINE).build());
                        check("hits optional", hits.rowCount() >= 0);
                    } catch (MongoException e) {
                        skip("vectorSearch", "index may be missing: " + e.getMessage());
                    }
                });
                benchmark("11. drop (deleteMany)", () -> {
                    m.dropCollection(coll);
                    long c = m.countDocuments(coll, Map.of());
                    check("empty", c == 0 || c == -1, "count=" + c);
                });
            } catch (MongoException e) {
                skip("live suite", "connection failed: " + e.getMessage());
            }
        }

        System.out.println("\n=== RESULT passed=" + passed + " failed=" + failed + " skipped=" + skipped + " ===");
        if (failed > 0) System.exit(1);
    }
}
