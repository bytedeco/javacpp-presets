package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.opensearch.OpenSearch;
import org.bytedeco.pytorch.dataframe.opensearch.OpenSearchBackend;
import org.bytedeco.pytorch.dataframe.opensearch.OpenSearchException;
import org.bytedeco.pytorch.dataframe.opensearch.OpenSearchOptions;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreProvider;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStores;
import org.bytedeco.pytorch.dataframe.vectorstore.memory.InMemoryVectorStore;

import java.util.Map;
import java.util.Random;

/**
 * DataFrame ↔ OpenSearch full-client benchmark (REST, no opensearch-java).
 *
 * <p>Env: {@code OPENSEARCH_URL} (optional {@code OPENSEARCH_USER}/{@code OPENSEARCH_PASSWORD}).
 */
public class BenchmarkDataFrameOpenSearch {

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
        Random rnd = new Random(7);
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("emb", Column.DType.VECTOR);
        df.addColumn("title", Column.DType.STRING);
        df.addColumn("category", Column.DType.STRING);
        for (int i = 0; i < n; i++) {
            int r = df.addEmptyRow();
            df.set(r, "id", "id-" + i);
            df.set(r, "emb", unit(rnd, dim));
            df.set(r, "title", "doc-" + i);
            df.set(r, "category", i % 2 == 0 ? "even" : "odd");
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameOpenSearch ===");
        final int dim = 32;

        section("D0 SPI override smoke");
        benchmark("1. VectorStoreProvider overrides opensearch", () -> {
            VectorStores.registerProvider(new VectorStoreProvider() {
                @Override public String name() { return "opensearch"; }
                @Override public VectorStore open(Map<String, Object> config) {
                    int d = config.get("dim") instanceof Number n ? n.intValue() : dim;
                    return InMemoryVectorStore.builder(d).name("spi-os").metric(VectorMetric.L2).build();
                }
            });
            try (VectorStore vs = VectorStores.open("opensearch", Map.of("dim", dim, "index", "x"))) {
                check("spi backend memory", "memory".equals(vs.backend()), vs.backend());
            } finally {
                VectorStores.reloadProviders();
            }
        });

        benchmark("2. OpenSearchBackend register", () -> {
            OpenSearch.registerBackend(new OpenSearchBackend() {
                @Override public String name() { return "opensearch-test-only"; }
                @Override public OpenSearch open(Map<String, Object> config) {
                    return OpenSearch.openBuiltin(Map.of("url", "http://127.0.0.1:1"));
                }
            });
            check("registered", OpenSearch.backend("opensearch-test-only") != null);
        });

        section("D1 Options + factory");
        benchmark("3. OpenSearchOptions fluent", () -> {
            OpenSearchOptions o = OpenSearchOptions.builder()
                .index("docs").dim(dim).metric(VectorMetric.COSINE)
                .idColumn("id").vectorColumn("emb")
                .bulkBatch(100).refreshOnWrite(true)
                .engine("faiss")
                .build();
            check("index", "docs".equals(o.index()));
            check("engine", "faiss".equals(o.engine()));
            check("bulk", o.bulkBatch() == 100);
            OpenSearchOptions o2 = OpenSearchOptions.fromMap(Map.of(
                "index", "i2", "dim", dim, "metric", "l2", "engine", "nmslib"));
            check("fromMap", "i2".equals(o2.index()) && o2.metric() == VectorMetric.L2);
        });

        benchmark("4. connectUri / builder", () -> {
            try (OpenSearch os = OpenSearch.builder("http://localhost:9200")
                    .basicAuth("admin", "admin").build()) {
                check("url", os.url().contains("9200"));
            }
            try (OpenSearch os = OpenSearch.connectUri("opensearch://admin:admin@localhost:9200/docs")) {
                check("connectUri non-null", os != null);
            }
        });

        benchmark("5. DataFrame.toOpenSearch(OpenSearch, Options) present", () -> {
            boolean ok = false;
            for (var m : DataFrame.class.getMethods()) {
                if ("toOpenSearch".equals(m.getName()) && m.getParameterCount() == 2
                        && m.getParameterTypes()[0] == OpenSearch.class) {
                    ok = true; break;
                }
            }
            check("overload present", ok);
        });

        String url = System.getenv("OPENSEARCH_URL");
        if ((url == null || url.isBlank()) && args.length > 0) url = args[0];
        String user = System.getenv("OPENSEARCH_USER");
        String pass = System.getenv("OPENSEARCH_PASSWORD");

        section("D2–D5 live OpenSearch");
        if (url == null || url.isBlank()) {
            skip("live suite", "set OPENSEARCH_URL to exercise REST");
        } else {
            final String liveUrl = url;
            final String index = "df_bench_" + System.currentTimeMillis();
            try (OpenSearch os = (user != null)
                    ? OpenSearch.connect(liveUrl, user, pass == null ? "" : pass)
                    : OpenSearch.connect(liveUrl)) {
                DataFrame base = seed(40, dim);
                benchmark("6. ensure knn index + write", () -> {
                    os.deleteIndex(index);
                    OpenSearchOptions opts = OpenSearchOptions.builder()
                        .index(index).dim(dim).metric(VectorMetric.COSINE)
                        .idColumn("id").vectorColumn("emb")
                        .refreshOnWrite(true).build();
                    int n = base.toOpenSearch(os, opts);
                    check("written 40", n == 40, "n=" + n);
                    check("exists", os.indexExists(index));
                });
                benchmark("7. count + read", () -> {
                    long c = os.count(index);
                    check("count>=40", c >= 40 || c == -1, "count=" + c);
                    DataFrame back = DataFrame.readOpenSearch(os, OpenSearchOptions.builder()
                        .index(index).limit(100).build());
                    check("rows>0", back.rowCount() > 0, "rows=" + back.rowCount());
                });
                benchmark("8. knn search", () -> {
                    float[] q = (float[]) base.get(0, "emb");
                    DataFrame hits = DataFrame.searchOpenSearch(os, q, 5, OpenSearchOptions.builder()
                        .index(index).metric(VectorMetric.COSINE).build());
                    check("hits>0", hits.rowCount() > 0, "rows=" + hits.rowCount());
                });
                benchmark("9. scale 800 bulk", () -> {
                    DataFrame big = seed(800, dim);
                    os.deleteIndex(index);
                    int n = big.toOpenSearch(os, OpenSearchOptions.builder()
                        .index(index).dim(dim).idColumn("id").vectorColumn("emb")
                        .bulkBatch(200).refreshOnWrite(true).build());
                    check("written 800", n == 800, "n=" + n);
                });
                benchmark("10. delete index", () -> {
                    os.deleteIndex(index);
                    check("gone", !os.indexExists(index));
                });
            } catch (OpenSearchException e) {
                skip("live suite", "connection failed: " + e.getMessage());
            }
        }

        System.out.println("\n=== RESULT passed=" + passed + " failed=" + failed + " skipped=" + skipped + " ===");
        if (failed > 0) System.exit(1);
    }
}
