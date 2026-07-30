package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStores;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * DataFrame ↔ VectorStore dedicated I/O benchmark.
 *
 * <p>Exercises the new per-backend convenience APIs on {@link DataFrame}:
 * <ul>
 *   <li>{@code toMemoryVectorStore} / {@code toVectorStore(uri|scheme)} / {@code toQdrant}/…</li>
 *   <li>{@code fromVectorStore} / {@code readVectorStore} / {@code searchVectorStore}</li>
 *   <li>{@code openVectorStore} factory</li>
 *   <li>Scale upsert + k-NN + scroll round-trip (in-memory HNSW — always available)</li>
 *   <li>Builder smoke for remote backends (no live server required)</li>
 * </ul>
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameVectorStore
 *   java ... samples.BenchmarkDataFrameVectorStore qdrant://localhost:6333/clips?dim=32&amp;metric=cosine
 * </pre>
 */
public class BenchmarkDataFrameVectorStore {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void section(String title) {
        System.out.println("\n=== " + title + " ===");
        report.append("\n=== ").append(title).append(" ===\n");
    }

    static void check(String name, boolean ok) {
        check(name, ok, null);
    }

    static void check(String name, boolean ok, String detail) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
            report.append("PASS  ").append(name).append('\n');
        } else {
            failed++;
            System.out.println("  FAIL  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
            report.append("FAIL  ").append(name);
            if (detail != null) report.append(" — ").append(detail);
            report.append('\n');
        }
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
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static float[] randomUnit(Random rnd, int dim) {
        float[] v = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) (rnd.nextGaussian());
            norm += v[i] * v[i];
        }
        norm = Math.sqrt(norm);
        if (norm > 1e-12) {
            for (int i = 0; i < dim; i++) v[i] = (float) (v[i] / norm);
        }
        return v;
    }

    static DataFrame seedVectors(int n, int dim, long seed) {
        Random rnd = new Random(seed);
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("emb", Column.DType.VECTOR);
        df.addColumn("title", Column.DType.STRING);
        df.addColumn("year", Column.DType.INT64);
        df.addColumn("category", Column.DType.STRING);
        for (int i = 0; i < n; i++) {
            int row = df.addEmptyRow();
            df.set(row, "id", "id-" + i);
            df.set(row, "emb", randomUnit(rnd, dim));
            df.set(row, "title", "doc-" + i);
            df.set(row, "year", 2000L + (i % 25));
            df.set(row, "category", i % 2 == 0 ? "even" : "odd");
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameVectorStore ===");
        final int dim = 32;
        final int n = 500;

        // ── D1: open factories ────────────────────────────────────────────
        section("D1 openVectorStore factories");
        benchmark("1. openVectorStore(scheme, config)", () -> {
            try (VectorStore vs = DataFrame.openVectorStore("memory", Map.of(
                    "name", "demo",
                    "dim", dim,
                    "metric", "cosine"))) {
                check("backend memory", "memory".equalsIgnoreCase(vs.backend())
                        || vs.backend().toLowerCase().contains("memory")
                        || vs.backend().toLowerCase().contains("hnsw")
                        || vs.backend().toLowerCase().contains("local"),
                        "backend=" + vs.backend());
                check("dim", vs.dim() == dim, "dim=" + vs.dim());
                vs.ensureCollection();
                check("count 0", vs.count() == 0 || vs.count() == -1, "count=" + vs.count());
            }
        });

        benchmark("2. openVectorStore(uri)", () -> {
            try (VectorStore vs = DataFrame.openVectorStore(
                    "memory://x/uri-demo?dim=" + dim + "&metric=l2")) {
                check("uri open dim", vs.dim() == dim);
                vs.ensureCollection();
            }
        });

        // ── D2: toMemoryVectorStore ───────────────────────────────────────
        section("D2 toMemoryVectorStore + fromVectorStore");
        DataFrame base = seedVectors(n, dim, 42);

        benchmark("3. toMemoryVectorStore upsert", () -> {
            try (VectorStore vs = base.toMemoryVectorStore(
                    "clips", "id", "emb", dim, VectorMetric.COSINE, "title", "year", "category")) {
                long c = vs.count();
                check("count after upsert", c == n || c == -1, "count=" + c);
                // self-search: first vector should rank #1
                Object emb0 = base.get(0, "emb");
                float[] q = emb0 instanceof float[] f ? f : null;
                check("emb0 is float[]", q != null && q.length == dim);
                DataFrame hits = DataFrame.vectorSearch(vs, q, 5);
                check("knn rows", hits.rowCount() == 5, "rows=" + hits.rowCount());
                Object topId = hits.get(0, "id");
                check("knn top is self", "id-0".equals(String.valueOf(topId)), "top=" + topId);
            }
        });

        benchmark("4. fromVectorStore / readVectorStore", () -> {
            try (VectorStore vs = base.toMemoryVectorStore(
                    "scroll-demo", "id", "emb", dim, "title", "category")) {
                DataFrame page = DataFrame.fromVectorStore(vs, 50);
                check("fromVectorStore(50) rows", page.rowCount() == 50, "rows=" + page.rowCount());
                check("has vector col", page.hasColumn("vector") || page.hasColumn("emb"));
                check("has id", page.hasColumn("id"));

                // scheme open + read (fresh empty store — just API smoke)
                DataFrame empty = DataFrame.readVectorStore("memory",
                        Map.of("name", "empty-read", "dim", dim, "metric", "cosine"), 10);
                check("read empty ok", empty != null && empty.rowCount() >= 0);
            }
        });

        // ── D3: toVectorStore(uri) / scheme ───────────────────────────────
        section("D3 toVectorStore(uri|scheme) + searchVectorStore");
        benchmark("5. toVectorStore(uri)", () -> {
            String uri = "memory://x/uri-write?dim=" + dim + "&metric=cosine";
            try (VectorStore vs = base.toVectorStore(uri, "id", "emb", "title")) {
                check("uri write count", vs.count() == n || vs.count() == -1, "count=" + vs.count());
                float[] q = (float[]) base.get(1, "emb");
                DataFrame hits = DataFrame.fromVectorSearch(vs, q, 3);
                check("fromVectorSearch rows", hits.rowCount() == 3);
                check("top is id-1", "id-1".equals(String.valueOf(hits.get(0, "id"))),
                        "top=" + hits.get(0, "id"));
            }
        });

        benchmark("6. toVectorStore(scheme, config)", () -> {
            Map<String, Object> cfg = new LinkedHashMap<>();
            cfg.put("name", "scheme-write");
            cfg.put("dim", dim);
            cfg.put("metric", "l2");
            try (VectorStore vs = base.toVectorStore("memory", cfg, "id", "emb", "category")) {
                check("scheme write count", vs.count() == n || vs.count() == -1);
            }
        });

        // ── D4: scale ─────────────────────────────────────────────────────
        section("D4 scale upsert + knn + scroll");
        benchmark("7. scale 2k upsert / knn / scroll", () -> {
            int bigN = 2_000;
            DataFrame big = seedVectors(bigN, dim, 7);
            long t0 = System.nanoTime();
            try (VectorStore vs = big.toMemoryVectorStore(
                    "scale", "id", "emb", dim, VectorMetric.L2, "title", "year")) {
                long upsertMs = (System.nanoTime() - t0) / 1_000_000L;
                check("scale count", vs.count() == bigN || vs.count() == -1, "count=" + vs.count());
                System.out.println("    upsert " + bigN + " in " + upsertMs + " ms ("
                        + (bigN * 1000.0 / Math.max(1, upsertMs)) + " pts/s)");

                float[] q = (float[]) big.get(0, "emb");
                t0 = System.nanoTime();
                VectorSearchResult top = vs.search(q, 10);
                long knnMs = (System.nanoTime() - t0) / 1_000_000L;
                check("knn 10", top.size() == 10);
                check("knn self", "id-0".equals(top.get(0).id()), "top=" + top.get(0).id());
                System.out.println("    knn top-10: " + knnMs + " ms");

                t0 = System.nanoTime();
                DataFrame all = DataFrame.fromVectorStore(vs, 500);
                long scrollMs = (System.nanoTime() - t0) / 1_000_000L;
                check("scroll 500", all.rowCount() == 500);
                System.out.println("    scroll 500 → DataFrame: " + scrollMs + " ms");
            }
        });

        // ── D5: remote builder smoke (no network I/O) ─────────────────────
        section("D5 remote backend builders (construct only)");
        benchmark("8. builder recipes for all backends", () -> {
            List<VectorStore> opened = new ArrayList<>();
            try {
                opened.add(VectorStores.qdrant("http://localhost:6333", "clips", dim, VectorMetric.COSINE));
                opened.add(VectorStores.redis("127.0.0.1", 6379, "idx:clips", dim, VectorMetric.COSINE));
                opened.add(VectorStores.milvus("http://localhost:9091", "clips", dim, VectorMetric.COSINE));
                opened.add(VectorStores.openSearch("http://localhost:9200", "clips", dim, VectorMetric.L2));
                opened.add(VectorStores.pgvector(
                        "jdbc:postgresql://localhost:5432/vec", "user", "pass",
                        "clips", dim, VectorMetric.COSINE));
                opened.add(VectorStores.mongoAtlas(
                        "https://data.mongodb-api.example.com/app/x/endpoint/data/v1",
                        "key", "Cluster0", "db", "clips", dim, VectorMetric.COSINE));

                // DataFrame convenience factories also construct without I/O
                DataFrame tiny = seedVectors(2, dim, 1);
                // Don't call ensureCollection/upsert — that would hit network.
                // Just verify openVectorStore for each scheme returns non-null.
                check("open qdrant", DataFrame.openVectorStore("qdrant", Map.of(
                        "url", "http://localhost:6333", "collection", "c", "dim", dim)) != null);
                check("open redis", DataFrame.openVectorStore("redis", Map.of(
                        "host", "127.0.0.1", "port", 6379, "index", "idx", "dim", dim)) != null);
                check("open milvus", DataFrame.openVectorStore("milvus", Map.of(
                        "url", "http://localhost:9091", "collection", "c", "dim", dim)) != null);
                check("open opensearch", DataFrame.openVectorStore("opensearch", Map.of(
                        "url", "http://localhost:9200", "index", "c", "dim", dim)) != null);
                check("open pgvector", DataFrame.openVectorStore("pgvector", Map.of(
                        "url", "jdbc:postgresql://localhost/v", "table", "t", "dim", dim)) != null);
                check("open mongo", DataFrame.openVectorStore("mongo", Map.of(
                        "url", "https://example.com", "apiKey", "k",
                        "database", "db", "collection", "c", "dim", dim)) != null);

                check("tiny rows", tiny.rowCount() == 2);
                check("builders constructed", opened.size() == 6, "n=" + opened.size());
            } finally {
                for (VectorStore vs : opened) {
                    try { vs.close(); } catch (Throwable ignored) {}
                }
            }
        });

        // ── D6: DataFrame.toVectorStore(store, …) classic path ────────────
        section("D6 classic toVectorStore(store, …) path");
        benchmark("9. classic upsertDataFrame path", () -> {
            try (VectorStore vs = VectorStores.memory("classic", dim, VectorMetric.COSINE)) {
                vs.ensureCollection();
                DataFrame df = seedVectors(20, dim, 99);
                df.toVectorStore(vs, "id", "emb", "title");
                check("classic count", vs.count() == 20 || vs.count() == -1, "count=" + vs.count());
                DataFrame back = DataFrame.fromVectorStore(vs);
                check("classic from rows", back.rowCount() >= 20 || back.rowCount() == 20,
                        "rows=" + back.rowCount());
            }
        });

        // ── D7: optional live URI ─────────────────────────────────────────
        if (args.length > 0 && args[0] != null && !args[0].isBlank()) {
            section("D7 live round-trip: " + args[0]);
            benchmark("10. live URI round-trip", () -> {
                String uri = args[0];
                // force dim into uri if missing
                if (!uri.contains("dim=")) {
                    uri = uri + (uri.contains("?") ? "&" : "?") + "dim=" + dim;
                }
                DataFrame df = seedVectors(30, dim, 123);
                try (VectorStore vs = df.toVectorStore(uri, "id", "emb", "title", "category")) {
                    long c = vs.count();
                    check("live count>0", c > 0 || c == -1, "count=" + c);
                    float[] q = (float[]) df.get(0, "emb");
                    DataFrame hits = DataFrame.vectorSearch(vs, q, 5);
                    check("live knn rows", hits.rowCount() > 0, "rows=" + hits.rowCount());
                    System.out.println("    live top ids: " + hits.column("id"));
                }
            });
        } else {
            System.out.println("\n(pass a URI arg to hit a live backend, e.g. memory://x/demo?dim=32)");
        }

        System.out.println("\n=== summary: passed=" + passed + " failed=" + failed + " ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
