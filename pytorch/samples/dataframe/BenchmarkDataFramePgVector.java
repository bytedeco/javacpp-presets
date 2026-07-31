package dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.pgvector.PgVector;
import org.bytedeco.pytorch.dataframe.pgvector.PgVectorBackend;
import org.bytedeco.pytorch.dataframe.pgvector.PgVectorException;
import org.bytedeco.pytorch.dataframe.pgvector.PgVectorOptions;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreProvider;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStores;
import org.bytedeco.pytorch.dataframe.vectorstore.memory.InMemoryVectorStore;

import java.util.Map;
import java.util.Random;

/**
 * DataFrame ↔ pgvector full-client benchmark (JDBC, no extra SDK).
 *
 * <p>Env: {@code PGVECTOR_JDBC} (optional {@code PGVECTOR_USER}/{@code PGVECTOR_PASSWORD}).
 * Soft-skips when the PostgreSQL driver or server is unavailable.
 */
public class BenchmarkDataFramePgVector {

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
        Random rnd = new Random(13);
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("emb", Column.DType.VECTOR);
        df.addColumn("title", Column.DType.STRING);
        df.addColumn("score", Column.DType.FLOAT64);
        for (int i = 0; i < n; i++) {
            int r = df.addEmptyRow();
            df.set(r, "id", "id-" + i);
            df.set(r, "emb", unit(rnd, dim));
            df.set(r, "title", "doc-" + i);
            df.set(r, "score", i * 0.1);
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFramePgVector ===");
        final int dim = 32;

        section("D0 SPI override smoke");
        benchmark("1. VectorStoreProvider overrides pgvector", () -> {
            VectorStores.registerProvider(new VectorStoreProvider() {
                @Override public String name() { return "pgvector"; }
                @Override public VectorStore open(Map<String, Object> config) {
                    int d = config.get("dim") instanceof Number n ? n.intValue() : dim;
                    return InMemoryVectorStore.builder(d).name("spi-pg").metric(VectorMetric.COSINE).build();
                }
            });
            try (VectorStore vs = VectorStores.open("pgvector", Map.of(
                    "dim", dim, "url", "jdbc:postgresql://localhost/x", "table", "t"))) {
                check("spi backend memory", "memory".equals(vs.backend()), vs.backend());
            } finally {
                VectorStores.reloadProviders();
            }
        });

        benchmark("2. PgVectorBackend register", () -> {
            PgVector.registerBackend(new PgVectorBackend() {
                @Override public String name() { return "pgvector-test-only"; }
                @Override public PgVector open(Map<String, Object> config) {
                    return PgVector.openBuiltin(Map.of(
                        "url", "jdbc:postgresql://127.0.0.1:1/postgres",
                        "user", "u", "password", "p"));
                }
            });
            check("registered", PgVector.backend("pgvector-test-only") != null);
        });

        section("D1 Options + factory + vector literals");
        benchmark("3. PgVectorOptions fluent + fromMap", () -> {
            PgVectorOptions o = PgVectorOptions.builder()
                .table("docs").dim(dim).metric(VectorMetric.COSINE)
                .idColumn("id").vectorColumn("emb")
                .payloadMode(PgVectorOptions.PayloadMode.JSONB)
                .indexMethod(PgVectorOptions.IndexMethod.HNSW)
                .chunksize(100)
                .build();
            check("table", "docs".equals(o.table()));
            check("mode jsonb", o.payloadMode() == PgVectorOptions.PayloadMode.JSONB);
            check("hnsw", o.indexMethod() == PgVectorOptions.IndexMethod.HNSW);
            PgVectorOptions o2 = PgVectorOptions.fromMap(Map.of(
                "table", "t2", "dim", dim, "metric", "l2", "payloadMode", "columns"));
            check("fromMap table", "t2".equals(o2.table()));
            check("fromMap columns", o2.payloadMode() == PgVectorOptions.PayloadMode.COLUMNS);
            check("fromMap l2", o2.metric() == VectorMetric.L2);
        });

        benchmark("4. vector literal encode/decode", () -> {
            float[] v = new float[]{0.1f, -0.2f, 0.3f};
            String lit = PgVector.toVectorLiteral(v);
            check("literal format", lit.startsWith("[") && lit.endsWith("]"), lit);
            float[] back = PgVector.parseVector(lit);
            check("roundtrip len", back != null && back.length == 3);
            check("roundtrip val", Math.abs(back[1] + 0.2f) < 1e-5, "v1=" + back[1]);
        });

        benchmark("5. connectUri parse + builder", () -> {
            try (PgVector pg = PgVector.builder()
                    .url("jdbc:postgresql://localhost:5432/postgres")
                    .user("postgres").password("postgres").build()) {
                check("jdbc set", pg.jdbcUrl() != null && pg.jdbcUrl().contains("postgresql"));
            }
            try (PgVector pg = PgVector.connectUri(
                    "pgvector://postgres:secret@localhost:5432/postgres?table=docs")) {
                check("connectUri non-null", pg != null);
                check("url rebuilt", pg.jdbcUrl() != null && pg.jdbcUrl().startsWith("jdbc:postgresql://"));
            }
        });

        benchmark("6. DataFrame.toPgVector(PgVector, Options) present", () -> {
            boolean ok = false;
            for (var m : DataFrame.class.getMethods()) {
                if ("toPgVector".equals(m.getName()) && m.getParameterCount() == 2
                        && m.getParameterTypes()[0] == PgVector.class) {
                    ok = true; break;
                }
            }
            check("overload present", ok);
            // classic vector-store path still present
            boolean classic = false;
            for (var m : DataFrame.class.getMethods()) {
                if ("toPgVector".equals(m.getName()) && m.getParameterCount() >= 7) {
                    classic = true; break;
                }
            }
            check("classic VectorStore overload present", classic);
        });

        String jdbc = System.getenv("PGVECTOR_JDBC");
        if ((jdbc == null || jdbc.isBlank()) && args.length > 0) jdbc = args[0];
        String user = System.getenv("PGVECTOR_USER");
        if (user == null) user = System.getenv("PGUSER");
        String pass = System.getenv("PGVECTOR_PASSWORD");
        if (pass == null) pass = System.getenv("PGPASSWORD");

        section("D2–D5 live pgvector");
        if (jdbc == null || jdbc.isBlank()) {
            skip("live suite", "set PGVECTOR_JDBC (optional PGVECTOR_USER/PASSWORD)");
        } else {
            // probe driver
            boolean driverOk = false;
            try {
                Class.forName("org.postgresql.Driver");
                driverOk = true;
            } catch (ClassNotFoundException e) {
                skip("live suite", "org.postgresql.Driver not on classpath");
            }
            if (driverOk) {
                final String liveJdbc = jdbc;
                final String liveUser = user;
                final String livePass = pass;
                final String table = "df_bench_" + System.currentTimeMillis();
                try (PgVector pg = (liveUser != null)
                        ? PgVector.connect(liveJdbc, liveUser, livePass)
                        : PgVector.connect(liveJdbc)) {
                    DataFrame base = seed(40, dim);
                    benchmark("7. ensure extension + table + write JSONB", () -> {
                        PgVectorOptions opts = PgVectorOptions.builder()
                            .table(table).dim(dim).metric(VectorMetric.COSINE)
                            .idColumn("id").vectorColumn("emb")
                            .payloadMode(PgVectorOptions.PayloadMode.JSONB)
                            .ifExists(PgVectorOptions.IfExists.REPLACE)
                            .build();
                        int n = base.toPgVector(pg, opts);
                        check("written 40", n == 40, "n=" + n);
                        long c = pg.count(opts);
                        check("count>=40", c >= 40 || c == -1, "count=" + c);
                    });
                    benchmark("8. readDataFrame JSONB", () -> {
                        DataFrame back = DataFrame.readPgVector(pg, PgVectorOptions.builder()
                            .table(table).limit(100).build());
                        check("rows>0", back.rowCount() > 0, "rows=" + back.rowCount());
                        check("has id", back.hasColumn("id") || back.hasColumn("id".equals("id") ? "id" : "id"));
                    });
                    benchmark("9. knn search", () -> {
                        float[] q = (float[]) base.get(0, "emb");
                        DataFrame hits = DataFrame.searchPgVector(pg, q, 5, PgVectorOptions.builder()
                            .table(table).metric(VectorMetric.COSINE).build());
                        check("hits>0", hits.rowCount() > 0, "rows=" + hits.rowCount());
                    });
                    benchmark("10. scale 500 upsert", () -> {
                        DataFrame big = seed(500, dim);
                        PgVectorOptions opts = PgVectorOptions.builder()
                            .table(table).dim(dim).idColumn("id").vectorColumn("emb")
                            .ifExists(PgVectorOptions.IfExists.REPLACE)
                            .chunksize(100).build();
                        int n = big.toPgVector(pg, opts);
                        check("written 500", n == 500, "n=" + n);
                    });
                    benchmark("11. drop table", () -> {
                        pg.dropTable(PgVectorOptions.table(table));
                        check("gone", !pg.tableExists(table));
                    });
                } catch (PgVectorException e) {
                    skip("live suite", "connection failed: " + e.getMessage());
                }
            }
        }

        System.out.println("\n=== RESULT passed=" + passed + " failed=" + failed + " skipped=" + skipped + " ===");
        if (failed > 0) System.exit(1);
    }
}
