package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.sql.SqlOptions;
import org.bytedeco.pytorch.dataframe.sql.Sqlite;

import java.nio.file.*;
import java.sql.Connection;

/**
 * SQLite / JDBC DataFrame I/O benchmark.
 */
public class BenchmarkDataFrameSql {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            System.out.println("  OK  " + name + " (" + ((System.nanoTime() - t0) / 1_000_000) + " ms)");
        } catch (Throwable e) {
            failed++;
            System.out.println(" FAIL " + name + ": " + e);
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

    static DataFrame seed() {
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.INT64);
        df.addColumn("name", Column.DType.STRING);
        df.addColumn("score", Column.DType.FLOAT64);
        df.addColumn("ok", Column.DType.BOOLEAN);
        Object[][] rows = {
            {1L, "alice", 9.5, true},
            {2L, "bob", 7.0, false},
            {3L, "carol", 8.25, true},
        };
        for (Object[] row : rows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < row.length; c++) df.set(ri, df.column(c).name(), row[c]);
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameSql ===");
        Path tmp = Files.createTempDirectory("df-sql-");
        try {
            DataFrame base = seed();

            benchmark("1. memory sqlite round-trip", () -> {
                try (Connection c = Sqlite.openInMemory()) {
                    base.toSql(c, "people", SqlOptions.builder()
                        .ifExists(SqlOptions.IfExists.REPLACE).build());
                    DataFrame back = DataFrame.readSqlTable(c, "people");
                    check("rows", back.rowCount() == 3);
                    check("alice", "alice".equals(String.valueOf(back.get(0, "name"))));
                    check("score", back.get(0, "score") instanceof Number);
                }
            });

            benchmark("2. file sqlite + query", () -> {
                Path db = tmp.resolve("t.db");
                base.toSql(db.toString(), "t", SqlOptions.builder()
                    .ifExists(SqlOptions.IfExists.REPLACE).build());
                DataFrame back = DataFrame.readSql(db.toString(),
                    "SELECT name, score FROM t WHERE id >= 2 ORDER BY id");
                check("filtered rows", back.rowCount() == 2);
                check("bob first", "bob".equals(String.valueOf(back.get(0, "name"))));
            });

            benchmark("3. APPEND ifExists", () -> {
                try (Connection c = Sqlite.openInMemory()) {
                    base.toSql(c, "t", SqlOptions.builder()
                        .ifExists(SqlOptions.IfExists.REPLACE).build());
                    base.toSql(c, "t", SqlOptions.builder()
                        .ifExists(SqlOptions.IfExists.APPEND).build());
                    DataFrame back = DataFrame.readSqlTable(c, "t");
                    check("appended 6", back.rowCount() == 6);
                }
            });

            benchmark("4. FAIL ifExists", () -> {
                try (Connection c = Sqlite.openInMemory()) {
                    base.toSql(c, "t", SqlOptions.builder()
                        .ifExists(SqlOptions.IfExists.REPLACE).build());
                    boolean threw = false;
                    try {
                        base.toSql(c, "t", SqlOptions.builder()
                            .ifExists(SqlOptions.IfExists.FAIL).build());
                    } catch (Exception e) {
                        threw = true;
                    }
                    check("fail throws", threw);
                }
            });

            benchmark("5. with index column", () -> {
                try (Connection c = Sqlite.openInMemory()) {
                    base.toSql(c, "t", SqlOptions.builder()
                        .ifExists(SqlOptions.IfExists.REPLACE)
                        .index(true)
                        .indexLabel("idx")
                        .build());
                    DataFrame back = DataFrame.readSql(c, "SELECT * FROM t");
                    check("has idx", back.hasColumn("idx"));
                    check("rows", back.rowCount() == 3);
                }
            });

            benchmark("6. scale 5k", () -> {
                DataFrame big = DataFrame.create();
                big.addColumn("i", Column.DType.INT64);
                big.addColumn("v", Column.DType.FLOAT64);
                for (int i = 0; i < 5000; i++) {
                    int ri = big.addEmptyRow();
                    big.set(ri, "i", (long) i);
                    big.set(ri, "v", i * 0.1);
                }
                Path db = tmp.resolve("big.db");
                long t0 = System.nanoTime();
                big.toSql(db.toString(), "big", SqlOptions.builder()
                    .ifExists(SqlOptions.IfExists.REPLACE).chunksize(500).build());
                DataFrame back = DataFrame.readSql(db.toString(), "SELECT COUNT(*) AS n FROM big");
                // count query returns 1 row
                DataFrame all = DataFrame.readSqlTable(
                    Sqlite.open(db.toString()), "big");
                all.close();
                try (Connection c = Sqlite.open(db.toString())) {
                    all = DataFrame.readSqlTable(c, "big");
                    check("big rows", all.rowCount() == 5000);
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                System.out.println("    scale 5k write+read: " + ms + " ms");
            });

        } finally {
            try {
                Files.walk(tmp).sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
            } catch (Exception ignored) {}
        }
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
