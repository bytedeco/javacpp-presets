package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.sql.SqlOptions;
import org.bytedeco.pytorch.utils.duckdb.DuckDB;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Dedicated DataFrame ↔ DuckDB I/O benchmark covering the new convenience APIs:
 * <ul>
 *   <li>{@code readDuckDB*} / {@code writeDuckDB*} / {@code toDuckDB} / {@code openDuckDB}</li>
 *   <li>Parquet / CSV / JSON export + scan via DuckDB table functions</li>
 *   <li>Persistent {@code .duckdb} file table round-trip</li>
 *   <li>{@code duckDBQuery} register + aggregate</li>
 *   <li>Scale write/read (10k rows)</li>
 * </ul>
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameDuckDB
 * </pre>
 */
public class BenchmarkDataFrameDuckDB {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<String> skipped = new ArrayList<>();

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
                {4L, "dave", 6.5, false},
                {5L, "erin", 9.0, true},
        };
        for (Object[] row : rows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < row.length; c++) {
                df.set(ri, df.column(c).name(), row[c]);
            }
        }
        return df;
    }

    static DataFrame scale(int n) {
        DataFrame df = DataFrame.create();
        df.addColumn("i", Column.DType.INT64);
        df.addColumn("v", Column.DType.FLOAT64);
        df.addColumn("label", Column.DType.STRING);
        for (int i = 0; i < n; i++) {
            int ri = df.addEmptyRow();
            df.set(ri, "i", (long) i);
            df.set(ri, "v", i * 0.1);
            df.set(ri, "label", "g" + (i % 10));
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameDuckDB ===");
        Path tmp = Files.createTempDirectory("df-duckdb-");
        try {
            // ── D1: driver / version ──────────────────────────────────────
            section("D1 DuckDB driver / version");
            benchmark("1. open in-memory + version", () -> {
                try (DuckDB db = DataFrame.openDuckDB()) {
                    String v = db.duckdbVersion();
                    check("version non-empty", v != null && !v.isBlank(), "v=" + v);
                    check("url memory", db.url() != null && db.url().startsWith("jdbc:duckdb"));
                }
            });

            // ── D2: register + query ──────────────────────────────────────
            section("D2 DataFrame.register / duckDBQuery");
            DataFrame base = seed();
            benchmark("2. duckDBQuery aggregate", () -> {
                DataFrame agg = base.duckDBQuery("t",
                        "SELECT COUNT(*) AS n, AVG(score) AS avg_score FROM t WHERE ok = true");
                check("agg rows", agg.rowCount() == 1);
                Object n = agg.get(0, "n");
                check("ok count=3", n instanceof Number && ((Number) n).intValue() == 3,
                        "n=" + n);
            });

            benchmark("3. DuckDB.register + tableToDataFrame", () -> {
                try (DuckDB db = DuckDB.inMemory()) {
                    db.register("people", base);
                    DataFrame back = db.tableToDataFrame("people");
                    check("round-trip rows", back.rowCount() == base.rowCount());
                    check("has name", back.hasColumn("name"));
                    DataFrame filtered = db.query(
                            "SELECT name, score FROM people WHERE score >= 8.0 ORDER BY score DESC");
                    check("filtered 3", filtered.rowCount() == 3, "rows=" + filtered.rowCount());
                }
            });

            // ── D3: file scans (parquet/csv/json) ─────────────────────────
            section("D3 DuckDB file export + scan");
            Path parquet = tmp.resolve("seed.parquet");
            Path csv = tmp.resolve("seed.csv");
            Path json = tmp.resolve("seed.json");

            benchmark("4. writeDuckDBParquet + readDuckDBParquet", () -> {
                base.writeDuckDBParquet(parquet.toString());
                check("parquet exists", Files.isRegularFile(parquet));
                DataFrame back = DataFrame.readDuckDBParquet(parquet.toString());
                check("parquet rows", back.rowCount() == base.rowCount(),
                        "rows=" + back.rowCount());
                check("parquet cols", back.columnCount() == base.columnCount());
            });

            benchmark("5. writeDuckDBCsv + readDuckDBCsv", () -> {
                base.writeDuckDBCsv(csv.toString());
                check("csv exists", Files.isRegularFile(csv));
                DataFrame back = DataFrame.readDuckDBCsv(csv.toString());
                check("csv rows", back.rowCount() == base.rowCount(), "rows=" + back.rowCount());
            });

            benchmark("6. writeDuckDBJson + readDuckDBJson", () -> {
                base.writeDuckDBJson(json.toString());
                check("json exists", Files.isRegularFile(json));
                DataFrame back = DataFrame.readDuckDBJson(json.toString());
                check("json rows", back.rowCount() == base.rowCount(), "rows=" + back.rowCount());
            });

            benchmark("7. readDuckDB auto-detect by extension", () -> {
                DataFrame p = DataFrame.readDuckDB(parquet.toString());
                DataFrame c = DataFrame.readDuckDB(csv.toString());
                check("auto parquet", p.rowCount() == base.rowCount());
                check("auto csv", c.rowCount() == base.rowCount());
            });

            // ── D4: persistent .duckdb file ───────────────────────────────
            section("D4 persistent DuckDB database file");
            Path dbFile = tmp.resolve("warehouse.duckdb");

            benchmark("8. toDuckDB + readDuckDBTable", () -> {
                base.toDuckDB(dbFile.toString(), "people");
                check("db file exists", Files.isRegularFile(dbFile));
                DataFrame back = DataFrame.readDuckDBTable(dbFile.toString(), "people");
                check("table rows", back.rowCount() == base.rowCount());
                check("alice", "alice".equals(String.valueOf(back.get(0, "name")))
                        || containsName(back, "alice"));
            });

            benchmark("9. writeDuckDB alias + SQL over file", () -> {
                base.writeDuckDB(dbFile.toString(), "people2");
                DataFrame q = DataFrame.readDuckDB(dbFile,
                        "SELECT name FROM people2 WHERE score > 8 ORDER BY name");
                check("sql over file rows>=2", q.rowCount() >= 2, "rows=" + q.rowCount());
            });

            benchmark("10. APPEND into existing duckdb table", () -> {
                base.toDuckDB(dbFile, "appended",
                        SqlOptions.builder().ifExists(SqlOptions.IfExists.REPLACE).build());
                base.toDuckDB(dbFile, "appended",
                        SqlOptions.builder().ifExists(SqlOptions.IfExists.APPEND).build());
                DataFrame back = DataFrame.readDuckDBTable(dbFile.toString(), "appended");
                check("appended 10", back.rowCount() == base.rowCount() * 2,
                        "rows=" + back.rowCount());
            });

            benchmark("11. openDuckDB(path) catalog", () -> {
                try (DuckDB db = DataFrame.openDuckDB(dbFile.toString())) {
                    DataFrame tables = db.showTables();
                    check("showTables rows>0", tables.rowCount() > 0, "n=" + tables.rowCount());
                    List<String> names = db.tables();
                    check("tables has people", names.stream()
                            .anyMatch(t -> t != null && t.toLowerCase(Locale.ROOT).contains("people")));
                }
            });

            // ── D5: SQL sugar ─────────────────────────────────────────────
            section("D5 readDuckDBSql / scan helpers");
            benchmark("12. readDuckDBSql pure SQL", () -> {
                DataFrame df = DataFrame.readDuckDBSql(
                        "SELECT 1 AS a, 2.5 AS b, 'x' AS c");
                check("literal rows", df.rowCount() == 1);
                check("literal cols", df.columnCount() == 3);
            });

            // ── D6: scale ─────────────────────────────────────────────────
            section("D6 scale 10k write/read");
            benchmark("13. scale 10k parquet via DuckDB", () -> {
                int n = 10_000;
                DataFrame big = scale(n);
                Path out = tmp.resolve("big.parquet");
                long t0 = System.nanoTime();
                big.writeDuckDBParquet(out.toString());
                long wMs = (System.nanoTime() - t0) / 1_000_000L;
                t0 = System.nanoTime();
                DataFrame back = DataFrame.readDuckDBParquet(out.toString());
                long rMs = (System.nanoTime() - t0) / 1_000_000L;
                check("scale rows", back.rowCount() == n, "rows=" + back.rowCount());
                System.out.println("    write " + n + " rows: " + wMs + " ms; read: " + rMs + " ms"
                        + " (" + (n * 1000.0 / Math.max(1, wMs + rMs)) + " rows/s round-trip)");
            });

            benchmark("14. scale 10k duckdb table file", () -> {
                int n = 10_000;
                DataFrame big = scale(n);
                Path out = tmp.resolve("big.duckdb");
                long t0 = System.nanoTime();
                big.toDuckDB(out.toString(), "big");
                long wMs = (System.nanoTime() - t0) / 1_000_000L;
                t0 = System.nanoTime();
                DataFrame back = DataFrame.readDuckDB(out,
                        "SELECT COUNT(*) AS n, AVG(v) AS avg_v FROM big");
                long rMs = (System.nanoTime() - t0) / 1_000_000L;
                Object cnt = back.get(0, "n");
                check("count 10k", cnt instanceof Number && ((Number) cnt).intValue() == n,
                        "n=" + cnt);
                System.out.println("    toDuckDB " + n + ": " + wMs + " ms; aggregate: " + rMs + " ms");
            });

            // ── D7: cross-format via DuckDB ───────────────────────────────
            section("D7 cross-format: DataFrame → parquet → csv via DuckDB");
            benchmark("15. register parquet view → export csv", () -> {
                try (DuckDB db = DuckDB.inMemory()) {
                    db.registerParquet("p", parquet.toString());
                    Path outCsv = tmp.resolve("from_parquet.csv");
                    db.exportCsv("SELECT * FROM p WHERE score > 7", outCsv.toString());
                    check("export csv exists", Files.isRegularFile(outCsv));
                    DataFrame back = DataFrame.readDuckDBCsv(outCsv.toString());
                    check("cross-format rows>0", back.rowCount() > 0, "rows=" + back.rowCount());
                }
            });

        } finally {
            try {
                Files.walk(tmp).sorted(java.util.Comparator.reverseOrder())
                        .forEach(p -> {
                            try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                        });
            } catch (Exception ignored) {}
        }

        System.out.println("\n=== summary: passed=" + passed + " failed=" + failed
                + (skipped.isEmpty() ? "" : " skipped=" + skipped.size()) + " ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }

    private static boolean containsName(DataFrame df, String name) {
        if (!df.hasColumn("name")) return false;
        for (int i = 0; i < df.rowCount(); i++) {
            if (name.equals(String.valueOf(df.get(i, "name")))) return true;
        }
        return false;
    }
}
