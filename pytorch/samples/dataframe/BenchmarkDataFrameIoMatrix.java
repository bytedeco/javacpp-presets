package dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.sql.SqlOptions;

import java.nio.file.*;
import java.util.*;

/**
 * Cross-format semantic equality matrix: same synthetic DF through many writers/readers.
 */
public class BenchmarkDataFrameIoMatrix {
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
            {4L, "dave", 6.0, false},
        };
        for (Object[] row : rows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < row.length; c++) df.set(ri, df.column(c).name(), row[c]);
        }
        return df;
    }

    /** Canonical comparable dump: sorted col names, stringified cells. */
    static String canonical(DataFrame df) {
        List<String> cols = new ArrayList<>(df.getColumnNames());
        Collections.sort(cols);
        StringBuilder sb = new StringBuilder();
        sb.append(cols).append('\n');
        for (int r = 0; r < df.rowCount(); r++) {
            for (int i = 0; i < cols.size(); i++) {
                if (i > 0) sb.append('|');
                Object v = df.get(r, cols.get(i));
                if (v == null) sb.append("∅");
                else if (v instanceof Number) {
                    double d = ((Number) v).doubleValue();
                    if (d == Math.rint(d) && !Double.isInfinite(d)) sb.append((long) d);
                    else sb.append(d);
                } else if (v instanceof Boolean) {
                    sb.append(((Boolean) v) ? "T" : "F");
                } else {
                    sb.append(v);
                }
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    static void assertClose(String label, DataFrame a, DataFrame b) {
        check(label + " rows", a.rowCount() == b.rowCount());
        // compare on shared numeric/string columns id,name,score
        for (String col : List.of("id", "name", "score")) {
            if (!a.hasColumn(col) || !b.hasColumn(col)) continue;
            for (int r = 0; r < a.rowCount(); r++) {
                Object va = a.get(r, col);
                Object vb = b.get(r, col);
                if (va == null && vb == null) continue;
                if (va instanceof Number && vb instanceof Number) {
                    check(label + " " + col + "[" + r + "]",
                        Math.abs(((Number) va).doubleValue() - ((Number) vb).doubleValue()) < 1e-9);
                } else {
                    check(label + " " + col + "[" + r + "]",
                        String.valueOf(va).equals(String.valueOf(vb)));
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameIoMatrix ===");
        Path tmp = Files.createTempDirectory("df-iomatrix-");
        DataFrame base = seed();
        String canon = canonical(base);
        System.out.println("Canonical:\n" + canon);

        try {
            benchmark("CSV", () -> {
                Path p = tmp.resolve("t.csv");
                base.toCsv(p.toString());
                assertClose("csv", base, DataFrame.readCsv(p.toString()));
            });
            benchmark("TSV", () -> {
                Path p = tmp.resolve("t.tsv");
                base.toTsv(p.toString());
                assertClose("tsv", base, DataFrame.readTsv(p.toString()));
            });
            benchmark("JSON", () -> {
                Path p = tmp.resolve("t.json");
                base.toJson(p.toString());
                assertClose("json", base, DataFrame.readJson(p.toString()));
            });
            benchmark("Pickle", () -> {
                Path p = tmp.resolve("t.pkl");
                base.toPickle(p.toString());
                assertClose("pkl", base, DataFrame.readPickle(p.toString()));
            });
            benchmark("Excel", () -> {
                Path p = tmp.resolve("t.xlsx");
                base.toExcel(p.toString());
                assertClose("xlsx", base, DataFrame.readExcel(p.toString()));
            });
            benchmark("SQLite", () -> {
                Path p = tmp.resolve("t.db");
                base.toSql(p.toString(), "t", SqlOptions.builder()
                    .ifExists(SqlOptions.IfExists.REPLACE).build());
                assertClose("sql", base, DataFrame.readSql(p.toString(), "SELECT * FROM t ORDER BY id"));
            });
            benchmark("Avro", () -> {
                Path p = tmp.resolve("t.avro");
                base.toAvro(p.toString());
                assertClose("avro", base, DataFrame.readAvro(p.toString()));
            });
            benchmark("HDF5", () -> {
                Path p = tmp.resolve("t.h5");
                base.toHdf(p.toString(), "/df");
                assertClose("h5", base, DataFrame.readHdf(p.toString(), "/df"));
            });
            benchmark("NPZ numeric subset", () -> {
                DataFrame num = DataFrame.create();
                num.addColumn("id", Column.DType.INT64);
                num.addColumn("score", Column.DType.FLOAT64);
                for (int r = 0; r < base.rowCount(); r++) {
                    int ri = num.addEmptyRow();
                    num.set(ri, "id", base.get(r, "id"));
                    num.set(ri, "score", base.get(r, "score"));
                }
                Path p = tmp.resolve("t.npz");
                num.toNpz(p.toString());
                DataFrame back = DataFrame.readNpz(p.toString());
                check("npz rows", back.rowCount() == num.rowCount());
            });
            benchmark("ORC", () -> {
                Path p = tmp.resolve("t.orc");
                base.toOrc(p.toString());
                assertClose("orc", base, DataFrame.readOrc(p.toString()));
            });
            benchmark("auto-detect read()", () -> {
                Path p = tmp.resolve("auto.tsv");
                base.toTsv(p.toString());
                assertClose("auto", base, DataFrame.read(p.toString()));
            });

            // Optional: parquet / arrow if native stack works in this env
            benchmark("Parquet (best-effort)", () -> {
                Path p = tmp.resolve("t.parquet");
                base.writeParquet(p.toString());
                assertClose("parquet", base, DataFrame.readParquet(p.toString()));
            });
            benchmark("Arrow/Feather (best-effort)", () -> {
                Path p = tmp.resolve("t.feather");
                base.toFeather(p.toString());
                assertClose("feather", base, DataFrame.readFeather(p.toString()));
            });

        } finally {
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder())
                    .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
            } catch (Exception ignored) {}
        }
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            // Don't hard-fail whole suite on parquet/arrow env issues alone — still exit 1 if any failed
            System.exit(1);
        }
    }
}
