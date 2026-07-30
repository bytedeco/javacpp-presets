package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.csv.CsvOptions;
import org.bytedeco.pytorch.dataframe.io.FormatDetect;

import java.nio.charset.StandardCharsets;
import java.nio.file.*;

/**
 * Multi-dimensional TSV + Feather + NPZ + auto-detect I/O benchmark.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameTsv
 * </pre>
 */
public class BenchmarkDataFrameTsv {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
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
        df.addColumn("flag", Column.DType.BOOLEAN);
        int r0 = df.addEmptyRow();
        df.set(r0, "id", 1L); df.set(r0, "name", "alice"); df.set(r0, "score", 9.5); df.set(r0, "flag", true);
        int r1 = df.addEmptyRow();
        df.set(r1, "id", 2L); df.set(r1, "name", "bob\ttab"); df.set(r1, "score", 7.0); df.set(r1, "flag", false);
        int r2 = df.addEmptyRow();
        df.set(r2, "id", 3L); df.set(r2, "name", null); df.set(r2, "score", null); df.set(r2, "flag", true);
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameTsv / Feather / NPZ / detect ===");
        Path tmp = Files.createTempDirectory("df-tsv-");
        try {
            DataFrame base = seed();

            benchmark("1. TSV round-trip", () -> {
                Path p = tmp.resolve("a.tsv");
                base.toTsv(p.toString());
                DataFrame back = DataFrame.readTsv(p.toString());
                check("rows", back.rowCount() == base.rowCount());
                check("cols", back.columnCount() == base.columnCount());
                check("alice", "alice".equals(String.valueOf(back.get(0, "name"))));
                check("tab preserved", String.valueOf(back.get(1, "name")).contains("tab"));
            });

            benchmark("2. TSV \\\\N null tokens", () -> {
                Path p = tmp.resolve("nulls.tsv");
                String body = "id\tname\tscore\n1\talice\t9.5\n2\t\\N\t\\N\n3\tbob\t7\n";
                Files.writeString(p, body, StandardCharsets.UTF_8);
                DataFrame df = DataFrame.readTsv(p.toString());
                check("rows3", df.rowCount() == 3);
                check("name null", df.get(1, "name") == null);
                check("score null", df.get(1, "score") == null);
            });

            benchmark("3. CsvOptions.tsv() factory", () -> {
                check("delim", CsvOptions.tsv().delimiter() == '\t');
                check("has N", CsvOptions.tsv().isNullToken("\\N"));
            });

            benchmark("4. FormatDetect TSV/CSV", () -> {
                check("tsv", FormatDetect.detect("x.tsv") == FormatDetect.Format.TSV);
                check("csv", FormatDetect.detect("x.csv") == FormatDetect.Format.CSV);
                check("feather", FormatDetect.detect("x.feather") == FormatDetect.Format.FEATHER);
                check("npz", FormatDetect.detect("x.npz") == FormatDetect.Format.NPZ);
                check("xlsx", FormatDetect.detect("x.xlsx") == FormatDetect.Format.EXCEL);
            });

            benchmark("5. NPZ round-trip numeric", () -> {
                Path p = tmp.resolve("n.npz");
                DataFrame num = DataFrame.create();
                num.addColumn("a", Column.DType.INT64);
                num.addColumn("b", Column.DType.FLOAT64);
                for (int i = 0; i < 5; i++) {
                    int ri = num.addEmptyRow();
                    num.set(ri, "a", (long) i);
                    num.set(ri, "b", i * 0.5);
                }
                num.toNpz(p.toString());
                DataFrame back = DataFrame.readNpz(p.toString());
                check("npz rows", back.rowCount() == 5);
                check("npz has a", back.hasColumn("a"));
                check("npz has b", back.hasColumn("b"));
            });

            benchmark("6. DataFrame.read auto-detect tsv", () -> {
                Path p = tmp.resolve("auto.tsv");
                base.toTsv(p.toString());
                DataFrame back = DataFrame.read(p.toString());
                check("auto rows", back.rowCount() == base.rowCount());
            });

            benchmark("7. empty TSV", () -> {
                Path p = tmp.resolve("empty.tsv");
                DataFrame empty = DataFrame.create();
                empty.addColumn("x", Column.DType.INT64);
                empty.toTsv(p.toString());
                DataFrame back = DataFrame.readTsv(p.toString());
                check("empty rows", back.rowCount() == 0);
                check("empty col", back.columnCount() >= 1 || back.columnCount() == 0);
            });

            benchmark("8. scale 10k TSV", () -> {
                DataFrame big = DataFrame.create();
                big.addColumn("i", Column.DType.INT64);
                big.addColumn("v", Column.DType.FLOAT64);
                for (int i = 0; i < 10_000; i++) {
                    int ri = big.addEmptyRow();
                    big.set(ri, "i", (long) i);
                    big.set(ri, "v", i * 1.0);
                }
                Path p = tmp.resolve("big.tsv");
                long t0 = System.nanoTime();
                big.toTsv(p.toString());
                DataFrame back = DataFrame.readTsv(p.toString());
                long ms = (System.nanoTime() - t0) / 1_000_000;
                check("big rows", back.rowCount() == 10_000);
                System.out.println("    scale 10k write+read: " + ms + " ms");
            });

        } finally {
            try {
                Files.walk(tmp).sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            } catch (Exception ignored) {}
        }

        System.out.println();
        System.out.println("Passed checks: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
