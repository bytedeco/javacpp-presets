package dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.hdf5.Hdf5Options;

import java.nio.file.*;

/**
 * HDF5 columnar layout round-trip benchmark (jhdf).
 */
public class BenchmarkDataFrameHdf5 {
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
        System.out.println("=== BenchmarkDataFrameHdf5 ===");
        Path tmp = Files.createTempDirectory("df-h5-");
        try {
            DataFrame base = seed();

            benchmark("1. columnar round-trip", () -> {
                Path p = tmp.resolve("a.h5");
                base.toHdf(p.toString(), "/df");
                DataFrame back = DataFrame.readHdf(p.toString(), "/df");
                check("rows", back.rowCount() == 3);
                check("has id", back.hasColumn("id"));
                check("has name", back.hasColumn("name"));
                check("alice", "alice".equals(String.valueOf(back.get(0, "name"))));
                check("score", back.get(0, "score") instanceof Number);
            });

            benchmark("2. nested key path", () -> {
                Path p = tmp.resolve("nested.h5");
                base.toHdf(p.toString(), "/data/table1");
                DataFrame back = DataFrame.readHdf(p.toString(), "/data/table1");
                check("rows", back.rowCount() == 3);
            });

            benchmark("3. missing key error", () -> {
                Path p = tmp.resolve("a.h5");
                base.toHdf(p.toString(), "/df");
                boolean threw = false;
                try {
                    DataFrame.readHdf(p.toString(), "/nope");
                } catch (Exception e) {
                    threw = true;
                }
                check("throws", threw);
            });

            benchmark("4. numeric-only matrix format", () -> {
                DataFrame num = DataFrame.create();
                num.addColumn("a", Column.DType.FLOAT64);
                num.addColumn("b", Column.DType.FLOAT64);
                for (int i = 0; i < 4; i++) {
                    int ri = num.addEmptyRow();
                    num.set(ri, "a", i * 1.0);
                    num.set(ri, "b", i * 2.0);
                }
                Path p = tmp.resolve("mat.h5");
                num.toHdf(p.toString(), "/m", Hdf5Options.builder()
                    .format(Hdf5Options.Format.MATRIX).build());
                // matrix writes "values" dataset under group
                DataFrame back = DataFrame.readHdf(p.toString(), "/m");
                check("matrix rows or cols present", back.rowCount() >= 1 || back.columnCount() >= 1);
            });

            benchmark("5. scale 5k", () -> {
                DataFrame big = DataFrame.create();
                big.addColumn("i", Column.DType.INT64);
                big.addColumn("v", Column.DType.FLOAT64);
                for (int i = 0; i < 5000; i++) {
                    int ri = big.addEmptyRow();
                    big.set(ri, "i", (long) i);
                    big.set(ri, "v", i * 0.5);
                }
                Path p = tmp.resolve("big.h5");
                long t0 = System.nanoTime();
                big.toHdf(p.toString(), "/df");
                DataFrame back = DataFrame.readHdf(p.toString(), "/df");
                long ms = (System.nanoTime() - t0) / 1_000_000;
                check("big rows", back.rowCount() == 5000);
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
