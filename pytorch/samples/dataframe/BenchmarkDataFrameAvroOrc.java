package dataframe;

import org.bytedeco.pytorch.data.avro.AvroOptions;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.data.orc.OrcOptions;

import java.nio.file.*;
import java.time.LocalDate;

/**
 * Avro + ORC DataFrame I/O benchmark.
 */
public class BenchmarkDataFrameAvroOrc {
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
        df.addColumn("when", Column.DType.DATE);
        Object[][] rows = {
            {1L, "alice", 9.5, true, LocalDate.of(2024, 1, 15)},
            {2L, "bob", 7.0, false, LocalDate.of(2024, 6, 1)},
            {3L, "carol", 8.25, true, LocalDate.of(2025, 3, 20)},
        };
        for (Object[] row : rows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < row.length; c++) df.set(ri, df.column(c).name(), row[c]);
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameAvroOrc ===");
        Path tmp = Files.createTempDirectory("df-avro-orc-");
        try {
            DataFrame base = seed();

            benchmark("1. Avro round-trip", () -> {
                Path p = tmp.resolve("a.avro");
                base.toAvro(p.toString());
                DataFrame back = DataFrame.readAvro(p.toString());
                check("rows", back.rowCount() == 3);
                check("alice", "alice".equals(String.valueOf(back.get(0, "name"))));
                check("score", back.get(0, "score") instanceof Number);
                check("ok bool-ish", back.get(0, "ok") != null);
            });

            benchmark("2. Avro deflate codec", () -> {
                Path p = tmp.resolve("deflate.avro");
                base.toAvro(p.toString(), AvroOptions.builder()
                    .codec(AvroOptions.Codec.DEFLATE).build());
                DataFrame back = DataFrame.readAvro(p.toString());
                check("rows", back.rowCount() == 3);
            });

            benchmark("3. Avro nulls", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("a", Column.DType.INT64);
                df.addColumn("b", Column.DType.STRING);
                int r0 = df.addEmptyRow();
                df.set(r0, "a", 1L); df.set(r0, "b", null);
                int r1 = df.addEmptyRow();
                df.set(r1, "a", null); df.set(r1, "b", "x");
                Path p = tmp.resolve("nulls.avro");
                df.toAvro(p.toString());
                DataFrame back = DataFrame.readAvro(p.toString());
                check("rows2", back.rowCount() == 2);
                check("b0 null", back.get(0, "b") == null);
                check("a1 null", back.get(1, "a") == null);
            });

            benchmark("4. ORC round-trip", () -> {
                Path p = tmp.resolve("a.orc");
                base.toOrc(p.toString());
                DataFrame back = DataFrame.readOrc(p.toString());
                check("rows", back.rowCount() == 3);
                check("alice", "alice".equals(String.valueOf(back.get(0, "name"))));
                check("id", ((Number) back.get(0, "id")).longValue() == 1L);
            });

            benchmark("5. ORC snappy + options", () -> {
                Path p = tmp.resolve("snappy.orc");
                base.toOrc(p.toString(), OrcOptions.builder()
                    .compress(OrcOptions.Compress.SNAPPY)
                    .batchSize(64)
                    .build());
                DataFrame back = DataFrame.readOrc(p.toString(),
                    OrcOptions.builder().batchSize(64).build());
                check("rows", back.rowCount() == 3);
            });

            benchmark("6. ORC maxRows", () -> {
                Path p = tmp.resolve("a.orc");
                base.toOrc(p.toString());
                DataFrame back = DataFrame.readOrc(p.toString(),
                    OrcOptions.builder().maxRows(2).build());
                check("max 2", back.rowCount() == 2);
            });

            benchmark("7. scale avro 5k", () -> {
                DataFrame big = DataFrame.create();
                big.addColumn("i", Column.DType.INT64);
                big.addColumn("v", Column.DType.FLOAT64);
                for (int i = 0; i < 5000; i++) {
                    int ri = big.addEmptyRow();
                    big.set(ri, "i", (long) i);
                    big.set(ri, "v", i * 0.1);
                }
                Path p = tmp.resolve("big.avro");
                long t0 = System.nanoTime();
                big.toAvro(p.toString());
                DataFrame back = DataFrame.readAvro(p.toString());
                long ms = (System.nanoTime() - t0) / 1_000_000;
                check("big rows", back.rowCount() == 5000);
                System.out.println("    scale avro 5k: " + ms + " ms");
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
