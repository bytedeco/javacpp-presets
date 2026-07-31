package dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.excel.ExcelOptions;

import java.nio.file.*;
import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Excel (.xlsx) multi-dimensional correctness benchmark.
 */
public class BenchmarkDataFrameExcel {
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
        df.addColumn("when", Column.DType.DATE);
        df.addColumn("ok", Column.DType.BOOLEAN);
        Object[][] rows = {
            {1L, "alice", 9.5, LocalDate.of(2024, 1, 15), true},
            {2L, "bob", 7.0, LocalDate.of(2024, 6, 1), false},
            {3L, null, null, LocalDate.of(2025, 3, 20), true},
        };
        for (Object[] row : rows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < row.length; c++) df.set(ri, df.column(c).name(), row[c]);
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameExcel ===");
        Path tmp = Files.createTempDirectory("df-xlsx-");
        try {
            DataFrame base = seed();

            benchmark("1. xlsx round-trip", () -> {
                Path p = tmp.resolve("a.xlsx");
                base.toExcel(p.toString());
                DataFrame back = DataFrame.readExcel(p.toString());
                check("rows", back.rowCount() == 3);
                check("cols", back.columnCount() == 5);
                check("alice", "alice".equals(String.valueOf(back.get(0, "name"))));
                check("score num", back.get(0, "score") instanceof Number);
                check("null name", back.get(2, "name") == null);
            });

            benchmark("2. sheet name + options", () -> {
                Path p = tmp.resolve("named.xlsx");
                base.toExcel(p.toString(), ExcelOptions.builder()
                    .writeSheetName("People")
                    .header(true)
                    .freezeHeader(true)
                    .build());
                DataFrame back = DataFrame.readExcel(p.toString(),
                    ExcelOptions.builder().sheet("People").build());
                check("named rows", back.rowCount() == 3);
            });

            benchmark("3. multi-sheet", () -> {
                Path p = tmp.resolve("multi.xlsx");
                Map<String, DataFrame> sheets = new LinkedHashMap<>();
                sheets.put("A", base);
                DataFrame b = DataFrame.create();
                b.addColumn("x", Column.DType.INT64);
                int ri = b.addEmptyRow();
                b.set(ri, "x", 42L);
                sheets.put("B", b);
                DataFrame.writeExcelSheets(p.toString(), sheets);
                Map<String, DataFrame> back = DataFrame.readExcelAll(p.toString());
                check("2 sheets", back.size() == 2);
                check("has A", back.containsKey("A"));
                check("has B", back.containsKey("B"));
                check("B val", ((Number) back.get("B").get(0, "x")).longValue() == 42L);
            });

            benchmark("4. missing sheet error", () -> {
                Path p = tmp.resolve("a.xlsx");
                base.toExcel(p.toString());
                boolean threw = false;
                try {
                    DataFrame.readExcel(p.toString(), ExcelOptions.builder().sheet("Nope").build());
                } catch (Exception e) {
                    threw = true;
                }
                check("throws", threw);
            });

            benchmark("5. empty frame", () -> {
                Path p = tmp.resolve("empty.xlsx");
                DataFrame empty = DataFrame.create();
                empty.addColumn("x", Column.DType.STRING);
                empty.toExcel(p.toString());
                DataFrame back = DataFrame.readExcel(p.toString());
                check("empty rows", back.rowCount() == 0);
            });

            benchmark("6. scale 2k", () -> {
                DataFrame big = DataFrame.create();
                big.addColumn("i", Column.DType.INT64);
                big.addColumn("v", Column.DType.FLOAT64);
                for (int i = 0; i < 2000; i++) {
                    int ri = big.addEmptyRow();
                    big.set(ri, "i", (long) i);
                    big.set(ri, "v", i * 1.1);
                }
                Path p = tmp.resolve("big.xlsx");
                long t0 = System.nanoTime();
                big.toExcel(p.toString());
                DataFrame back = DataFrame.readExcel(p.toString());
                long ms = (System.nanoTime() - t0) / 1_000_000;
                check("big rows", back.rowCount() == 2000);
                System.out.println("    scale 2k write+read: " + ms + " ms");
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
