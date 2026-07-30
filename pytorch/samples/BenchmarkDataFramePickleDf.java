package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.pickle.PickleOptions;
import org.bytedeco.pytorch.data.pickle.Pickle;

import java.io.File;
import java.nio.file.*;
import java.util.*;

/**
 * DataFrame pickle layouts + security benchmark.
 */
public class BenchmarkDataFramePickleDf {
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
        Object[][] rows = {
            {1L, "alice", 9.5},
            {2L, "bob", 7.0},
            {3L, "carol", 8.25},
        };
        for (Object[] row : rows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < row.length; c++) df.set(ri, df.column(c).name(), row[c]);
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFramePickleDf ===");
        Path tmp = Files.createTempDirectory("df-pkl-");
        try {
            DataFrame base = seed();

            benchmark("1. SELF_DESC round-trip", () -> {
                Path p = tmp.resolve("self.pkl");
                base.toPickle(p.toString());
                DataFrame back = DataFrame.readPickle(p.toString());
                check("rows", back.rowCount() == 3);
                check("alice", "alice".equals(String.valueOf(back.get(0, "name"))));
                check("score", ((Number) back.get(0, "score")).doubleValue() == 9.5);
            });

            benchmark("2. RECORDS layout", () -> {
                Path p = tmp.resolve("rec.pkl");
                base.toPickle(p.toString(), PickleOptions.records());
                DataFrame back = DataFrame.readPickle(p.toString());
                check("rows", back.rowCount() == 3);
                check("bob", "bob".equals(String.valueOf(back.get(1, "name"))));
            });

            benchmark("3. COLUMNS layout", () -> {
                Path p = tmp.resolve("cols.pkl");
                base.toPickle(p.toString(), PickleOptions.builder()
                    .layout(PickleOptions.Layout.COLUMNS).build());
                DataFrame back = DataFrame.readPickle(p.toString());
                check("rows", back.rowCount() == 3);
                check("cols", back.columnCount() == 3);
            });

            benchmark("4. raw list-of-dicts still loads", () -> {
                Path p = tmp.resolve("raw.pkl");
                List<Map<String, Object>> rows = new ArrayList<>();
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("x", 1L);
                r.put("y", "hi");
                rows.add(r);
                Pickle.dump(rows, p.toFile());
                DataFrame back = DataFrame.readPickle(p.toString());
                check("rows1", back.rowCount() == 1);
                check("x", ((Number) back.get(0, "x")).longValue() == 1L);
            });

            benchmark("5. reject unsafe GLOBAL/REDUCE", () -> {
                // Craft minimal pickle with GLOBAL opcode for os.system — must fail
                // PROTO 4 + GLOBAL 'os\nsystem\n' + STOP is incomplete but should hit default
                byte[] evil = new byte[]{
                    (byte) 0x80, 0x04, // PROTO 4
                    (byte) 'c', // GLOBAL
                    // "os\nsystem\n"
                    'o','s','\n','s','y','s','t','e','m','\n',
                    (byte) '.' // STOP
                };
                Path p = tmp.resolve("evil.pkl");
                Files.write(p, evil);
                boolean threw = false;
                try {
                    Pickle.load(p.toFile());
                } catch (Exception e) {
                    threw = true;
                }
                check("evil rejected", threw);
            });

            benchmark("6. empty DF", () -> {
                Path p = tmp.resolve("empty.pkl");
                DataFrame empty = DataFrame.create();
                empty.addColumn("a", Column.DType.INT64);
                empty.toPickle(p.toString());
                DataFrame back = DataFrame.readPickle(p.toString());
                check("empty rows", back.rowCount() == 0);
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
            System.exit(1);
        }
    }
}
