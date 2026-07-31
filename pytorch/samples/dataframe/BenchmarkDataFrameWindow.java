package dataframe;

import org.bytedeco.pytorch.dataframe.*;
import org.bytedeco.pytorch.dataframe.window.WindowSpec;

import java.util.*;

import static org.bytedeco.pytorch.dataframe.Functions.*;

/**
 * Correctness suite for Spark-style window functions.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... BenchmarkDataFrameWindow
 * </pre>
 */
public class BenchmarkDataFrameWindow {
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

    static long asLong(Object v) {
        return ((Number) v).longValue();
    }

    static double asDouble(Object v) {
        return ((Number) v).doubleValue();
    }

    public static void main(String[] args) {
        System.out.println("=== BenchmarkDataFrameWindow ===");

        DataFrame emp = DataFrame.create();
        emp.addColumn("dept", Column.DType.STRING);
        emp.addColumn("name", Column.DType.STRING);
        emp.addColumn("salary", Column.DType.FLOAT64);
        emp.addColumn("year", Column.DType.INT64);
        emp.addRow("eng", "alice", 100.0, 2020L);
        emp.addRow("eng", "bob",   100.0, 2021L); // tie with alice on salary
        emp.addRow("eng", "carol",  80.0, 2019L);
        emp.addRow("hr",  "dave",   90.0, 2022L);
        emp.addRow("hr",  "erin",   70.0, 2018L);
        emp.addRow("hr",  "frank",  90.0, 2020L); // tie with dave

        benchmark("1. row_number partition+order", () -> {
            WindowSpec w = window().partitionBy("dept").orderBy(desc("salary"), asc("name"));
            DataFrame out = emp.withColumn("rn", row_number().over(w));
            // eng ordered by salary desc, name asc: alice(100), bob(100), carol(80) → rn 1,2,3
            // Find alice/bob/carol
            Map<String, Long> rn = new HashMap<>();
            for (int i = 0; i < out.rowCount(); i++) {
                rn.put((String) out.get(i, "name"), asLong(out.get(i, "rn")));
            }
            check("alice rn", rn.get("alice") == 1L || rn.get("alice") == 2L);
            check("bob rn", rn.get("bob") == 1L || rn.get("bob") == 2L);
            check("alice != bob rn", !rn.get("alice").equals(rn.get("bob")));
            check("carol rn 3", rn.get("carol") == 3L);
            check("erin rn 3 in hr", rn.get("erin") == 3L);
        });

        benchmark("2. rank vs dense_rank", () -> {
            WindowSpec w = window().partitionBy("dept").orderBy(desc("salary"));
            DataFrame out = emp
                .withColumn("rk", rank().over(w))
                .withColumn("dr", dense_rank().over(w));
            // eng: alice/bob salary 100 → rank 1, dense 1; carol 80 → rank 3, dense 2
            Map<String, long[]> m = new HashMap<>();
            for (int i = 0; i < out.rowCount(); i++) {
                m.put((String) out.get(i, "name"),
                    new long[]{asLong(out.get(i, "rk")), asLong(out.get(i, "dr"))});
            }
            check("alice rank 1", m.get("alice")[0] == 1L);
            check("bob rank 1", m.get("bob")[0] == 1L);
            check("carol rank 3", m.get("carol")[0] == 3L);
            check("alice dense 1", m.get("alice")[1] == 1L);
            check("carol dense 2", m.get("carol")[1] == 2L);
            // hr: dave/frank 90 rank 1; erin 70 rank 3 dense 2
            check("dave rank 1", m.get("dave")[0] == 1L);
            check("frank rank 1", m.get("frank")[0] == 1L);
            check("erin rank 3", m.get("erin")[0] == 3L);
            check("erin dense 2", m.get("erin")[1] == 2L);
        });

        benchmark("3. percent_rank ntile cume_dist", () -> {
            WindowSpec w = window().partitionBy("dept").orderBy(asc("salary"));
            DataFrame out = emp
                .withColumn("pct", percent_rank().over(w))
                .withColumn("nt", ntile(2).over(w))
                .withColumn("cd", cume_dist().over(w));
            // eng size 3: lowest salary carol pct=0; highest alice/bob pct=1
            for (int i = 0; i < out.rowCount(); i++) {
                if ("carol".equals(out.get(i, "name"))) {
                    check("carol pct 0", Math.abs(asDouble(out.get(i, "pct"))) < 1e-9);
                }
                if ("erin".equals(out.get(i, "name"))) {
                    check("erin pct 0", Math.abs(asDouble(out.get(i, "pct"))) < 1e-9);
                }
                double cd = asDouble(out.get(i, "cd"));
                check("cume_dist in (0,1]", cd > 0 && cd <= 1.0 + 1e-9);
                long nt = asLong(out.get(i, "nt"));
                check("ntile 1 or 2", nt == 1L || nt == 2L);
            }
        });

        benchmark("4. lag lead", () -> {
            WindowSpec w = window().partitionBy("dept").orderBy(asc("year"));
            DataFrame out = emp
                .withColumn("prev", lag(col("salary"), 1).over(w))
                .withColumn("next", lead(col("salary"), 1, -1.0).over(w));
            // eng by year: carol 2019, alice 2020, bob 2021
            for (int i = 0; i < out.rowCount(); i++) {
                if ("carol".equals(out.get(i, "name"))) {
                    check("carol lag null", out.get(i, "prev") == null);
                    check("carol lead alice sal", Math.abs(asDouble(out.get(i, "next")) - 100.0) < 1e-9);
                }
                if ("bob".equals(out.get(i, "name"))) {
                    check("bob lag alice", Math.abs(asDouble(out.get(i, "prev")) - 100.0) < 1e-9);
                    check("bob lead default -1", Math.abs(asDouble(out.get(i, "next")) + 1.0) < 1e-9);
                }
            }
        });

        benchmark("5. sum/mean over partition (whole frame)", () -> {
            // no order → whole partition frame
            WindowSpec w = window().partitionBy("dept");
            DataFrame out = emp
                .withColumn("dept_sum", col("salary").sum().over(w))
                .withColumn("dept_mean", col("salary").mean().over(w));
            // eng: 100+100+80=280 mean ~93.333
            // hr: 90+70+90=250 mean ~83.333
            for (int i = 0; i < out.rowCount(); i++) {
                String dept = (String) out.get(i, "dept");
                double s = asDouble(out.get(i, "dept_sum"));
                double m = asDouble(out.get(i, "dept_mean"));
                if ("eng".equals(dept)) {
                    check("eng sum 280", Math.abs(s - 280.0) < 1e-6);
                    check("eng mean", Math.abs(m - 280.0 / 3) < 1e-6);
                } else {
                    check("hr sum 250", Math.abs(s - 250.0) < 1e-6);
                    check("hr mean", Math.abs(m - 250.0 / 3) < 1e-6);
                }
            }
        });

        benchmark("6. framed rolling sum rowsBetween(-1,0)", () -> {
            WindowSpec w = window().partitionBy("dept").orderBy(asc("year")).rowsBetween(-1, 0);
            DataFrame out = emp.withColumn("roll2", col("salary").sum().over(w));
            // eng year order: carol 80, alice 100, bob 100
            // carol: 80; alice: 80+100=180; bob: 100+100=200
            Map<String, Double> m = new HashMap<>();
            for (int i = 0; i < out.rowCount(); i++) {
                m.put((String) out.get(i, "name"), asDouble(out.get(i, "roll2")));
            }
            check("carol roll 80", Math.abs(m.get("carol") - 80.0) < 1e-6);
            check("alice roll 180", Math.abs(m.get("alice") - 180.0) < 1e-6);
            check("bob roll 200", Math.abs(m.get("bob") - 200.0) < 1e-6);
        });

        benchmark("7. withWindow sugar + empty partition edge", () -> {
            WindowSpec w = window().orderBy(asc("salary"));
            DataFrame out = emp.withWindow("rn_all", row_number(), w);
            check("rn_all rows", out.rowCount() == 6);
            Set<Long> rns = new HashSet<>();
            for (int i = 0; i < out.rowCount(); i++) rns.add(asLong(out.get(i, "rn_all")));
            check("rn 1..6", rns.size() == 6 && rns.contains(1L) && rns.contains(6L));

            DataFrame empty = DataFrame.create();
            empty.addColumn("dept", Column.DType.STRING);
            empty.addColumn("salary", Column.DType.FLOAT64);
            DataFrame e2 = empty.withColumn("rn", row_number().over(window().partitionBy("dept")));
            check("empty window rows 0", e2.rowCount() == 0);
        });

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
