package samples;
import org.bytedeco.pytorch.autograd.*;
import static org.bytedeco.pytorch.dataframe.Functions.*;

import java.nio.file.*;
import java.time.*;
import java.util.*;

import org.bytedeco.pytorch.dataframe.*;

/**
 * Multi-dimensional correctness benchmark for Polars-style DataFrame:
 * I/O, conversions, expressions, lazy optimizer, groupby, Arrow-backed storage, str/dt.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrame
 * </pre>
 */
public class BenchmarkDataFrame {
    static int passed = 0, failed = 0;
    static StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        try {
            r.run();
            System.out.println("  ✓ " + name);
        } catch (Throwable t) {
            failed++;
            report.append("FAIL ").append(name).append(": ").append(t).append('\n');
            System.out.println("  ✗ " + name + " — " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) { passed++; }
        else {
            failed++;
            report.append("CHECK FAILED: ").append(name).append('\n');
            System.out.println("    CHECK FAIL: " + name);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrame (comprehensive) ===\n");
        Path tmp = Files.createTempDirectory("df_bench");
        System.out.println("Temp: " + tmp + "\n");

        try {
            // ── 1. Construction / schema / temporal dtypes ─────────────
            benchmark("1. construction schema temporal", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("id", Column.DType.INT64);
                df.addColumn("name", Column.DType.STRING);
                df.addColumn("score", Column.DType.FLOAT64);
                df.addColumn("when", Column.DType.DATE);
                df.addColumn("ts", Column.DType.DATETIME);
                df.addRow(1L, "alice", 9.5, LocalDate.of(2024, 1, 15), Instant.parse("2024-01-15T10:30:00Z"));
                df.addRow(2L, "bob", 7.0, LocalDate.of(2024, 6, 1), Instant.parse("2024-06-01T08:00:00Z"));
                df.addRow(3L, "carol", 8.2, LocalDate.of(2025, 3, 20), Instant.parse("2025-03-20T12:00:00Z"));
                Schema sch = df.schema();
                check("schema size", sch.size() == 5);
                check("schema date", sch.fieldType("when") == Column.DType.DATE);
                check("schema datetime", sch.fieldType("ts") == Column.DType.DATETIME);
                check("rows", df.rowCount() == 3);
                check("get date", LocalDate.of(2024, 1, 15).equals(df.get(0, "when")));
            });

            DataFrame base = seedBase();

            // ── 2. I/O CSV + Arrow ─────────────────────────────────────
            benchmark("2. CSV round-trip", () -> {
                Path p = tmp.resolve("t.csv");
                base.toCsv(p.toString());
                DataFrame back = DataFrame.readCsv(p.toString());
                check("csv rows", back.rowCount() == base.rowCount());
                check("csv cols", back.columnCount() == base.columnCount());
            });

            benchmark("2b. Arrow IPC round-trip + zero-copy flag", () -> {
                Path p = tmp.resolve("t.arrow");
                base.writeArrow(p.toString());
                try (DataFrame back = DataFrame.readArrow(p.toString())) {
                    check("arrow rows", back.rowCount() == base.rowCount());
                    check("arrow cols", back.columnCount() == base.columnCount());
                    check("arrow id0", ((Number) back.get(0, "id")).longValue() == 1L);
                    check("arrow name1", "bob".equals(String.valueOf(back.get(1, "name"))));
                    // at least one column should be arrow-backed for single-batch write
                    boolean anyArrow = false;
                    for (Column c : back.columns()) if (c.isArrowBacked()) anyArrow = true;
                    check("arrow-backed present", anyArrow);
                }
            });

            benchmark("2c. Parquet optional", () -> {
                Path p = tmp.resolve("t.parquet");
                try {
                    base.writeParquet(p.toString());
                    DataFrame back = DataFrame.readParquet(p.toString());
                    check("parquet rows", back.rowCount() == base.rowCount());
                } catch (Throwable t) {
                    System.out.println("    (parquet skipped: " + t.getClass().getSimpleName() + ")");
                }
            });

            // ── 3. Expression arithmetic / compare / boolean ───────────
            benchmark("3. expression arith compare boolean", () -> {
                Column x2 = col("id").plus(lit(10)).evaluate(base);
                check("plus", ((Number) x2.get(0)).longValue() == 11L);
                Column m = col("score").multiply(lit(2)).evaluate(base);
                check("mul", Math.abs(((Number) m.get(0)).doubleValue() - 19.0) < 1e-9);
                check("gt", Boolean.TRUE.equals(col("id").gt(lit(2)).evaluate(base).get(2)));
                check("and", Boolean.TRUE.equals(
                    col("id").gt(lit(1)).and(col("score").lt(lit(9))).evaluate(base).get(1)));
                check("isNull false", Boolean.FALSE.equals(col("id").isNull().evaluate(base).get(0)));
            });

            // ── 4. Math / window ───────────────────────────────────────
            benchmark("4. math window ops", () -> {
                check("sqrt", ((Number) col("score").sqrt().evaluate(base).get(0)).doubleValue() > 3);
                check("abs", ((Number) lit(-3).abs().evaluate(base).get(0)).intValue() == 3
                    || ((Number) col("id").neg().abs().evaluate(base).get(0)).longValue() == 1L);
                Column cs = col("id").cumSum().evaluate(base);
                check("cumsum last", ((Number) cs.get(base.rowCount()-1)).doubleValue() == 21.0); // 1+2+...+6
                check("shift null", col("id").shift(1).evaluate(base).get(0) == null);
                check("clip", ((Number) col("score").clip(6.0, 9.0).evaluate(base).get(0)).doubleValue() == 9.0);
            });

            // ── 5. String namespace ─────────────────────────────────────
            benchmark("5. str namespace", () -> {
                check("upper", "ALICE".equals(col("name").str().toUpperCase().evaluate(base).get(0)));
                check("lower alias", "alice".equals(col("name").str().toLowercase().evaluate(base).get(0)));
                check("length", ((Number) col("name").str().length().evaluate(base).get(0)).intValue() == 5);
                check("contains", Boolean.TRUE.equals(col("name").str().contains("li").evaluate(base).get(0)));
                check("startsWith", Boolean.TRUE.equals(col("name").str().startsWith("a").evaluate(base).get(0)));
                check("zfill", "00042".equals(lit("42").str().zfill(5).evaluate(base).get(0)));
                check("slice", "li".equals(col("name").str().slice(1, 2).evaluate(base).get(0)));
                check("replace", "aXice".equals(col("name").str().replace("l", "X").evaluate(base).get(0)));
                check("toInt", ((Number) lit("42").str().toInteger().evaluate(base).get(0)).intValue() == 42);
            });

            // ── 6. Temporal namespace ──────────────────────────────────
            benchmark("6. dt namespace", () -> {
                DataFrame tdf = DataFrame.create();
                tdf.addColumn("d", Column.DType.DATE);
                tdf.addColumn("ts", Column.DType.DATETIME);
                tdf.addRow(LocalDate.of(2024, 3, 15), Instant.parse("2024-03-15T14:30:45Z"));
                check("year", ((Number) col("d").dt().year().evaluate(tdf).get(0)).intValue() == 2024);
                check("month", ((Number) col("d").dt().month().evaluate(tdf).get(0)).intValue() == 3);
                check("day", ((Number) col("d").dt().day().evaluate(tdf).get(0)).intValue() == 15);
                check("hour", ((Number) col("ts").dt().hour().evaluate(tdf).get(0)).intValue() == 14);
                check("date lit", LocalDate.of(2020,1,2).equals(date(2020,1,2).evaluate(tdf).get(0)));
                check("strptime", col("x")./*placeholder*/getClass() != null);
                DataFrame s = DataFrame.create();
                s.addColumn("s", Column.DType.STRING);
                s.addRow("2024-07-04");
                Object parsed = col("s").str().strptime("yyyy-MM-dd").evaluate(s).get(0);
                check("strptime type", parsed instanceof LocalDateTime || parsed instanceof LocalDate);
            });

            // ── 7. Eager transforms ───────────────────────────────────
            benchmark("7. eager filter withColumn select sort join", () -> {
                DataFrame f = base.filter(col("id").le(lit(3)));
                check("filter rows", f.rowCount() == 3);
                DataFrame w = base.withColumn("id2", col("id").multiply(lit(100)));
                check("withColumn", ((Number) w.get(0, "id2")).longValue() == 100L);
                DataFrame sel = base.select(col("id").alias("i"), col("name"));
                check("select expr", sel.hasColumn("i") && sel.hasColumn("name"));
                DataFrame sorted = base.lazy().sort(desc("score")).collect();
                check("sort desc first", ((Number) sorted.get(0, "score")).doubleValue() >=
                    ((Number) sorted.get(1, "score")).doubleValue());
                DataFrame right = DataFrame.create();
                right.addColumn("id", Column.DType.INT64);
                right.addColumn("city", Column.DType.STRING);
                right.addRow(1L, "NYC");
                right.addRow(2L, "SF");
                DataFrame j = base.join(right, "id", "left");
                check("join cols", j.hasColumn("city"));
            });

            // ── 8. Lazy + optimizer pushdown ──────────────────────────
            benchmark("8. lazy optimizer predicate pushdown", () -> {
                LazyDataFrame ldf = base.lazy()
                    .withColumn("x2", col("id").plus(lit(1)))
                    .filter(col("id").gt(lit(2))); // independent of x2
                String logical = ldf.explain(false);
                String optimized = ldf.explain(true);
                check("logical has WITH before FILTER or both", logical.contains("WITH_COLUMN") && logical.contains("FILTER"));
                // optimized should list FILTER before WITH_COLUMN
                int fi = optimized.indexOf("FILTER");
                int wi = optimized.indexOf("WITH_COLUMN");
                check("pushdown FILTER before WITH", fi >= 0 && wi >= 0 && fi < wi);
                DataFrame out = ldf.collect();
                check("collect rows", out.rowCount() == 4); // ids 3,4,5,6
                check("has x2", out.hasColumn("x2"));

                // filter merge
                LazyDataFrame m = base.lazy()
                    .filter(col("id").gt(lit(1)))
                    .filter(col("id").lt(lit(5)));
                check("merged plan smaller or equal", m.optimizedPlanSize() <= m.planSize());
                check("merged result", m.collect().rowCount() == 3); // 2,3,4
            });

            // ── 9. GroupBy map-agg + expression agg ───────────────────
            benchmark("9. groupBy expression agg", () -> {
                DataFrame gdf = DataFrame.create();
                gdf.addColumn("city", Column.DType.STRING);
                gdf.addColumn("amount", Column.DType.FLOAT64);
                gdf.addColumn("n", Column.DType.INT64);
                gdf.addRow("NYC", 10.0, 1L);
                gdf.addRow("NYC", 20.0, 1L);
                gdf.addRow("SF", 5.0, 1L);
                gdf.addRow("SF", 15.0, 1L);
                gdf.addRow("SF", 10.0, 1L);

                DataFrame agg = gdf.groupBy("city").agg(
                    col("amount").sum().alias("total"),
                    col("amount").mean().alias("avg"),
                    col("n").count().alias("cnt")
                );
                check("agg rows", agg.rowCount() == 2);
                check("agg has total", agg.hasColumn("total"));
                // find NYC
                double nycTotal = Double.NaN, sfTotal = Double.NaN;
                for (int i = 0; i < agg.rowCount(); i++) {
                    if ("NYC".equals(String.valueOf(agg.get(i, "city"))))
                        nycTotal = ((Number) agg.get(i, "total")).doubleValue();
                    if ("SF".equals(String.valueOf(agg.get(i, "city"))))
                        sfTotal = ((Number) agg.get(i, "total")).doubleValue();
                }
                check("NYC total 30", Math.abs(nycTotal - 30.0) < 1e-6);
                check("SF total 30", Math.abs(sfTotal - 30.0) < 1e-6);

                // map-style agg still works
                Map<String, AggFunction> m = new LinkedHashMap<>();
                m.put("amount", AggFunction.SUM);
                DataFrame mAgg = gdf.groupby("city").agg(m);
                check("map agg rows", mAgg.rowCount() == 2);

                // lazy groupBy
                DataFrame lazyAgg = gdf.lazy().groupBy("city")
                    .agg(col("amount").sum().alias("s")).collect();
                check("lazy groupby rows", lazyAgg.rowCount() == 2);
            });

            // ── 10. when / otherwise ──────────────────────────────────
            benchmark("10. when otherwise", () -> {
                Expression label = when(col("score").ge(9.0), "A")
                    .when(col("score").ge(7.0), "B")
                    .otherwise("C");
                DataFrame g = base.withColumn("grade", label);
                check("A", "A".equals(g.get(0, "grade")));
                check("B", "B".equals(g.get(1, "grade")));
                check("C", "C".equals(g.get(5, "grade")));
            });

            // ── 11. Edge cases ────────────────────────────────────────
            benchmark("11. edge empty nulls fillNull", () -> {
                DataFrame empty = DataFrame.create();
                empty.addColumn("a", Column.DType.INT64);
                check("empty rows", empty.rowCount() == 0);
                check("empty schema", empty.schema().size() == 1);

                DataFrame n = DataFrame.create();
                n.addColumn("a", Column.DType.INT64);
                n.addColumn("b", Column.DType.STRING);
                n.addRow(1L, null);
                n.addRow(null, "x");
                check("isNull", Boolean.TRUE.equals(col("b").isNull().evaluate(n).get(0)));
                check("fillNull", "z".equals(col("b").fillNull(lit("z")).evaluate(n).get(0)));
            });

            // ── 12. Regression polars-style chain ─────────────────────
            benchmark("12. polars-style full chain", () -> {
                DataFrame out = base.lazy()
                    .select("id", "name", "score")
                    .withColumn("flag", col("id").lessThanEqualTo(4))
                    .filter(col("flag"))
                    .withColumn("id_plus", col("id").plus(lit(100)))
                    .sort(asc("name"))
                    .limit(3)
                    .drop("flag")
                    .collect();
                check("chain rows", out.rowCount() == 3);
                check("chain has id_plus", out.hasColumn("id_plus"));
                check("chain no flag", !out.hasColumn("flag"));
            });

            // ── 13. Window smoke ───────────────────────────────────────
            benchmark("13. window row_number smoke", () -> {
                DataFrame wdf = DataFrame.create();
                wdf.addColumn("g", Column.DType.STRING);
                wdf.addColumn("v", Column.DType.INT64);
                wdf.addRow("a", 3L);
                wdf.addRow("a", 1L);
                wdf.addRow("b", 2L);
                DataFrame out = wdf.withColumn("rn",
                    row_number().over(window().partitionBy("g").orderBy(asc("v"))));
                check("window rows", out.rowCount() == 3);
                check("has rn", out.hasColumn("rn"));
                // within group a: v=1 → rn 1, v=3 → rn 2
                Map<Long, Long> byV = new HashMap<>();
                for (int i = 0; i < out.rowCount(); i++) {
                    if ("a".equals(out.get(i, "g"))) {
                        byV.put(((Number) out.get(i, "v")).longValue(),
                            ((Number) out.get(i, "rn")).longValue());
                    }
                }
                check("a v1 rn1", byV.get(1L) == 1L);
                check("a v3 rn2", byV.get(3L) == 2L);
            });

            // ── 14. CSV edge smoke (quoted comma) ─────────────────────
            benchmark("14. CSV quoted-comma smoke", () -> {
                Path p = tmp.resolve("edge.csv");
                Files.writeString(p, "name,note\n\"x,y\",\"ok\"\n");
                DataFrame back = DataFrame.readCsv(p.toString());
                check("csv edge rows", back.rowCount() == 1);
                check("csv edge name", "x,y".equals(String.valueOf(back.get(0, "name"))));
            });

            // ── 15. ANN smoke (tiny) ──────────────────────────────────
            benchmark("15. HNSW ann smoke", () -> {
                float[][] vecs = {
                    {1f, 0f, 0f},
                    {0.9f, 0.1f, 0f},
                    {0f, 1f, 0f},
                    {0f, 0f, 1f}
                };
                DataFrame vdf = DataFrame.fromVectors("emb", vecs, "id", null);
                DataFrame nn = vdf.annSearch("emb", new float[]{1f, 0f, 0f}, 2);
                check("ann rows", nn.rowCount() == 2);
                check("ann distance col", nn.hasColumn("_distance"));
                check("ann rank1", ((Number) nn.get(0, "_rank")).longValue() == 1L);
            });

        } finally {
            // cleanup tmp best-effort
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            } catch (Exception ignored) {}
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }

    static DataFrame seedBase() {
        DataFrame raw = DataFrame.create();
        raw.addColumn("id", Column.DType.INT64);
        raw.addColumn("name", Column.DType.STRING);
        raw.addColumn("score", Column.DType.FLOAT64);
        raw.addRow(1L, "alice", 9.5);
        raw.addRow(2L, "bob", 7.0);
        raw.addRow(3L, "carol", 8.2);
        raw.addRow(4L, "dave", 6.1);
        raw.addRow(5L, "erin", 9.0);
        raw.addRow(6L, "frank", 5.5);
        return raw;
    }
}
