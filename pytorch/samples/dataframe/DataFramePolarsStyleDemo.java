package dataframe;
import org.bytedeco.pytorch.jit.*;

import static org.bytedeco.pytorch.dataframe.Functions.*;

import java.nio.file.Files;
import java.nio.file.Path;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Expression;
import org.bytedeco.pytorch.dataframe.LazyDataFrame;

/**
 * Polars-style DataFrame expressions + Arrow IPC smoke demo.
 * Mirrors scala-polars ApplyingSimpleExpressions flow on pure-Java storage.
 *
 * <p>Arrow IPC requires JVM opens on modern JDKs:
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED ... DataFramePolarsStyleDemo
 * </pre>
 */
public class DataFramePolarsStyleDemo {

    static int passed = 0;
    static int failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("=== DataFrame Polars-style Expressions + Arrow IPC ===\n");

        // ── seed data ────────────────────────────────────────────────────
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
        System.out.println("Source:\n" + raw);

        // ── expression eval ──────────────────────────────────────────────
        Column x2 = col("id").plus(lit(1)).evaluate(raw);
        check("col+lit size", x2.size() == 6);
        check("col+lit[0]", ((Number) x2.get(0)).longValue() == 2L);
        check("col+lit[5]", ((Number) x2.get(5)).longValue() == 7L);

        Column mask = col("id").lessThanEqualTo(4).evaluate(raw);
        check("lte dtype", mask.dtype() == Column.DType.BOOLEAN);
        check("lte[0]", Boolean.TRUE.equals(mask.get(0)));
        check("lte[5]", Boolean.FALSE.equals(mask.get(5)));

        // ── eager filter / withColumn ────────────────────────────────────
        DataFrame filtered = raw.filter(col("id").lessThanEqualTo(4));
        check("eager filter rows", filtered.rowCount() == 4);

        DataFrame with = raw.withColumn("id2", col("id").multiply(lit(10)));
        check("withColumn has id2", with.hasColumn("id2"));
        check("withColumn value", ((Number) with.get(0, "id2")).longValue() == 10L);

        // ── when / then / otherwise ──────────────────────────────────────
        Expression label = when(col("score").ge(9.0), "A")
            .when(col("score").ge(7.0), "B")
            .otherwise("C");
        DataFrame graded = raw.withColumn("grade", label);
        check("when A", "A".equals(graded.get(0, "grade")));
        check("when B", "B".equals(graded.get(1, "grade")));
        check("when C", "C".equals(graded.get(5, "grade")));

        // ── string namespace ─────────────────────────────────────────────
        DataFrame upper = raw.withColumn("NAME", col("name").str().toUpperCase());
        check("str upper", "ALICE".equals(upper.get(0, "NAME")));

        // ── lazy chain (ApplyingSimpleExpressions style) ─────────────────
        LazyDataFrame ldf = raw.lazy()
            .cache()
            .select("id", "name", "score")
            .withColumn("lower_than_four", col("id").lessThanEqualTo(4))
            .filter(col("lower_than_four"))
            .withColumn("long_value", lit(42L))
            .withColumn("id_plus", col("id").plus(lit(100)))
            .sort(asc("name"))
            .setSorted("name", false, false)
            .topK(3, "id", true) // reverse=true → ascending, then head 3
            .limit(3)
            .drop("long_value")
            .rename("lower_than_four", "less_than_four")
            .dropNulls();

        System.out.println(ldf.explain());
        DataFrame out = ldf.collect();
        System.out.println("Lazy result:\n" + out);
        check("lazy rowCount > 0", out.rowCount() > 0);
        check("lazy has less_than_four", out.hasColumn("less_than_four"));
        check("lazy dropped long_value", !out.hasColumn("long_value"));
        check("lazy has id_plus", out.hasColumn("id_plus"));

        // unique + concat
        LazyDataFrame cat = out.lazy().concat(out.lazy()).unique();
        DataFrame uniq = cat.collect();
        check("unique after concat", uniq.rowCount() == out.rowCount());

        // ── math / window ────────────────────────────────────────────────
        DataFrame math = raw.withColumn("sqrt_score", col("score").sqrt())
            .withColumn("clip", col("score").clip(lit(6.0), lit(9.0)))
            .withColumn("shift_id", col("id").shift(1));
        check("sqrt finite", Double.isFinite(((Number) math.get(0, "sqrt_score")).doubleValue()));
        check("clip high", ((Number) math.get(0, "clip")).doubleValue() == 9.0);
        check("shift null first", math.get(0, "shift_id") == null);
        check("shift second", ((Number) math.get(1, "shift_id")).longValue() == 1L);

        // ── Arrow IPC round-trip ─────────────────────────────────────────
        Path tmp = Files.createTempDirectory("df_arrow");
        Path arrowPath = tmp.resolve("demo.arrow");
        try {
            out.writeArrow(arrowPath.toString());
            check("arrow file exists", Files.exists(arrowPath) && Files.size(arrowPath) > 0);

            DataFrame back = DataFrame.readArrow(arrowPath.toString());
            System.out.println("Arrow round-trip:\n" + back);
            check("arrow rows", back.rowCount() == out.rowCount());
            check("arrow cols", back.columnCount() == out.columnCount());
            for (int c = 0; c < out.columnCount(); c++) {
                check("arrow col name " + c, out.column(c).name().equals(back.column(c).name()));
            }
            // compare a few values
            if (out.rowCount() > 0 && out.hasColumn("id") && back.hasColumn("id")) {
                Object a = out.get(0, "id");
                Object b = back.get(0, "id");
                check("arrow id[0]", a != null && b != null
                    && ((Number) a).longValue() == ((Number) b).longValue());
            }
        } finally {
            try { Files.deleteIfExists(arrowPath); } catch (Exception ignored) {}
            try { Files.deleteIfExists(tmp); } catch (Exception ignored) {}
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            System.out.println("  FAIL  " + name);
        }
    }
}
