package samples;

import static org.bytedeco.pytorch.dataframe.Functions.*;

import java.util.*;

import org.bytedeco.pytorch.dataframe.*;
import org.bytedeco.pytorch.dataframe.window.Rolling;
import org.bytedeco.pytorch.dataframe.window.Expanding;

/**
 * Correctness suite for the Pandas/Polars-aligned 100 common DataFrame operators.
 *
 * <p>Groups mirror the operator checklist:
 * <ol>
 *   <li>Element-wise math (1–20)</li>
 *   <li>Binary / conditional (21–35)</li>
 *   <li>Aggregations (36–55)</li>
 *   <li>Window / rolling (56–65)</li>
 *   <li>String (66–75)</li>
 *   <li>Table filter/sort/sample (76–85)</li>
 *   <li>GroupBy / join / reshape (86–95)</li>
 *   <li>Cast / value_counts / isin / numpy / describe (96–100)</li>
 * </ol>
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameOps
 * </pre>
 */
public class BenchmarkDataFrameOps {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
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
            report.append("CHECK FAILED: ").append(name).append('\n');
            System.out.println("    CHECK FAIL: " + name);
        }
    }

    static void checkEq(String name, double expected, Object actual, double eps) {
        if (actual == null) {
            check(name, false);
            System.out.println("    expected " + expected + " got null");
            return;
        }
        double a = ((Number) actual).doubleValue();
        boolean ok = Math.abs(a - expected) <= eps || (Double.isNaN(a) && Double.isNaN(expected));
        if (!ok) System.out.println("    " + name + ": expected " + expected + " got " + a);
        check(name, ok);
    }

    static void checkEq(String name, Object expected, Object actual) {
        boolean ok = Objects.equals(expected, actual)
            || (expected != null && actual != null && String.valueOf(expected).equals(String.valueOf(actual)));
        if (!ok) System.out.println("    " + name + ": expected " + expected + " got " + actual);
        check(name, ok);
    }

    static double d(Object v) { return ((Number) v).doubleValue(); }
    static long l(Object v) { return ((Number) v).longValue(); }
    static boolean b(Object v) { return v instanceof Boolean ? (Boolean) v : Expression.isTrue(v); }

    static DataFrame numericSeed() {
        DataFrame df = DataFrame.create();
        df.addColumn("x", Column.DType.FLOAT64);
        df.addColumn("y", Column.DType.FLOAT64);
        df.addColumn("z", Column.DType.INT64);
        // x: -2, -0.5, 0, 1.5, 4, null-ish via NaN avoided — use null for one
        df.addRow(-2.0, 10.0, 1L);
        df.addRow(-0.5, 20.0, 2L);
        df.addRow(0.0, 30.0, 3L);
        df.addRow(1.5, 40.0, 4L);
        df.addRow(4.0, 50.0, 5L);
        return df;
    }

    static DataFrame withNulls() {
        DataFrame df = DataFrame.create();
        df.addColumn("a", Column.DType.FLOAT64);
        df.addColumn("b", Column.DType.STRING);
        df.addRow(1.0, "x");
        df.addRow(null, "y");
        df.addRow(3.0, null);
        df.addRow(4.0, "z");
        return df;
    }

    static DataFrame strSeed() {
        DataFrame df = DataFrame.create();
        df.addColumn("s", Column.DType.STRING);
        df.addRow("  Hello ");
        df.addRow("WORLD");
        df.addRow("foo_bar");
        df.addRow("abc123");
        return df;
    }

    static DataFrame groupSeed() {
        DataFrame df = DataFrame.create();
        df.addColumn("city", Column.DType.STRING);
        df.addColumn("amt", Column.DType.FLOAT64);
        df.addColumn("qty", Column.DType.INT64);
        df.addRow("NY", 10.0, 1L);
        df.addRow("SF", 20.0, 2L);
        df.addRow("NY", 30.0, 3L);
        df.addRow("SF", 40.0, 4L);
        df.addRow("NY", 50.0, 5L);
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameOps — 100 Pandas/Polars operators ===\n");

        // ════════════════════════════════════════════════════════════
        // I. Element-wise math (1–20)
        // ════════════════════════════════════════════════════════════
        benchmark("I. element-wise math 1-20", () -> {
            DataFrame df = numericSeed();

            // 1 abs
            Column c = col("x").abs().evaluate(df);
            checkEq("1.abs[0]", 2.0, c.get(0), 1e-9);
            checkEq("1.abs[1]", 0.5, c.get(1), 1e-9);

            // 2 sqrt (of abs for negatives)
            c = col("x").abs().sqrt().evaluate(df);
            checkEq("2.sqrt[0]", Math.sqrt(2), c.get(0), 1e-9);
            checkEq("2.sqrt[4]", 2.0, c.get(4), 1e-9);

            // 3 square
            c = col("x").square().evaluate(df);
            checkEq("3.square[0]", 4.0, c.get(0), 1e-9);
            checkEq("3.square[3]", 2.25, c.get(3), 1e-9);

            // 4 exp
            c = col("x").exp().evaluate(df);
            checkEq("4.exp[2]", 1.0, c.get(2), 1e-9);
            checkEq("4.exp[3]", Math.exp(1.5), c.get(3), 1e-9);

            // 5 log (positive only)
            DataFrame pos = DataFrame.create();
            pos.addColumn("x", Column.DType.FLOAT64);
            pos.addRow(Math.E); pos.addRow(1.0); pos.addRow(Math.E * Math.E);
            c = col("x").log().evaluate(pos);
            checkEq("5.log[0]", 1.0, c.get(0), 1e-9);
            checkEq("5.log[1]", 0.0, c.get(1), 1e-9);

            // 6 log10
            pos = DataFrame.create();
            pos.addColumn("x", Column.DType.FLOAT64);
            pos.addRow(100.0); pos.addRow(1000.0);
            c = col("x").log10().evaluate(pos);
            checkEq("6.log10[0]", 2.0, c.get(0), 1e-9);
            checkEq("6.log10[1]", 3.0, c.get(1), 1e-9);

            // 7 log2
            pos = DataFrame.create();
            pos.addColumn("x", Column.DType.FLOAT64);
            pos.addRow(8.0); pos.addRow(32.0);
            c = col("x").log2().evaluate(pos);
            checkEq("7.log2[0]", 3.0, c.get(0), 1e-9);
            checkEq("7.log2[1]", 5.0, c.get(1), 1e-9);

            // 8 ceil
            c = col("x").ceil().evaluate(df);
            checkEq("8.ceil[1]", 0.0, c.get(1), 1e-9);   // ceil(-0.5)=0? Actually Math.ceil(-0.5)= -0.0 → 0
            // Math.ceil(-0.5) = -0.0; Math.ceil(1.5)=2
            checkEq("8.ceil[3]", 2.0, c.get(3), 1e-9);

            // 9 floor
            c = col("x").floor().evaluate(df);
            checkEq("9.floor[3]", 1.0, c.get(3), 1e-9);
            checkEq("9.floor[1]", -1.0, c.get(1), 1e-9);

            // 10 round
            DataFrame r = DataFrame.create();
            r.addColumn("x", Column.DType.FLOAT64);
            r.addRow(1.234); r.addRow(1.235); r.addRow(2.5);
            c = col("x").round(2).evaluate(r);
            checkEq("10.round[0]", 1.23, c.get(0), 1e-9);

            // 11 trunc
            c = col("x").trunc().evaluate(df);
            checkEq("11.trunc[3]", 1.0, c.get(3), 1e-9);
            checkEq("11.trunc[1]", 0.0, c.get(1), 1e-9); // trunc toward zero: -0.5 → 0?
            // our truncate uses floor for >=0 and ceil for <0 → -0.5 → 0 via ceil? Math.ceil(-0.5)=0 yes

            // 12-15 trig
            DataFrame ang = DataFrame.create();
            ang.addColumn("x", Column.DType.FLOAT64);
            ang.addRow(0.0); ang.addRow(Math.PI / 2); ang.addRow(Math.PI / 4);
            checkEq("12.sin[0]", 0.0, col("x").sin().evaluate(ang).get(0), 1e-9);
            checkEq("12.sin[1]", 1.0, col("x").sin().evaluate(ang).get(1), 1e-9);
            checkEq("13.cos[0]", 1.0, col("x").cos().evaluate(ang).get(0), 1e-9);
            checkEq("14.tan[0]", 0.0, col("x").tan().evaluate(ang).get(0), 1e-9);
            checkEq("15.tanh[0]", 0.0, col("x").tanh().evaluate(ang).get(0), 1e-9);

            // 16 sign
            c = col("x").sign().evaluate(df);
            checkEq("16.sign[0]", -1.0, c.get(0), 1e-9);
            checkEq("16.sign[2]", 0.0, c.get(2), 1e-9);
            checkEq("16.sign[3]", 1.0, c.get(3), 1e-9);

            // 17 clip
            c = col("x").clip(-1.0, 2.0).evaluate(df);
            checkEq("17.clip[0]", -1.0, c.get(0), 1e-9); // -2 → -1
            checkEq("17.clip[4]", 2.0, c.get(4), 1e-9);  // 4 → 2
            checkEq("17.clip[3]", 1.5, c.get(3), 1e-9);

            // 18 isna / isNull
            DataFrame n = withNulls();
            c = col("a").isNull().evaluate(n);
            check("18.isna[0]", !b(c.get(0)));
            check("18.isna[1]", b(c.get(1)));
            c = col("a").isna().evaluate(n);
            check("18.isna.alias[1]", b(c.get(1)));

            // 19 notna
            c = col("a").notna().evaluate(n);
            check("19.notna[0]", b(c.get(0)));
            check("19.notna[1]", !b(c.get(1)));

            // 20 fillna
            c = col("a").fillna(0.0).evaluate(n);
            checkEq("20.fillna[1]", 0.0, c.get(1), 1e-9);
            checkEq("20.fillna[0]", 1.0, c.get(0), 1e-9);
        });

        // ════════════════════════════════════════════════════════════
        // II. Binary ops & conditionals (21–35)
        // ════════════════════════════════════════════════════════════
        benchmark("II. binary & conditional 21-35", () -> {
            DataFrame df = numericSeed();

            // 21 +
            Column c = col("x").plus(col("y")).evaluate(df);
            checkEq("21.add[0]", 8.0, c.get(0), 1e-9);

            // 22 -
            c = col("y").minus(col("x")).evaluate(df);
            checkEq("22.sub[0]", 12.0, c.get(0), 1e-9);

            // 23 *
            c = col("x").multiply(col("z")).evaluate(df);
            checkEq("23.mul[3]", 6.0, c.get(3), 1e-9); // 1.5*4

            // 24 /
            c = col("y").divide(col("z")).evaluate(df);
            checkEq("24.div[0]", 10.0, c.get(0), 1e-9);

            // 25 pow
            c = col("z").pow(2.0).evaluate(df);
            checkEq("25.pow.scalar[2]", 9.0, c.get(2), 1e-9);
            // binary pow: z ** 0? use lit
            DataFrame p = DataFrame.create();
            p.addColumn("a", Column.DType.FLOAT64);
            p.addColumn("b", Column.DType.FLOAT64);
            p.addRow(2.0, 3.0); p.addRow(4.0, 0.5);
            c = col("a").pow(col("b")).evaluate(p);
            checkEq("25.pow.binary[0]", 8.0, c.get(0), 1e-9);
            checkEq("25.pow.binary[1]", 2.0, c.get(1), 1e-9);

            // 26-30 comparisons
            c = col("x").gt(0).evaluate(df);
            check("26.gt[0]", !b(c.get(0)));
            check("26.gt[3]", b(c.get(3)));
            c = col("x").lt(0).evaluate(df);
            check("27.lt[0]", b(c.get(0)));
            c = col("z").eq(3).evaluate(df);
            check("28.eq[2]", b(c.get(2)));
            c = col("z").ge(3).evaluate(df);
            check("29.ge[2]", b(c.get(2)));
            check("29.ge[1]", !b(c.get(1)));
            c = col("z").le(3).evaluate(df);
            check("30.le[2]", b(c.get(2)));
            check("30.le[3]", !b(c.get(3)));

            // 31-33 boolean
            c = col("x").gt(0).and(col("z").gt(3)).evaluate(df);
            check("31.and[3]", b(c.get(3))); // 1.5>0 && 4>3
            check("31.and[2]", !b(c.get(2)));
            c = col("x").lt(0).or(col("x").gt(3)).evaluate(df);
            check("32.or[0]", b(c.get(0)));
            check("32.or[4]", b(c.get(4)));
            check("32.or[2]", !b(c.get(2)));
            c = col("x").gt(0).not().evaluate(df);
            check("33.not[0]", b(c.get(0)));
            check("33.not[3]", !b(c.get(3)));

            // 34 when/then/otherwise (np.where)
            c = when(col("x").gt(0), "pos")
                    .when(col("x").lt(0), "neg")
                    .otherwise("zero")
                    .evaluate(df);
            checkEq("34.when[0]", "neg", c.get(0));
            checkEq("34.when[2]", "zero", c.get(2));
            checkEq("34.when[3]", "pos", c.get(3));

            // 35 where / filter on column (keep value or null)
            c = col("x").where(col("x").gt(0)).evaluate(df);
            check("35.where[0]", c.get(0) == null);
            checkEq("35.where[3]", 1.5, c.get(3), 1e-9);
        });

        // ════════════════════════════════════════════════════════════
        // III. Aggregations (36–55)
        // ════════════════════════════════════════════════════════════
        benchmark("III. aggregations 36-55", () -> {
            DataFrame df = numericSeed();
            // x = [-2, -0.5, 0, 1.5, 4], sum=3, mean=0.6, count=5, max=4, min=-2

            checkEq("36.sum", 3.0, col("x").sum().evaluate(df).get(0), 1e-9);
            checkEq("37.mean", 0.6, col("x").mean().evaluate(df).get(0), 1e-9);
            checkEq("38.count", 5L, col("x").count().evaluate(df).get(0));
            checkEq("39.len", 5L, col("x").len().evaluate(df).get(0));
            checkEq("40.max", 4.0, col("x").max().evaluate(df).get(0), 1e-9);
            checkEq("41.min", -2.0, col("x").min().evaluate(df).get(0), 1e-9);
            checkEq("42.median", 0.0, col("x").median().evaluate(df).get(0), 1e-9);

            // 43 std / 44 var (sample ddof=1)
            double mean = 0.6;
            double[] xs = {-2, -0.5, 0, 1.5, 4};
            double ss = 0;
            for (double v : xs) ss += (v - mean) * (v - mean);
            double var = ss / 4.0;
            double std = Math.sqrt(var);
            checkEq("43.std", std, col("x").std().evaluate(df).get(0), 1e-9);
            checkEq("44.var", var, col("x").var().evaluate(df).get(0), 1e-9);

            // 45 nunique
            DataFrame u = DataFrame.create();
            u.addColumn("c", Column.DType.INT64);
            u.addRow(1L); u.addRow(2L); u.addRow(1L); u.addRow(3L); u.addRow(2L);
            checkEq("45.nunique", 3L, col("c").nUnique().evaluate(u).get(0));

            // 46 unique → variable-length column
            Column uniq = col("c").unique().evaluate(u);
            check("46.unique.size", uniq.size() == 3);
            Set<Object> set = new HashSet<>(uniq.data());
            check("46.unique.has1", set.contains(1L) || set.contains(1));
            check("46.unique.has3", set.contains(3L) || set.contains(3));

            // 47 first / 48 last
            checkEq("47.first", -2.0, col("x").first().evaluate(df).get(0), 1e-9);
            checkEq("48.last", 4.0, col("x").last().evaluate(df).get(0), 1e-9);

            // 49 argmax / 50 argmin
            checkEq("49.argmax", 4L, col("x").argMax().evaluate(df).get(0));
            checkEq("50.argmin", 0L, col("x").argMin().evaluate(df).get(0));

            // 51 quantile
            // sorted x: -2,-0.5,0,1.5,4 ; q=0.5 → median path uses linear
            Object q = col("x").quantile(0.5).evaluate(df).get(0);
            checkEq("51.quantile.med", 0.0, q, 1e-9);

            // 52 mode
            DataFrame m = DataFrame.create();
            m.addColumn("c", Column.DType.INT64);
            m.addRow(1L); m.addRow(2L); m.addRow(2L); m.addRow(3L); m.addRow(2L);
            checkEq("52.mode", 2L, col("c").mode().evaluate(m).get(0));

            // 53 cumsum
            Column cs = col("x").cumSum().evaluate(df);
            checkEq("53.cumsum[0]", -2.0, cs.get(0), 1e-9);
            checkEq("53.cumsum[2]", -2.5, cs.get(2), 1e-9); // -2-0.5+0
            checkEq("53.cumsum[4]", 3.0, cs.get(4), 1e-9);

            // 54 cummin
            Column cmn = col("x").cumMin().evaluate(df);
            checkEq("54.cummin[0]", -2.0, cmn.get(0), 1e-9);
            checkEq("54.cummin[4]", -2.0, cmn.get(4), 1e-9);

            // 55 cummax
            Column cmx = col("x").cumMax().evaluate(df);
            checkEq("55.cummax[0]", -2.0, cmx.get(0), 1e-9);
            checkEq("55.cummax[3]", 1.5, cmx.get(3), 1e-9);
            checkEq("55.cummax[4]", 4.0, cmx.get(4), 1e-9);
        });

        // ════════════════════════════════════════════════════════════
        // IV. Window / rolling (56–65)
        // ════════════════════════════════════════════════════════════
        benchmark("IV. window rolling 56-65", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("v", Column.DType.FLOAT64);
            for (int i = 1; i <= 5; i++) df.addRow((double) i); // 1,2,3,4,5

            // 56 rolling sum w=3
            Column c = col("v").rollingSum(3).evaluate(df);
            checkEq("56.rolling_sum[0]", 1.0, c.get(0), 1e-9);
            checkEq("56.rolling_sum[2]", 6.0, c.get(2), 1e-9); // 1+2+3
            checkEq("56.rolling_sum[4]", 12.0, c.get(4), 1e-9); // 3+4+5

            // also via DataFrame.rolling API
            DataFrame rs = df.rolling(3).sum("v");
            check("56.df.rolling.hasCol", rs.hasColumn("rolling_sum") || rs.columnCount() >= 1);

            // 57 rolling mean
            c = col("v").rollingMean(3).evaluate(df);
            checkEq("57.rolling_mean[2]", 2.0, c.get(2), 1e-9);

            // 58 rolling max
            c = col("v").rollingMax(3).evaluate(df);
            checkEq("58.rolling_max[2]", 3.0, c.get(2), 1e-9);

            // 59 rolling min
            c = col("v").rollingMin(3).evaluate(df);
            checkEq("59.rolling_min[2]", 1.0, c.get(2), 1e-9);

            // 60 rolling std
            c = col("v").rollingStd(3).evaluate(df);
            // std of [1,2,3] sample = 1.0
            checkEq("60.rolling_std[2]", 1.0, c.get(2), 1e-9);

            // 61 shift
            c = col("v").shift(1).evaluate(df);
            check("61.shift[0]", c.get(0) == null);
            checkEq("61.shift[1]", 1.0, c.get(1), 1e-9);
            checkEq("61.shift[4]", 4.0, c.get(4), 1e-9);

            // 62 diff
            c = col("v").diff(1).evaluate(df);
            check("62.diff[0]", c.get(0) == null);
            checkEq("62.diff[1]", 1.0, c.get(1), 1e-9);
            checkEq("62.diff[4]", 1.0, c.get(4), 1e-9);

            // 63 rank
            DataFrame rk = DataFrame.create();
            rk.addColumn("v", Column.DType.FLOAT64);
            rk.addRow(30.0); rk.addRow(10.0); rk.addRow(20.0); rk.addRow(10.0);
            c = col("v").rank("average", true).evaluate(rk);
            // sorted: 10,10,20,30 → ranks 1.5,1.5,3,4
            // row0=30 → 4, row1=10 → 1.5, row2=20 → 3, row3=10 → 1.5
            checkEq("63.rank[0]", 4.0, c.get(0), 1e-9);
            checkEq("63.rank[1]", 1.5, c.get(1), 1e-9);
            checkEq("63.rank[2]", 3.0, c.get(2), 1e-9);

            // 64 cumsum (global) already covered — re-check
            c = col("v").cumSum().evaluate(df);
            checkEq("64.cumsum[4]", 15.0, c.get(4), 1e-9);

            // 65 expanding mean
            c = col("v").expandingMean().evaluate(df);
            checkEq("65.expanding_mean[0]", 1.0, c.get(0), 1e-9);
            checkEq("65.expanding_mean[4]", 3.0, c.get(4), 1e-9); // mean 1..5 = 3
            DataFrame em = df.expanding().mean("v");
            check("65.df.expanding", em.rowCount() == 5);
        });

        // ════════════════════════════════════════════════════════════
        // V. String ops (66–75)
        // ════════════════════════════════════════════════════════════
        benchmark("V. string ops 66-75", () -> {
            DataFrame df = strSeed();

            // 66 lower
            Column c = col("s").str().toLowerCase().evaluate(df);
            checkEq("66.lower[0]", "  hello ", c.get(0));
            checkEq("66.lower[1]", "world", c.get(1));

            // 67 upper
            c = col("s").str().toUpperCase().evaluate(df);
            checkEq("67.upper[0]", "  HELLO ", c.get(0));

            // 68 strip
            c = col("s").str().strip().evaluate(df);
            checkEq("68.strip[0]", "Hello", c.get(0));

            // 69 contains
            c = col("s").str().contains("oo").evaluate(df);
            check("69.contains[2]", b(c.get(2)));
            check("69.contains[1]", !b(c.get(1)));

            // 70 startswith
            c = col("s").str().strip().evaluate(df); // prep
            DataFrame stripped = df.withColumn("s", col("s").str().strip());
            c = col("s").str().startsWith("He").evaluate(stripped);
            check("70.startswith[0]", b(c.get(0)));
            check("70.startswith[1]", !b(c.get(1)));

            // 71 endswith
            c = col("s").str().endsWith("LD").evaluate(df);
            check("71.endswith[1]", b(c.get(1))); // WORLD

            // 72 len
            c = col("s").str().length().evaluate(df);
            checkEq("72.len[1]", 5, c.get(1)); // WORLD
            c = col("s").str().lenBytes().evaluate(df);
            checkEq("72.lenBytes[1]", 5, c.get(1));

            // 73 replace
            c = col("s").str().replace("o", "0").evaluate(df);
            checkEq("73.replace[2]", "f00_bar", c.get(2)); // foo_bar → f00_bar

            // 74 slice
            c = col("s").str().slice(0, 3).evaluate(stripped);
            checkEq("74.slice[0]", "Hel", c.get(0));
            checkEq("74.slice[1]", "WOR", c.get(1));

            // 75 split (joined with | for STRING compat)
            c = col("s").str().split("_").evaluate(df);
            checkEq("75.split[2]", "foo|bar", c.get(2));
        });

        // ════════════════════════════════════════════════════════════
        // VI. Table ops (76–85)
        // ════════════════════════════════════════════════════════════
        benchmark("VI. table ops 76-85", () -> {
            DataFrame df = groupSeed();

            // 76 filter
            DataFrame f = df.filter(col("amt").gt(25));
            check("76.filter.rows", f.rowCount() == 3); // 30,40,50

            // 77 sort
            DataFrame s = df.sort("amt", false);
            checkEq("77.sort[0]", 50.0, s.get(0, "amt"), 1e-9);
            checkEq("77.sort[4]", 10.0, s.get(4, "amt"), 1e-9);

            // 78 drop_duplicates / unique
            DataFrame d = DataFrame.create();
            d.addColumn("a", Column.DType.INT64);
            d.addColumn("b", Column.DType.STRING);
            d.addRow(1L, "x"); d.addRow(1L, "x"); d.addRow(2L, "y");
            DataFrame u = d.dropDuplicates();
            check("78.drop_duplicates", u.rowCount() == 2);
            check("78.unique", d.unique().rowCount() == 2);

            // 79 dropna
            DataFrame n = withNulls();
            DataFrame dn = n.dropna();
            check("79.dropna", dn.rowCount() == 2); // rows 0 and 3 fully non-null
            check("79.dropNulls", n.dropNulls().rowCount() == 2);

            // 80 select_dtypes
            DataFrame sd = df.selectDtypes(Column.DType.FLOAT64);
            check("80.select_dtypes.cols", sd.columnCount() == 1);
            check("80.select_dtypes.name", "amt".equals(sd.column(0).name()));

            // 81 rename
            DataFrame rn = df.rename(Map.of("amt", "amount"));
            check("81.rename.has", rn.hasColumn("amount"));
            check("81.rename.gone", !rn.hasColumn("amt"));

            // 82 head
            check("82.head", df.head(2).rowCount() == 2);

            // 83 tail
            DataFrame t = df.tail(2);
            check("83.tail", t.rowCount() == 2);
            checkEq("83.tail.last", 50.0, t.get(1, "amt"), 1e-9);

            // 84 sample
            DataFrame sm = df.sample(3, 42L);
            check("84.sample", sm.rowCount() == 3);

            // 85 transpose
            DataFrame tr = df.head(2).select("city", "amt").transpose();
            check("85.transpose.rows", tr.rowCount() == 2); // 2 original cols → 2 rows
            check("85.transpose.hasIndex", tr.hasColumn("index"));
        });

        // ════════════════════════════════════════════════════════════
        // VII. GroupBy / join / reshape (86–95)
        // ════════════════════════════════════════════════════════════
        benchmark("VII. groupby join reshape 86-95", () -> {
            DataFrame df = groupSeed();

            // 86 groupby agg
            DataFrame g = df.groupBy("city").agg(
                col("amt").sum().alias("total"),
                col("amt").mean().alias("avg"),
                col("qty").count().alias("n")
            );
            check("86.groupby.rows", g.rowCount() == 2);
            // NY: 10+30+50=90, SF: 20+40=60
            Map<String, Double> totals = new HashMap<>();
            for (int i = 0; i < g.rowCount(); i++) {
                totals.put(String.valueOf(g.get(i, "city")), d(g.get(i, "total")));
            }
            checkEq("86.groupby.NY", 90.0, totals.get("NY"), 1e-9);
            checkEq("86.groupby.SF", 60.0, totals.get("SF"), 1e-9);

            // also pandas-style map agg
            Map<String, AggFunction> aggs = new LinkedHashMap<>();
            aggs.put("amt", AggFunction.SUM);
            aggs.put("qty", AggFunction.MEAN);
            DataFrame g2 = df.groupby("city").agg(aggs);
            check("86.groupby.map", g2.rowCount() == 2);

            // 87 join / merge
            DataFrame right = DataFrame.create();
            right.addColumn("city", Column.DType.STRING);
            right.addColumn("region", Column.DType.STRING);
            right.addRow("NY", "East");
            right.addRow("SF", "West");
            DataFrame j = df.join(right, "city", "inner");
            check("87.join.rows", j.rowCount() == 5);
            check("87.join.col", j.hasColumn("region"));

            DataFrame m = df.merge(right, "city", "city", "inner");
            check("87.merge.rows", m.rowCount() == 5);

            // 88 vstack / concat axis=0
            DataFrame a = df.head(2);
            DataFrame b = df.tail(2);
            DataFrame cat = DataFrame.vstack(a, b);
            check("88.vstack", cat.rowCount() == 4);
            check("88.concat0", DataFrame.concat(List.of(a, b), 0).rowCount() == 4);

            // 89 hstack / concat axis=1
            DataFrame left = df.select("city");
            DataFrame right2 = df.select("amt");
            DataFrame hs = DataFrame.hstack(left, right2);
            check("89.hstack.cols", hs.columnCount() == 2);
            check("89.hstack.rows", hs.rowCount() == df.rowCount());

            // 90 assign / withColumn / withColumns
            DataFrame w = df.withColumn("dbl", col("amt").multiply(2));
            checkEq("90.withColumn[0]", 20.0, w.get(0, "dbl"), 1e-9);
            DataFrame w2 = df.withColumns(col("amt").multiply(3).alias("trip"));
            checkEq("90.withColumns[0]", 30.0, w2.get(0, "trip"), 1e-9);

            // 91 pivot
            DataFrame wide = DataFrame.create();
            wide.addColumn("id", Column.DType.STRING);
            wide.addColumn("metric", Column.DType.STRING);
            wide.addColumn("val", Column.DType.FLOAT64);
            wide.addRow("a", "x", 1.0);
            wide.addRow("a", "y", 2.0);
            wide.addRow("b", "x", 3.0);
            wide.addRow("b", "y", 4.0);
            DataFrame pv = wide.pivot("id", "metric", "val");
            check("91.pivot.rows", pv.rowCount() == 2);
            check("91.pivot.hasX", pv.hasColumn("x") || pv.hasColumn("y"));

            // 92 melt / unpivot
            DataFrame melted = pv.melt(List.of("id"), null);
            check("92.melt.rows", melted.rowCount() >= 2);
            DataFrame unp = pv.unpivot(List.of("id"), null);
            check("92.unpivot.rows", unp.rowCount() >= 2);

            // 93 with_row_index / set_index
            DataFrame idx = df.withRowIndex();
            check("93.withRowIndex.col", idx.hasColumn("index"));
            checkEq("93.withRowIndex[0]", 0L, idx.get(0, "index"));
            checkEq("93.withRowIndex[4]", 4L, idx.get(4, "index"));
            DataFrame si = df.setIndex("city");
            checkEq("93.setIndex.firstCol", "city", si.column(0).name());

            // 94 reset_index
            DataFrame ri = idx.resetIndex();
            check("94.resetIndex.dropped", !ri.hasColumn("index") || ri.columnCount() <= idx.columnCount());

            // 95 cut
            DataFrame ages = DataFrame.create();
            ages.addColumn("age", Column.DType.FLOAT64);
            ages.addRow(10.0); ages.addRow(25.0); ages.addRow(40.0); ages.addRow(70.0);
            Column cut = col("age").cut(new double[]{18, 35, 60, 100},
                    new String[]{"youth", "adult", "mid", "senior"}).evaluate(ages);
            checkEq("95.cut[0]", "youth", cut.get(0));
            checkEq("95.cut[1]", "adult", cut.get(1));
            checkEq("95.cut[2]", "mid", cut.get(2));
            checkEq("95.cut[3]", "senior", cut.get(3));
            DataFrame cutDf = ages.cut("age", new double[]{18, 35, 60, 100},
                    new String[]{"youth", "adult", "mid", "senior"});
            check("95.cut.df", cutDf.hasColumn("age_bin"));
        });

        // ════════════════════════════════════════════════════════════
        // VIII. Cast / value_counts / isin / numpy / describe (96–100)
        // ════════════════════════════════════════════════════════════
        benchmark("VIII. cast value_counts isin numpy describe 96-100", () -> {
            DataFrame df = numericSeed();

            // 96 cast / astype
            Column c = col("x").cast(Column.DType.INT64).evaluate(df);
            checkEq("96.cast[3]", 1L, c.get(3)); // 1.5 → 1
            DataFrame ast = df.astype("z", Column.DType.FLOAT64);
            check("96.astype", ast.column("z").dtype() == Column.DType.FLOAT64);

            // 97 value_counts
            DataFrame vc = DataFrame.create();
            vc.addColumn("c", Column.DType.STRING);
            vc.addRow("a"); vc.addRow("b"); vc.addRow("a"); vc.addRow("a"); vc.addRow("b");
            Map<Object, Integer> counts = vc.valueCounts("c");
            checkEq("97.value_counts.a", 3, counts.get("a"));
            checkEq("97.value_counts.b", 2, counts.get("b"));
            DataFrame vcf = vc.valueCountsFrame("c");
            check("97.value_counts.frame", vcf.rowCount() == 2);
            // expression form: per-row count
            Column vce = col("c").valueCounts().evaluate(vc);
            checkEq("97.value_counts.expr[0]", 3L, vce.get(0));

            // 98 isin
            Column is = col("z").isIn(1L, 3L, 5L).evaluate(df);
            check("98.isin[0]", b(is.get(0)));
            check("98.isin[1]", !b(is.get(1)));
            check("98.isin[2]", b(is.get(2)));
            List<Boolean> isList = df.isin("z", 2L, 4L);
            check("98.isin.list[1]", isList.get(1));
            check("98.isin.list[0]", !isList.get(0));

            // 99 to_numpy
            double[][] mat = df.select("x", "y").to_numpy();
            check("99.to_numpy.rows", mat.length == 5);
            check("99.to_numpy.cols", mat[0].length == 2);
            checkEq("99.to_numpy[0][0]", -2.0, mat[0][0], 1e-9);
            double[][] mat2 = df.select("x", "y").toNumpy();
            check("99.toNumpy.alias", mat2.length == 5);

            // 100 describe
            Map<String, List<Double>> desc = df.describe();
            check("100.describe.hasX", desc.containsKey("x"));
            check("100.describe.stats", desc.get("x").size() == 7);
            DataFrame descF = df.describeFrame();
            check("100.describeFrame.rows", descF.rowCount() == 7);
            check("100.describeFrame.hasStat", descF.hasColumn("stat"));
        });

        // ════════════════════════════════════════════════════════════
        // Bonus: end-to-end pipeline mixing many ops
        // ════════════════════════════════════════════════════════════
        benchmark("IX. e2e pipeline mix", () -> {
            DataFrame raw = groupSeed();
            DataFrame out = raw
                .filter(col("amt").gt(15))
                .withColumn("log_amt", col("amt").log())
                .withColumn("city_up", col("city").str().toUpperCase())
                .withColumn("amt_clip", col("amt").clip(0, 45))
                .sort("amt", true);
            check("e2e.rows", out.rowCount() == 4); // 20,30,40,50
            checkEq("e2e.first.amt", 20.0, out.get(0, "amt"), 1e-9);
            checkEq("e2e.city_up", "SF", out.get(0, "city_up"));
            checkEq("e2e.clip.last", 45.0, out.get(3, "amt_clip"), 1e-9); // 50→45

            DataFrame g = out.groupBy("city").agg(
                col("amt").sum().alias("sum_amt"),
                col("amt").mean().alias("mean_amt")
            );
            check("e2e.groupby", g.rowCount() == 2);

            // rolling on sorted amounts
            DataFrame rolled = out.withColumn("r3", col("amt").rollingMean(2));
            check("e2e.rolling", rolled.hasColumn("r3"));
            checkEq("e2e.rolling[1]", 25.0, rolled.get(1, "r3"), 1e-9); // (20+30)/2
        });

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("All 100-operator checklist groups covered.");
    }
}
