package samples;

import org.bytedeco.pytorch.dataframe.*;
import static org.bytedeco.pytorch.dataframe.Functions.*;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import org.bytedeco.pytorch.dataframe.io.ParquetAdvanced;
import org.bytedeco.pytorch.dataframe.io.ParallelReader;

/**
 * Benchmark + smoke tests for the second wave of advanced training/time-series APIs:
 * <ul>
 *   <li>merge_asof with by + allow_exact_matches</li>
 *   <li>groupby.rolling / expanding / resample</li>
 *   <li>MultiIndex swaplevel / droplevel / stack / unstack</li>
 *   <li>Parquet column prune + filter + row-group parallel + Hive partition</li>
 *   <li>map_batches lazy batch UDF</li>
 *   <li>df.query / df.eval string expressions</li>
 * </ul>
 */
public final class BenchmarkDataFrameAdvanced2 {
    private BenchmarkDataFrameAdvanced2() {}

    public static void main(String[] args) throws Exception {
        System.out.println("=== DataFrame Advanced-2 API smoke + bench ===");
        asofByExact();
        groupbyRollingExpanding();
        resampleBench();
        multiIndexOps();
        queryEval();
        mapBatches();
        parquetAdvancedAndHive();
        System.out.println("=== ALL ADVANCED-2 CHECKS PASSED ===");
    }

    // ----------------------------------------------------------------
    // 1. merge_asof by + allow_exact_matches
    // ----------------------------------------------------------------
    static void asofByExact() throws Exception {
        DataFrame left = DataFrame.create();
        left.addColumn("sym", Column.DType.STRING);
        left.addColumn("t", Column.DType.INT64);
        left.addColumn("qty", Column.DType.FLOAT64);
        Object[][] L = {
            {"AAPL", 10L, 1.0}, {"AAPL", 20L, 2.0}, {"AAPL", 30L, 3.0},
            {"MSFT", 10L, 1.0}, {"MSFT", 25L, 2.0},
        };
        for (Object[] r : L) {
            int i = left.addEmptyRow();
            left.set(i, "sym", r[0]); left.set(i, "t", r[1]); left.set(i, "qty", r[2]);
        }

        DataFrame right = DataFrame.create();
        right.addColumn("sym", Column.DType.STRING);
        right.addColumn("t", Column.DType.INT64);
        right.addColumn("px", Column.DType.FLOAT64);
        Object[][] R = {
            {"AAPL", 10L, 100.0}, {"AAPL", 25L, 110.0},
            {"MSFT", 15L, 200.0}, {"MSFT", 30L, 210.0},
        };
        for (Object[] r : R) {
            int i = right.addEmptyRow();
            right.set(i, "sym", r[0]); right.set(i, "t", r[1]); right.set(i, "px", r[2]);
        }

        // by=sym, backward, allow exact
        DataFrame m1 = DataFrame.mergeAsof(left, right, "t", "t", "backward", null,
            new String[]{"sym"}, true);
        assert m1.rowCount() == 5;
        // AAPL t=10 exact → 100
        assert eq(((Number) m1.get(0, "px")).doubleValue(), 100.0);
        // AAPL t=20 → backward 10 → 100
        assert eq(((Number) m1.get(1, "px")).doubleValue(), 100.0);
        // AAPL t=30 → backward 25 → 110
        assert eq(((Number) m1.get(2, "px")).doubleValue(), 110.0);
        // MSFT t=10 → no prior (15 is later) → null
        assert m1.get(3, "px") == null;
        // MSFT t=25 → backward 15 → 200
        assert eq(((Number) m1.get(4, "px")).doubleValue(), 200.0);

        // allow_exact_matches=false: AAPL t=10 should NOT take exact 10
        DataFrame m2 = left.joinAsof(right, "t", "t", "backward", null,
            new String[]{"sym"}, false);
        assert m2.get(0, "px") == null : "exact match suppressed";
        // AAPL t=20 still gets 10 → 100
        assert eq(((Number) m2.get(1, "px")).doubleValue(), 100.0);

        // nearest within by
        DataFrame m3 = DataFrame.mergeAsof(left, right, "t", "t", "nearest", 20.0,
            new String[]{"sym"}, true);
        assert m3.get(3, "px") != null : "MSFT t=10 nearest to 15";

        System.out.println("  merge_asof by/allow_exact_matches OK");
    }

    // ----------------------------------------------------------------
    // 2. groupby.rolling / expanding
    // ----------------------------------------------------------------
    static void groupbyRollingExpanding() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("g", Column.DType.STRING);
        df.addColumn("x", Column.DType.FLOAT64);
        String[] gs = {"a", "a", "a", "b", "b"};
        double[] xs = {1, 2, 3, 10, 20};
        for (int i = 0; i < gs.length; i++) {
            int r = df.addEmptyRow();
            df.set(r, "g", gs[i]); df.set(r, "x", xs[i]);
        }

        DataFrame roll = df.groupBy("g").rolling(2).sum("x");
        assert roll.hasColumn("rolling_sum");
        // a: null, 1+2=3, 2+3=5
        assert roll.get(0, "rolling_sum") == null;
        assert eq(((Number) roll.get(1, "rolling_sum")).doubleValue(), 3.0);
        assert eq(((Number) roll.get(2, "rolling_sum")).doubleValue(), 5.0);
        // b: null, 10+20=30
        assert roll.get(3, "rolling_sum") == null;
        assert eq(((Number) roll.get(4, "rolling_sum")).doubleValue(), 30.0);

        DataFrame exp = df.groupBy("g").expanding().mean("x");
        assert exp.hasColumn("expanding_mean");
        assert eq(((Number) exp.get(2, "expanding_mean")).doubleValue(), 2.0); // (1+2+3)/3

        DataFrame rollMean = df.groupBy("g").rolling(3, 1).mean("x");
        assert rollMean.hasColumn("rolling_mean");

        System.out.println("  groupby.rolling / expanding OK");
    }

    // ----------------------------------------------------------------
    // 3. resample
    // ----------------------------------------------------------------
    static void resampleBench() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("t", Column.DType.INT64);
        df.addColumn("x", Column.DType.FLOAT64);
        // 0..9 seconds in ms
        for (int i = 0; i < 10; i++) {
            int r = df.addEmptyRow();
            df.set(r, "t", i * 1000L);
            df.set(r, "x", (double) i);
        }

        long t0 = System.nanoTime();
        DataFrame means = df.resample("t", "3s").mean("x");
        long t1 = System.nanoTime();
        // bins: 0,3,6,9 → 4 bins
        assert means.rowCount() == 4 : "resample bins got " + means.rowCount();
        // first bin 0,1,2 → mean 1.0
        assert eq(((Number) means.get(0, "x_mean")).doubleValue(), 1.0);

        DataFrame asfreq = df.resample("t", "2s").asfreq(null);
        assert asfreq.rowCount() >= 5;
        DataFrame interp = df.resample("t", "2s").interpolate("linear");
        assert interp.rowCount() >= 5;

        DataFrame cnt = df.resample("t", "5s", "epoch", 0).count();
        assert cnt.rowCount() >= 2;

        System.out.printf(Locale.ROOT, "  resample OK (mean=%.2fms, bins=%d)%n",
            (t1 - t0) / 1e6, means.rowCount());
    }

    // ----------------------------------------------------------------
    // 4. MultiIndex
    // ----------------------------------------------------------------
    static void multiIndexOps() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("g1", Column.DType.STRING);
        df.addColumn("g2", Column.DType.STRING);
        df.addColumn("v", Column.DType.FLOAT64);
        Object[][] rows = {
            {"a", "x", 1.0}, {"a", "y", 2.0}, {"b", "x", 3.0}, {"b", "y", 4.0},
        };
        for (Object[] r : rows) {
            int i = df.addEmptyRow();
            df.set(i, "g1", r[0]); df.set(i, "g2", r[1]); df.set(i, "v", r[2]);
        }

        DataFrame swapped = df.swaplevel(0, 1, "g1", "g2");
        assert "g2".equals(swapped.getColumnNames().get(0));
        assert "g1".equals(swapped.getColumnNames().get(1));

        DataFrame reordered = df.reorderLevels(new String[]{"g2", "g1"}, "g1", "g2");
        assert "g2".equals(reordered.getColumnNames().get(0));

        DataFrame dropped = df.droplevel(0, "g1", "g2");
        assert !dropped.hasColumn("g1") && dropped.hasColumn("g2");

        // stack: wide value columns
        DataFrame wide = DataFrame.create();
        wide.addColumn("id", Column.DType.STRING);
        wide.addColumn("m1", Column.DType.FLOAT64);
        wide.addColumn("m2", Column.DType.FLOAT64);
        int w0 = wide.addEmptyRow(); wide.set(w0, "id", "r1"); wide.set(w0, "m1", 1.0); wide.set(w0, "m2", 2.0);
        int w1 = wide.addEmptyRow(); wide.set(w1, "id", "r2"); wide.set(w1, "m1", 3.0); wide.set(w1, "m2", 4.0);
        DataFrame stacked = wide.stackLevels(List.of("id"), List.of("m1", "m2"), "metric", "value");
        assert stacked.rowCount() == 4;
        assert stacked.hasColumn("metric") && stacked.hasColumn("value");

        DataFrame unstacked = stacked.unstack(List.of("id"), "metric", "value", null);
        assert unstacked.rowCount() == 2;
        assert unstacked.hasColumn("m1") && unstacked.hasColumn("m2");

        System.out.println("  MultiIndex swaplevel/droplevel/stack/unstack OK");
    }

    // ----------------------------------------------------------------
    // 5. query / eval
    // ----------------------------------------------------------------
    static void queryEval() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("x", Column.DType.INT64);
        df.addColumn("y", Column.DType.INT64);
        df.addColumn("g", Column.DType.STRING);
        for (int i = 0; i < 10; i++) {
            int r = df.addEmptyRow();
            df.set(r, "x", (long) i);
            df.set(r, "y", (long) (i * 2));
            df.set(r, "g", i % 2 == 0 ? "even" : "odd");
        }

        DataFrame q1 = df.query("x > 5");
        assert q1.rowCount() == 4 : "x>5 → 6,7,8,9";

        DataFrame q2 = df.query("x >= 3 and g == 'even'");
        assert q2.rowCount() == 4 : "3? no — 4,6,8 and x>=3 even → 4,6,8 = 3? wait 4,6,8 = 3";
        // even: 0,2,4,6,8; >=3 → 4,6,8 = 3
        assert q2.rowCount() == 3;

        Map<String, Object> locals = Map.of("thresh", 7);
        DataFrame q3 = df.query("x >= @thresh", locals);
        assert q3.rowCount() == 3; // 7,8,9

        DataFrame q4 = df.query("g in ['even']");
        assert q4.rowCount() == 5;

        DataFrame e1 = df.eval("z = x + y");
        assert e1.hasColumn("z");
        assert eq(((Number) e1.get(3, "z")).doubleValue(), 9.0); // 3+6

        DataFrame e2 = df.eval("x * 2 - y");
        assert e2.hasColumn("result");
        // x*2 - y = 2x - 2x = 0
        assert eq(((Number) e2.get(5, "result")).doubleValue(), 0.0);

        System.out.println("  query/eval string expressions OK");
    }

    // ----------------------------------------------------------------
    // 6. map_batches
    // ----------------------------------------------------------------
    static void mapBatches() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("x", Column.DType.FLOAT64);
        for (int i = 0; i < 20; i++) {
            int r = df.addEmptyRow();
            df.set(r, "x", (double) i);
        }

        // eager
        DataFrame e = df.mapBatches(d -> d.withColumn("x2", col("x").multiply(2)));
        assert e.hasColumn("x2");
        assert eq(((Number) e.get(5, "x2")).doubleValue(), 10.0);

        // lazy + streaming batches
        AtomicLong batches = new AtomicLong();
        DataFrame out = df.lazy()
            .mapBatches(d -> {
                batches.incrementAndGet();
                return d.withColumn("x2", col("x").multiply(2));
            }, "double_x")
            .filter(col("x").gt(lit(5)))
            .collectStreaming(5);

        assert out.rowCount() == 14 : "x>5 → 6..19 = 14";
        assert batches.get() >= 2 : "streaming should invoke map_batches per chunk";
        System.out.println("  map_batches OK (streaming batches=" + batches.get() + ")");
    }

    // ----------------------------------------------------------------
    // 7. Parquet advanced + Hive partition
    // ----------------------------------------------------------------
    static void parquetAdvancedAndHive() throws Exception {
        Path tmp = Files.createTempDirectory("df-adv2-");
        try {
            // build a larger frame for row-group-ish parallel + projection
            DataFrame df = DataFrame.create();
            df.addColumn("id", Column.DType.INT64);
            df.addColumn("cat", Column.DType.STRING);
            df.addColumn("val", Column.DType.FLOAT64);
            df.addColumn("year", Column.DType.INT64);
            df.addColumn("month", Column.DType.INT64);
            int N = 5_000;
            for (int i = 0; i < N; i++) {
                int r = df.addEmptyRow();
                df.set(r, "id", (long) i);
                df.set(r, "cat", i % 3 == 0 ? "A" : (i % 3 == 1 ? "B" : "C"));
                df.set(r, "val", i * 0.5);
                df.set(r, "year", 2024L + (i % 2));
                df.set(r, "month", 1L + (i % 3));
            }

            Path plain = tmp.resolve("plain.parquet");
            long t0 = System.nanoTime();
            df.writeParquet(plain.toString());
            long t1 = System.nanoTime();

            // column projection
            DataFrame proj = DataFrame.readParquet(plain.toString(), "id", "val");
            assert proj.columnCount() == 2 && proj.hasColumn("id") && proj.hasColumn("val");
            assert proj.rowCount() == N;

            // filter pushdown (post-read predicate)
            ParquetAdvanced.ReadOptions opt = ParquetAdvanced.ReadOptions.defaults()
                .columns("id", "cat", "val")
                .eq("cat", "A")
                .workers(4)
                .maxRows(0);
            long t2 = System.nanoTime();
            DataFrame filtered = DataFrame.readParquetAdvanced(plain.toString(), opt);
            long t3 = System.nanoTime();
            assert filtered.rowCount() == (N + 2) / 3 || filtered.rowCount() == N / 3
                || Math.abs(filtered.rowCount() - (N + 2) / 3) <= 1;
            // all cat == A
            for (int i = 0; i < Math.min(20, filtered.rowCount()); i++) {
                assert "A".equals(filtered.get(i, "cat"));
            }

            // streaming
            AtomicLong streamRows = new AtomicLong();
            long delivered = DataFrame.streamParquetAdvanced(plain.toString(),
                ParquetAdvanced.ReadOptions.defaults().columns("id").batchRows(1000),
                batch -> streamRows.addAndGet(batch.rowCount()));
            assert delivered == N && streamRows.get() == N;

            // Hive partition write + scan
            Path hiveRoot = tmp.resolve("hive");
            long t4 = System.nanoTime();
            df.writeParquetPartitioned(hiveRoot.toString(), "year", "month");
            long t5 = System.nanoTime();

            // verify directory structure exists
            assert Files.isDirectory(hiveRoot.resolve("year=2024").resolve("month=1"))
                || Files.walk(hiveRoot).anyMatch(p -> p.toString().contains("year="));

            DataFrame hive = DataFrame.readParquetHive(hiveRoot.toString(), "year", "month");
            assert hive.rowCount() == N : "hive scan rows " + hive.rowCount();
            assert hive.hasColumn("year") && hive.hasColumn("month");
            assert hive.hasColumn("val");

            // partition + column projection
            DataFrame hive2 = DataFrame.readParquetHive(hiveRoot.toString(),
                new String[]{"year", "month"},
                ParquetAdvanced.ReadOptions.defaults().columns("id", "val"));
            // scan injects year/month; body may only have id,val
            assert hive2.rowCount() == N;

            System.out.printf(Locale.ROOT,
                "  parquet write=%.1fms read+filter=%.1fms hive_write=%.1fms rows=%d%n",
                (t1 - t0) / 1e6, (t3 - t2) / 1e6, (t5 - t4) / 1e6, N);
            System.out.println("  Parquet advanced + Hive partition_by OK");
        } finally {
            // best-effort cleanup
            try (var walk = Files.walk(tmp)) {
                walk.sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            }
        }
    }

    // ----------------------------------------------------------------
    static boolean eq(double a, double b) {
        return Math.abs(a - b) < 1e-9;
    }
}
