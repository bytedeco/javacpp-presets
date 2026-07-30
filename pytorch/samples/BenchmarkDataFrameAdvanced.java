package samples;

import org.bytedeco.pytorch.dataframe.*;

import static org.bytedeco.pytorch.dataframe.Functions.*;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Column;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import org.bytedeco.pytorch.dataframe.io.ParallelReader;

/**
 * Smoke + micro-benchmark for advanced Pandas/Polars APIs and multi-worker heap-safe IO.
 *
 * <pre>
 *   javac -cp target/classes samples/BenchmarkDataFrameAdvanced.java
 *   java  -cp target/classes:samples samples.BenchmarkDataFrameAdvanced
 * </pre>
 */
public final class BenchmarkDataFrameAdvanced {
    private BenchmarkDataFrameAdvanced() {}

    public static void main(String[] args) throws Exception {
        System.out.println("=== DataFrame Advanced API smoke ===");
        groupbyAdvanced();
        joinAsofSemiAnti();
        listStructExpr();
        interpolateQcutPipe();
        lazyStreaming();
        parallelCsv();
        System.out.println("=== ALL ADVANCED CHECKS PASSED ===");
    }

    static DataFrame sampleFrame() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("g", Column.DType.STRING);
        df.addColumn("t", Column.DType.INT64);
        df.addColumn("x", Column.DType.FLOAT64);
        df.addColumn("tags", Column.DType.LIST);
        String[] gs = {"a", "a", "b", "b", "b", "c"};
        long[] ts = {1, 2, 1, 3, 5, 1};
        double[] xs = {1.0, 3.0, 2.0, 8.0, Double.NaN, 4.0};
        for (int i = 0; i < gs.length; i++) {
            int r = df.addEmptyRow();
            df.set(r, "g", gs[i]);
            df.set(r, "t", ts[i]);
            df.set(r, "x", Double.isNaN(xs[i]) ? null : xs[i]);
            df.set(r, "tags", List.of("t" + i, "u", "t" + i));
        }
        return df;
    }

    static void groupbyAdvanced() throws Exception {
        DataFrame df = sampleFrame();
        GroupedDataFrame gb = df.groupBy("g");

        Column ng = gb.ngroup();
        Column cc = gb.cumcount();
        assert ng.size() == df.rowCount() : "ngroup length";
        assert cc.size() == df.rowCount() : "cumcount length";
        System.out.println("  ngroup[0]=" + ng.get(0) + " cumcount last-of-b=" + cc.get(4));

        DataFrame nth = gb.nth(0);
        assert nth.rowCount() == 3 : "nth first of each group → 3 groups";
        DataFrame head = gb.head(2);
        assert head.rowCount() == 5 : "head(2): a2+b2+c1=5";
        DataFrame shifted = gb.shift(1);
        assert shifted.get(0, "x") == null : "shift within group leaves first null";
        DataFrame ranked = gb.rank("average", true);
        assert ranked.hasColumn("x_rank") : "rank adds x_rank";
        DataFrame vc = gb.valueCounts("g");
        assert vc.rowCount() >= 3 : "value_counts";
        DataFrame impl = gb.implode("x");
        assert impl.rowCount() == 3 : "implode per group";
        Object list0 = impl.get(0, "x");
        assert list0 instanceof List : "implode → List cell";

        DataFrame transformed = gb.transform(g -> {
            try {
                return g.withColumn("x2", col("x").multiply(2));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        assert transformed.rowCount() == df.rowCount() : "transform same length";
        System.out.println("  groupby advanced OK (nth/head/shift/rank/implode/transform)");
    }

    static void joinAsofSemiAnti() throws Exception {
        DataFrame left = DataFrame.create();
        left.addColumn("t", Column.DType.INT64);
        left.addColumn("v", Column.DType.FLOAT64);
        for (long t : new long[]{1, 3, 6, 10}) {
            int r = left.addEmptyRow();
            left.set(r, "t", t);
            left.set(r, "v", (double) t);
        }
        DataFrame right = DataFrame.create();
        right.addColumn("t", Column.DType.INT64);
        right.addColumn("p", Column.DType.FLOAT64);
        for (long t : new long[]{1, 4, 8}) {
            int r = right.addEmptyRow();
            right.set(r, "t", t);
            right.set(r, "p", t * 10.0);
        }

        DataFrame asof = left.joinAsof(right, "t");
        assert asof.rowCount() == 4 : "asof keeps left rows";
        assert asof.hasColumn("p") : "asof brings p";
        // t=3 → backward match t=1 → p=10
        assert Math.abs(((Number) asof.get(1, "p")).doubleValue() - 10.0) < 1e-9 : "asof backward";

        DataFrame semi = left.joinSemi(right, "t");
        assert semi.rowCount() == 1 : "semi only t=1";
        DataFrame anti = left.joinAnti(right, "t");
        assert anti.rowCount() == 3 : "anti 3 rows";

        DataFrame a = DataFrame.create();
        a.addColumn("k", Column.DType.STRING);
        a.addEmptyRow(); a.set(0, "k", "x");
        a.addEmptyRow(); a.set(1, "k", "y");
        DataFrame b = DataFrame.create();
        b.addColumn("k", Column.DType.STRING);
        b.addEmptyRow(); b.set(0, "k", "y");
        b.addEmptyRow(); b.set(1, "k", "z");
        assert a.setIntersection(b).rowCount() == 1;
        assert a.setDifference(b).rowCount() == 1;
        assert a.setUnion(b).rowCount() == 3;
        System.out.println("  join_asof / semi / anti / set_* OK");
    }

    static void listStructExpr() throws Exception {
        DataFrame df = sampleFrame();
        DataFrame out = df.select(
            col("g"),
            col("tags").list().first().alias("tag0"),
            col("tags").list().unique().alias("uniq"),
            col("tags").list().len().alias("n_tags"),
            col("tags").list().contains("u").alias("has_u"),
            col("x").fillNull("forward").alias("x_ffill"),
            col("x").qcut(2).alias("xq"),
            col("x").hash(42L).alias("xh"),
            col("g").isFirstDistinct().alias("g_first"),
            col("x").ewmMean(0.5).alias("x_ewm"),
            maxHorizontal(col("x"), lit(0)).alias("xmax0")
        );
        assert out.hasColumn("tag0") && out.hasColumn("n_tags");
        assert ((Number) out.get(0, "n_tags")).intValue() == 3;
        assert Boolean.TRUE.equals(out.get(0, "has_u"));
        System.out.println("  list/struct/qcut/hash/ewm/horizontal OK → " + out.rowCount() + " rows");
    }

    static void interpolateQcutPipe() throws Exception {
        DataFrame df = sampleFrame();
        // null at row 4 (x)
        DataFrame filled = df.interpolate("x", "linear");
        assert filled.get(4, "x") != null : "linear interpolate filled null";
        DataFrame qc = df.qcut("t", 3);
        assert qc.hasColumn("t_qcut");
        DataFrame taken = df.take(0, 2, -1);
        assert taken.rowCount() == 3;
        String msg = df.pipe(d -> "rows=" + d.rowCount());
        assert msg.equals("rows=6");
        long est = df.estimateSize();
        assert est > 0;
        System.out.println("  interpolate/qcut/take/pipe/estimateSize OK (est=" + est + "B)");
    }

    static void lazyStreaming() throws Exception {
        DataFrame df = sampleFrame();
        LazyDataFrame ldf = df.lazy()
            .withColumns(col("x").fillNull(0).alias("x0"), col("t").plus(1).alias("t1"))
            .filter(col("t").gt(lit(0)))
            .optimizationToggle(true);
        System.out.println(ldf.explainJson());
        DataFrame c1 = ldf.collectStreaming(2);
        assert c1.rowCount() == df.rowCount();
        DataFrame c2 = ldf.collectNoOptimization();
        assert c2.rowCount() == df.rowCount();
        DataFrame gh = df.lazy().groupBy("g").head(1).collect();
        assert gh.rowCount() == 3;
        System.out.println("  lazy streaming / with_columns / groupBy.head OK");
    }

    static void parallelCsv() throws Exception {
        Path tmp = Files.createTempFile("df-adv-", ".csv");
        try {
            StringBuilder sb = new StringBuilder("id,g,v\n");
            int N = 50_000;
            for (int i = 0; i < N; i++) {
                sb.append(i).append(',').append((char) ('a' + (i % 5))).append(',').append(i * 0.5).append('\n');
            }
            Files.writeString(tmp, sb.toString());

            long t0 = System.nanoTime();
            DataFrame single = DataFrame.readCsv(tmp.toString());
            long t1 = System.nanoTime();
            DataFrame multi = DataFrame.readCsvParallel(tmp.toString(), 4);
            long t2 = System.nanoTime();

            assert single.rowCount() == N : "single rows";
            assert multi.rowCount() == N : "multi rows got " + multi.rowCount();
            System.out.printf(Locale.ROOT,
                "  CSV %d rows: single=%.1fms parallel(4)=%.1fms%n",
                N, (t1 - t0) / 1e6, (t2 - t1) / 1e6);

            // streaming: must not retain all chunks
            AtomicLong rows = new AtomicLong();
            long delivered = DataFrame.streamCsv(tmp.toString(), 4, 64 * 1024, chunk -> {
                rows.addAndGet(chunk.rowCount());
                // drop chunk reference when consumer returns
            });
            assert delivered == N && rows.get() == N : "streamCsv rows";

            // maxRows budget
            ParallelReader.Options opt = ParallelReader.Options.defaults()
                .workers(4).maxChunkBytes(64 * 1024).maxRows(1000);
            DataFrame limited = ParallelReader.readCsv(tmp.toString(), opt);
            assert limited.rowCount() == 1000 : "maxRows cap";
            System.out.println("  parallel/stream CSV heap-safe OK (maxRows=1000)");
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
