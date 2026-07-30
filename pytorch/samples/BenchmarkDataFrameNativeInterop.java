package samples;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.dataloader.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.Dataset;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.data.ExampleVectorIterator;
import org.bytedeco.pytorch.data.TensorExample;
import org.bytedeco.pytorch.data.TensorExampleVector;
import org.bytedeco.pytorch.data.TensorExampleVectorIterator;
import org.bytedeco.pytorch.data.dataloader.RandomDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialTensorDataLoader;
import org.bytedeco.pytorch.data.datasets.JavaTensorDataset;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataLoader;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataset;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameJavaTensorDataset;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameNativeDataset;
import org.bytedeco.pytorch.dataframe.dataset.NativeBatchSupport;
import org.bytedeco.pytorch.dataframe.dataset.NativeViewOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Multi-dimensional correctness + throughput suite for DataFrameDataset /
 * DataFrameDataLoader interop with native Dataset / DataLoader.
 *
 * <p>Dimensions covered:
 * <ul>
 *   <li>API surface (instanceof Dataset / JavaTensorDataset)</li>
 *   <li>Sample parity (pure-Java get vs asDataset.get)</li>
 *   <li>Batch parity (Java loader vs native sequential stacked)</li>
 *   <li>Conversion round-trip (to/from TensorDataset, feature-label tensors, native)</li>
 *   <li>Feature modes (AUTO scalars / FIRST_SEQUENCE / PRIMARY)</li>
 *   <li>Throughput (Java / native / tensor loaders) × N × batch</li>
 *   <li>Training smoke (tiny Linear on both loaders)</li>
 * </ul>
 *
 * <pre>
 *   java --enable-native-access=ALL-UNNAMED \
 *        --add-opens=java.base/java.nio=ALL-UNNAMED \
 *        -cp ... samples.BenchmarkDataFrameNativeInterop
 * </pre>
 *
 * Env: {@code MICROLENS_DIR} optional parquet root; synthetic data used otherwise.
 */
public class BenchmarkDataFrameNativeInterop {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<String[]> table = new ArrayList<>();

    static final String DEFAULT_DIR =
        "/Users/muller/Documents/code/cpp/VideoMMCTR/data/MicroLens_1M_x1";

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void check(String name, boolean cond) {
        if (cond) {
            passed++;
        } else {
            failed++;
            report.append("FAIL ").append(name).append('\n');
            System.out.println("    ✗ " + name);
        }
    }

    static void checkEq(String name, long expected, long actual) {
        check(name + " expected=" + expected + " actual=" + actual, expected == actual);
    }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("  ✓ " + name + " (" + ms + " ms)");
        } catch (Throwable t) {
            failed++;
            report.append("FAIL ").append(name).append(": ").append(t).append('\n');
            System.out.println("  ✗ " + name + " — " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    static void row(String path, int n, int batch, long ms, double rowsPerSec, String status) {
        table.add(new String[]{
            path, String.valueOf(n), String.valueOf(batch),
            String.valueOf(ms), String.format(Locale.US, "%.0f", rowsPerSec), status
        });
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        try {
            Tensor warm = torch.tensor(new float[]{1f, 2f, 3f});
            check("torch warmup", warm.numel() == 3);
        } catch (Throwable t) {
            System.out.println("WARN: torch warmup failed: " + t);
        }

        DataFrame df = loadOrSynthetic();
        System.out.println("rows=" + df.rowCount() + " cols=" + df.columnCount());

        // ========== 1. Build dataset ==========
        System.out.println("\n== 1. DataFrameDataset build ==");
        final DataFrameDataset[] holder = new DataFrameDataset[1];
        int nBuild = Math.min(df.rowCount(), 2048);
        benchmark("builder features+seq+labels", () -> {
            DataFrameDataset ds = df.head(nBuild).toDataset()
                .features("user_id", "item_id", "likes_level", "views_level")
                .sequenceFeature("item_seq")
                .labels("label")
                .labelsAsLong(true)
                .build();
            holder[0] = ds;
            checkEq("size", nBuild, ds.size());
            checkEq("scalars", 4, ds.scalarFeatureNames().length);
            checkEq("seqs", 1, ds.sequenceFeatureNames().length);
        });
        if (holder[0] == null) {
            System.out.println("ABORT: dataset build failed");
            System.exit(1);
        }
        DataFrameDataset ds = holder[0];

        // ========== 2. API surface / inheritance ==========
        System.out.println("\n== 2. API surface (instanceof) ==");
        benchmark("asDataset / asJavaTensorDataset types", () -> {
            DataFrameNativeDataset nds = ds.asDataset();
            check("instanceof Dataset", nds instanceof Dataset);
            check("source same size", nds.source().size() == ds.size());
            checkEq("native size()", ds.sizeLong(), nds.size().get());

            DataFrameJavaTensorDataset jtds = ds.asJavaTensorDataset();
            check("instanceof JavaTensorDataset", jtds instanceof JavaTensorDataset);
            checkEq("jtds size()", ds.sizeLong(), jtds.size().get());

            DataFrameDataLoader javaLoader = ds.dataloader().batchSize(64).shuffle(false).build();
            check("fromNativeDataset symmetry",
                DataFrameDataLoader.fromNativeDataset(nds, 64, false).numBatches()
                    == javaLoader.numBatches());
        });

        // ========== 3. Sample parity ==========
        System.out.println("\n== 3. Sample parity ==");
        benchmark("get(i) pure-Java vs asDataset", () -> {
            DataFrameNativeDataset nds = ds.asDataset();
            for (int i : new int[]{0, 1, Math.min(7, ds.size() - 1), ds.size() - 1}) {
                DataFrameDataset.Sample s = ds.get(i);
                Example ex = nds.get(i);
                Tensor javaData = s.data();
                Tensor nativeData = ex.data();
                check("data numel i=" + i, javaData.numel() == nativeData.numel());
                check("data close i=" + i, tensorsClose(javaData, nativeData, 1e-5));
                Tensor javaY = s.labels();
                Tensor nativeY = ex.target();
                if (javaY != null) {
                    check("target numel i=" + i, javaY.numel() == nativeY.numel());
                    check("target close i=" + i, tensorsClose(javaY, nativeY, 1e-5));
                }
                // Sample.toExample path
                Example ex2 = s.toExample();
                check("toExample data close i=" + i, tensorsClose(ex2.data(), nativeData, 1e-5));
            }
        });

        // ========== 4. Batch parity (shuffle=false) ==========
        System.out.println("\n== 4. Batch parity (sequential) ==");
        benchmark("Java loader vs native sequential stacked", () -> {
            int B = 64;
            DataFrameDataLoader jLoader = ds.dataloader()
                .batchSize(B).shuffle(false).dropLast(false).build();
            SequentialDataLoader nLoader = ds.nativeDataLoader()
                .batchSize(B).shuffle(false).dropLast(false).buildSequential();

            Iterator<DataFrameDataLoader.Batch> jIt = jLoader.iterator();
            int batches = 0;
            int rows = 0;
            for (ExampleVectorIterator it = nLoader.begin();
                 !it.equals(nLoader.end());
                 it = it.increment()) {
                check("java hasNext batch " + batches, jIt.hasNext());
                DataFrameDataLoader.Batch jb = jIt.next();
                ExampleVector nv = it.access();
                Example stacked = NativeBatchSupport.stack(nv);
                checkEq("batch size " + batches, jb.size(), NativeBatchSupport.batchSize(nv));
                check("data shape0 " + batches, stacked.data().size(0) == jb.size());
                check("features close " + batches,
                    tensorsClose(jb.features(), stacked.data(), 1e-4));
                if (jb.labels() != null) {
                    Tensor nt = stacked.target();
                    // labels may be [B] vs [B,1]
                    check("labels close " + batches,
                        tensorsClose(jb.labels().reshape(new long[]{-1}),
                            nt.reshape(new long[]{-1}), 1e-4));
                }
                rows += jb.size();
                batches++;
            }
            check("java exhausted", !jIt.hasNext());
            checkEq("all rows", ds.size(), rows);
            checkEq("numBatches", jLoader.numBatches(), batches);
            System.out.println("    batches=" + batches + " rows=" + rows);
        });

        // ========== 5. Feature modes ==========
        System.out.println("\n== 5. Feature modes ==");
        benchmark("FIRST_SEQUENCE / PRIMARY / STACKED_SCALARS", () -> {
            NativeViewOptions seqOpts = NativeViewOptions.defaults()
                .mode(NativeViewOptions.Mode.FIRST_SEQUENCE);
            DataFrameNativeDataset seqDs = ds.asDataset(seqOpts);
            Example e0 = seqDs.get(0);
            checkEq("seq primary dim", 64, e0.data().numel());

            NativeViewOptions prim = NativeViewOptions.defaults()
                .primaryFeature("item_seq");
            Example e1 = ds.asDataset(prim).get(0);
            checkEq("PRIMARY item_seq", 64, e1.data().numel());
            check("PRIMARY matches FIRST_SEQUENCE",
                tensorsClose(e0.data(), e1.data(), 1e-5));

            NativeViewOptions sc = NativeViewOptions.defaults()
                .mode(NativeViewOptions.Mode.STACKED_SCALARS);
            Example e2 = ds.asDataset(sc).get(0);
            checkEq("stacked scalars n_feat", 4, e2.data().numel());
        });

        // ========== 6. Conversion round-trip ==========
        System.out.println("\n== 6. Conversion round-trip ==");
        benchmark("toTensorDataset / fromTensorDataset", () -> {
            TensorDataset tds = ds.toTensorDataset();
            checkEq("tds size", ds.sizeLong(), tds.size().get());
            DataFrameDataset back = DataFrameDataset.fromTensorDataset(tds, "f");
            checkEq("back size", ds.size(), back.size());
            // first row close (float features)
            Tensor a = ds.featuresTensor().select(0, 0);
            Tensor b = back.featuresTensor().select(0, 0);
            check("row0 close", tensorsClose(a, b, 1e-4));
        });

        benchmark("toFeatureLabelTensors / fromFeatureLabelTensors", () -> {
            Tensor[] xy = ds.toFeatureLabelTensors();
            DataFrameDataset back = DataFrameDataset.fromFeatureLabelTensors(
                xy[0], xy[1],
                ds.scalarFeatureNames().length > 0 ? ds.scalarFeatureNames() : null,
                ds.labelNames());
            checkEq("roundtrip size", ds.size(), back.size());
            check("X close", tensorsClose(xy[0], back.featuresTensor(), 1e-4));
            check("Y close", tensorsClose(
                xy[1].reshape(new long[]{-1}),
                back.labelsTensor().reshape(new long[]{-1}), 1e-4));
        });

        benchmark("fromNativeDataset materialize", () -> {
            DataFrameNativeDataset nds = ds.asDataset();
            // materialize a small slice via head dataset
            DataFrameDataset small = df.head(Math.min(128, df.rowCount())).toDataset()
                .features("user_id", "item_id", "likes_level", "views_level")
                .labels("label")
                .labelsAsLong(true)
                .build();
            DataFrameNativeDataset nsmall = small.asDataset();
            DataFrameDataset back = DataFrameDataset.fromNativeDataset(nsmall,
                small.scalarFeatureNames(), small.labelNames());
            checkEq("fromNative size", small.size(), back.size());
            check("fromNative X close",
                tensorsClose(small.featuresTensor(), back.featuresTensor(), 1e-3));
        });

        benchmark("toTensorDatasetWithLabels", () -> {
            TensorDataset tds = ds.toTensorDatasetWithLabels();
            checkEq("withLabels size", ds.sizeLong(), tds.size().get());
            // F+L columns
            Tensor t = tds.tensor();
            check("combined cols >= 4", t.dim() == 2 && t.size(1) >= 4);
        });

        // ========== 7. Throughput ==========
        System.out.println("\n== 7. Throughput ==");
        int[] Ns = df.rowCount() >= 10000
            ? new int[]{1024, 10000}
            : new int[]{Math.min(512, ds.size()), ds.size()};
        int[] Bs = new int[]{32, 256};
        for (int N : Ns) {
            DataFrameDataset dsn = df.head(N).toDataset()
                .features("user_id", "item_id", "likes_level", "views_level")
                .sequenceFeature("item_seq")
                .labels("label")
                .labelsAsLong(true)
                .build();
            for (int B : Bs) {
                // pure Java
                {
                    long t0 = System.nanoTime();
                    DataFrameDataLoader loader = dsn.dataloader()
                        .batchSize(B).shuffle(false).build();
                    int rows = 0;
                    for (DataFrameDataLoader.Batch b : loader) {
                        rows += b.size();
                        // touch tensors
                        b.features().numel();
                        if (b.labels() != null) b.labels().numel();
                    }
                    long ms = Math.max(1, (System.nanoTime() - t0) / 1_000_000L);
                    double rps = rows * 1000.0 / ms;
                    checkEq("java rows N=" + N + " B=" + B, N, rows);
                    row("java-DataFrameDataLoader", N, B, ms, rps, "ok");
                    System.out.printf(Locale.US,
                        "  java     N=%d B=%d  %d ms  %.0f rows/s%n", N, B, ms, rps);
                }
                // native sequential
                {
                    long t0 = System.nanoTime();
                    SequentialDataLoader loader = dsn.nativeDataLoader()
                        .batchSize(B).shuffle(false).workers(0).buildSequential();
                    int rows = 0;
                    for (ExampleVectorIterator it = loader.begin();
                         !it.equals(loader.end());
                         it = it.increment()) {
                        Example stacked = NativeBatchSupport.stack(it.access());
                        rows += (int) stacked.data().size(0);
                        stacked.data().numel();
                        stacked.target().numel();
                    }
                    long ms = Math.max(1, (System.nanoTime() - t0) / 1_000_000L);
                    double rps = rows * 1000.0 / ms;
                    checkEq("native rows N=" + N + " B=" + B, N, rows);
                    row("native-SequentialDataLoader", N, B, ms, rps, "ok");
                    System.out.printf(Locale.US,
                        "  native   N=%d B=%d  %d ms  %.0f rows/s%n", N, B, ms, rps);
                }
                // JavaTensorDataset sequential (features only)
                {
                    long t0 = System.nanoTime();
                    SequentialTensorDataLoader loader = dsn.nativeTensorDataLoader()
                        .batchSize(B).shuffle(false).workers(0).buildSequential();
                    int rows = 0;
                    for (TensorExampleVectorIterator it = loader.begin();
                         !it.equals(loader.end());
                         it = it.increment()) {
                        TensorExampleVector v = it.access();
                        Tensor data = NativeBatchSupport.stackData(v);
                        rows += (int) data.size(0);
                        data.numel();
                    }
                    long ms = Math.max(1, (System.nanoTime() - t0) / 1_000_000L);
                    double rps = rows * 1000.0 / ms;
                    checkEq("tensor-loader rows N=" + N + " B=" + B, N, rows);
                    row("native-SequentialTensorDataLoader", N, B, ms, rps, "ok");
                    System.out.printf(Locale.US,
                        "  tensor   N=%d B=%d  %d ms  %.0f rows/s%n", N, B, ms, rps);
                }
            }
        }

        // random native path (coverage only, not parity)
        benchmark("native RandomDataLoader iterate", () -> {
            RandomDataLoader loader = ds.nativeDataLoader()
                .batchSize(128).shuffle(true).dropLast(false).workers(0).buildRandom();
            int rows = 0;
            int batches = 0;
            for (ExampleVectorIterator it = loader.begin();
                 !it.equals(loader.end());
                 it = it.increment()) {
                Example stacked = NativeBatchSupport.stack(it.access());
                rows += (int) stacked.data().size(0);
                batches++;
            }
            check("random covered all rows", rows == ds.size());
            check("random batches > 0", batches > 0);
            System.out.println("    random batches=" + batches + " rows=" + rows);
        });

        // DataFrameDataLoader → native conversion helpers
        benchmark("DataFrameDataLoader.toSequentialDataLoader", () -> {
            DataFrameDataLoader j = ds.dataloader().batchSize(100).shuffle(false).build();
            SequentialDataLoader n = j.toSequentialDataLoader();
            int rows = 0;
            for (ExampleVectorIterator it = n.begin(); !it.equals(n.end()); it = it.increment()) {
                rows += NativeBatchSupport.batchSize(it.access());
            }
            checkEq("toSequential rows", ds.size(), rows);
        });

        // ========== 8. Training smoke ==========
        // Use only low-magnitude features so a tiny Linear does not explode to NaN.
        System.out.println("\n== 8. Training smoke ==");
        benchmark("tiny Linear — pure Java loader", () -> {
            DataFrameDataset small = df.head(Math.min(512, df.rowCount())).toDataset()
                .features("likes_level", "views_level")
                .labels("label")
                .labelsAsLong(false)
                .build();
            float loss = trainOneEpochJava(small, 64, 1e-3);
            check("java loss finite", Float.isFinite(loss));
            System.out.println("    java loss=" + loss);
        });

        benchmark("tiny Linear — native SequentialDataLoader", () -> {
            DataFrameDataset small = df.head(Math.min(512, df.rowCount())).toDataset()
                .features("likes_level", "views_level")
                .labels("label")
                .labelsAsLong(false)
                .build();
            float loss = trainOneEpochNative(small, 64, 1e-3);
            check("native loss finite", Float.isFinite(loss));
            System.out.println("    native loss=" + loss);
        });

        // ========== 9. get_batch direct ==========
        System.out.println("\n== 9. get_batch efficiency path ==");
        benchmark("native get_batch via indices", () -> {
            DataFrameNativeDataset nds = ds.asDataset();
            ExampleVector batch = nds.get_batch(0L, 1L, 2L, 3L);
            checkEq("get_batch size 4", 4, batch.size());
            Example stacked = NativeBatchSupport.stack(batch);
            checkEq("stacked B", 4, stacked.data().size(0));
            checkEq("stacked F", 4, stacked.data().size(1));
        });

        // ========== summary ==========
        System.out.println("\n== Summary table ==");
        System.out.printf(Locale.US, "%-36s %6s %5s %8s %12s %6s%n",
            "path", "N", "B", "ms", "rows/s", "status");
        for (String[] r : table) {
            System.out.printf(Locale.US, "%-36s %6s %5s %8s %12s %6s%n",
                r[0], r[1], r[2], r[3], r[4], r[5]);
        }
        System.out.println("\npassed=" + passed + " failed=" + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL CHECKS PASSED");
    }

    // ---- helpers ------------------------------------------------------------

    static DataFrame loadOrSynthetic() throws Exception {
        String env = System.getenv("MICROLENS_DIR");
        Path dir = Path.of(env != null ? env : DEFAULT_DIR);
        Path valid = dir.resolve("valid.parquet");
        if (Files.isRegularFile(valid)) {
            System.out.println("loading " + valid);
            return DataFrame.readParquet(valid.toString());
        }
        System.out.println("parquet not found; using synthetic DataFrame");
        return synthetic(2048, 64);
    }

    static DataFrame synthetic(int n, int seqLen) {
        DataFrame df = DataFrame.create();
        df.addColumn("user_id", Column.DType.INT64);
        df.addColumn("item_id", Column.DType.INT64);
        df.addColumn("likes_level", Column.DType.FLOAT32);
        df.addColumn("views_level", Column.DType.FLOAT32);
        df.addColumn("item_seq", Column.DType.LIST);
        df.addColumn("label", Column.DType.INT64);
        Random rng = new Random(42);
        for (int i = 0; i < n; i++) {
            int ri = df.addRow();
            df.set(ri, "user_id", (long) (rng.nextInt(10000)));
            df.set(ri, "item_id", (long) (rng.nextInt(50000)));
            df.set(ri, "likes_level", rng.nextFloat());
            df.set(ri, "views_level", rng.nextFloat() * 10f);
            long[] seq = new long[seqLen];
            for (int j = 0; j < seqLen; j++) seq[j] = rng.nextInt(100000);
            df.set(ri, "item_seq", seq);
            df.set(ri, "label", (long) (rng.nextInt(2)));
        }
        return df;
    }

    static boolean tensorsClose(Tensor a, Tensor b, double atol) {
        if (a == null || b == null) return a == b;
        if (a.numel() != b.numel()) return false;
        Tensor af = a.contiguous().cpu().to(ScalarType.Float).reshape(new long[]{-1});
        Tensor bf = b.contiguous().cpu().to(ScalarType.Float).reshape(new long[]{-1});
        Tensor diff = af.sub(bf).abs();
        return diff.max().item_float() <= atol;
    }

    static float trainOneEpochJava(DataFrameDataset ds, int batchSize, double lr) {
        long in = ds.scalarFeatureNames().length;
        LinearImpl lin = new LinearImpl(in, 1);
        SGD opt = new SGD(lin.parameters(), new SGDOptions(lr));
        DataFrameDataLoader loader = ds.dataloader()
            .batchSize(batchSize).shuffle(true).seed(0L).build();
        float last = Float.NaN;
        for (DataFrameDataLoader.Batch b : loader) {
            Tensor x = b.features();
            Tensor y = b.labels().to(ScalarType.Float).reshape(new long[]{-1, 1});
            opt.zero_grad();
            Tensor pred = lin.forward(x);
            Tensor loss = mse_loss(pred, y);
            loss.backward();
            opt.step();
            last = loss.item_float();
        }
        return last;
    }

    static float trainOneEpochNative(DataFrameDataset ds, int batchSize, double lr) {
        long in = ds.scalarFeatureNames().length;
        LinearImpl lin = new LinearImpl(in, 1);
        SGD opt = new SGD(lin.parameters(), new SGDOptions(lr));
        SequentialDataLoader loader = ds.nativeDataLoader()
            .batchSize(batchSize).shuffle(false).workers(0).buildSequential();
        float last = Float.NaN;
        for (ExampleVectorIterator it = loader.begin();
             !it.equals(loader.end());
             it = it.increment()) {
            Example batch = NativeBatchSupport.stack(it.access());
            Tensor x = batch.data();
            Tensor y = batch.target().to(ScalarType.Float);
            if (y.dim() == 1) y = y.unsqueeze(1);
            opt.zero_grad();
            Tensor pred = lin.forward(x);
            Tensor loss = mse_loss(pred, y);
            loss.backward();
            opt.step();
            last = loss.item_float();
        }
        return last;
    }
}
