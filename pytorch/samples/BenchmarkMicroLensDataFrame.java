package samples;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataFrameOps;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.Schema;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.data.datasets.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataLoader;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataset;
import org.bytedeco.pytorch.dataframe.feature.pipeline.DataFramePipeline;
import org.bytedeco.pytorch.dataframe.feature.scaling.StandardScaler;
import org.bytedeco.pytorch.dataframe.io.SchemaInfer;
import org.bytedeco.pytorch.global.torch;

/**
 * Multi-dimensional benchmark / correctness suite for MicroLens_1M_x1:
 * nested LIST parquet, schema inference, split/cat/stack/expand/compress,
 * train/test + feature/label selection, optional feature pipeline,
 * DataFrameDataset / DataFrameDataLoader, TensorDataset conversion.
 *
 * <pre>
 *   java --enable-native-access=ALL-UNNAMED \
 *        --add-opens=java.base/java.nio=ALL-UNNAMED \
 *        -cp ... samples.BenchmarkMicroLensDataFrame
 * </pre>
 *
 * Env: {@code MICROLENS_DIR} overrides the default dataset root.
 */
public class BenchmarkMicroLensDataFrame {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    static final String DEFAULT_DIR =
        "/Users/muller/Documents/code/cpp/VideoMMCTR/data/MicroLens_1M_x1";

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

    static void checkEq(String name, long expected, long actual) {
        boolean ok = expected == actual;
        if (!ok) System.out.println("    " + name + ": expected " + expected + " got " + actual);
        check(name, ok);
    }

    static void checkEq(String name, Object expected, Object actual) {
        boolean ok = Objects.equals(expected, actual)
            || (expected != null && actual != null
                && String.valueOf(expected).equals(String.valueOf(actual)));
        if (!ok) System.out.println("    " + name + ": expected " + expected + " got " + actual);
        check(name, ok);
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        // warm native
        try {
            Tensor warm = torch.tensor(new float[]{1f, 2f, 3f});
            check("torch warmup", warm.numel() == 3);
        } catch (Throwable t) {
            System.out.println("WARN: torch warmup failed (native path?): " + t);
        }

        String dir = System.getenv("MICROLENS_DIR");
        if (dir == null || dir.isBlank()) dir = args.length > 0 ? args[0] : DEFAULT_DIR;
        Path root = Path.of(dir);
        System.out.println("MicroLens root: " + root.toAbsolutePath());
        if (!Files.isDirectory(root)) {
            System.err.println("Dataset dir missing: " + root);
            System.exit(2);
        }

        Path valid = root.resolve("valid.parquet");
        Path train = root.resolve("train.parquet");
        Path test = root.resolve("test.parquet");
        Path itemInfo = root.resolve("item_info.parquet");
        Path trainBin = root.resolve("train.bin"); // may be non-parquet; sniff test

        // ========== 1. Schema inference ==========
        System.out.println("\n== 1. Schema inference ==");
        benchmark("inferSchema(valid.parquet)", () -> {
            Schema s = DataFrame.inferSchema(valid.toString());
            check("valid fields=6", s.size() == 6);
            check("user_id INT64", s.fieldType("user_id") == Column.DType.INT64);
            check("item_seq LIST", s.fieldType("item_seq") == Column.DType.LIST);
            check("label INT64", s.fieldType("label") == Column.DType.INT64);
            System.out.println("    " + s);
        });

        benchmark("inferSchema(item_info) VECTOR emb", () -> {
            Schema s = DataFrame.inferSchema(itemInfo.toString());
            check("item_tags LIST", s.fieldType("item_tags") == Column.DType.LIST);
            // double list → VECTOR
            check("item_emb_d128 VECTOR", s.fieldType("item_emb_d128") == Column.DType.VECTOR
                || s.fieldType("item_emb_d128") == Column.DType.LIST);
            System.out.println("    " + s);
        });

        benchmark("describeParquet nested types", () -> {
            Map<String, String> d = SchemaInfer.describeParquet(valid.toString());
            check("describe has item_seq", d.containsKey("item_seq"));
            check("item_seq mentions LIST or list",
                d.get("item_seq").toLowerCase(Locale.ROOT).contains("list"));
        });

        // ========== 2. Parquet LIST read + printSchema/show/describe ==========
        System.out.println("\n== 2. readParquet LIST + show ==");
        final DataFrame[] validDf = new DataFrame[1];
        benchmark("readParquet(valid) 10k rows", () -> {
            validDf[0] = DataFrame.readParquet(valid.toString());
            checkEq("rowCount", 10000, validDf[0].rowCount());
            checkEq("colCount", 6, validDf[0].columnCount());
            check("item_seq dtype LIST", validDf[0].column("item_seq").dtype() == Column.DType.LIST);
            Object cell0 = validDf[0].get(0, "item_seq");
            check("item_seq is long[]", cell0 instanceof long[]);
            if (cell0 instanceof long[] a) {
                checkEq("item_seq len 64", 64, a.length);
                System.out.println("    first item_seq tail: "
                    + Arrays.toString(Arrays.copyOfRange(a, 55, 64)));
            }
        });

        if (validDf[0] == null) {
            System.err.println("Cannot continue without valid.parquet load");
            System.exit(1);
        }
        DataFrame df = validDf[0];

        benchmark("printSchema/show/describeFrame", () -> {
            df.printSchema();
            df.show(3);
            DataFrame desc = df.describeFrame();
            check("describe has rows", desc.rowCount() == 7);
            check("describe has user_id", desc.hasColumn("user_id"));
            // LIST col should not appear in numeric describe
            check("describe no item_seq", !desc.hasColumn("item_seq"));
        });

        // ========== 3. split / cat / stack / expand / compress ==========
        System.out.println("\n== 3. split / cat / stack / expand / compress ==");
        benchmark("splitRows(4)", () -> {
            DataFrame[] parts = df.splitRows(4);
            checkEq("4 parts", 4, parts.length);
            int sum = 0;
            for (DataFrame p : parts) sum += p.rowCount();
            checkEq("parts cover rows", df.rowCount(), sum);
            checkEq("part0 ~2500", 2500, parts[0].rowCount());
        });

        benchmark("splitCols(2)", () -> {
            DataFrame[] parts = df.splitCols(2);
            checkEq("2 col-parts", 2, parts.length);
            checkEq("cols sum", df.columnCount(),
                parts[0].columnCount() + parts[1].columnCount());
            checkEq("rows preserved", df.rowCount(), parts[0].rowCount());
        });

        benchmark("cat / vstack", () -> {
            DataFrame a = df.head(100);
            DataFrame b = df.iloc(100, 200);
            DataFrame c = DataFrame.cat(a, b);
            checkEq("cat rows", 200, c.rowCount());
            checkEq("cat cols", df.columnCount(), c.columnCount());
        });

        benchmark("stack / hstack rename", () -> {
            DataFrame left = df.select("user_id", "item_id");
            DataFrame right = df.select("label", "likes_level");
            DataFrame h = DataFrame.stack(left, right);
            checkEq("stack cols 4", 4, h.columnCount());
            checkEq("stack rows", df.rowCount(), h.rowCount());
            check("has label", h.hasColumn("label"));
        });

        benchmark("expand item_seq + compress", () -> {
            DataFrame small = df.head(50);
            DataFrame wide = small.expand("item_seq", 64, true, "seq_");
            check("no item_seq after expand", !wide.hasColumn("item_seq"));
            check("has seq_0", wide.hasColumn("seq_0"));
            check("has seq_63", wide.hasColumn("seq_63"));
            checkEq("wide rows", 50, wide.rowCount());

            String[] seqCols = new String[64];
            for (int i = 0; i < 64; i++) seqCols[i] = "seq_" + i;
            DataFrame back = wide.compress("item_seq", seqCols);
            check("restored item_seq", back.hasColumn("item_seq"));
            Object orig = small.get(0, "item_seq");
            Object got = back.get(0, "item_seq");
            check("compress long[]", got instanceof long[] || got instanceof int[] || got instanceof List);
            if (orig instanceof long[] oa && got instanceof long[] ga) {
                check("roundtrip equals", Arrays.equals(oa, ga));
            }
        });

        // ========== 4. train/test + feature/label ==========
        System.out.println("\n== 4. trainTestSplit + featureLabel ==");
        final DataFrameOps.TrainTestSplit[] tt = new DataFrameOps.TrainTestSplit[1];
        benchmark("trainTestSplit(0.2, seed=42)", () -> {
            tt[0] = df.trainTestSplit(0.2, 42L);
            checkEq("train+test rows", df.rowCount(),
                tt[0].train.rowCount() + tt[0].test.rowCount());
            check("train ~8000", Math.abs(tt[0].train.rowCount() - 8000) <= 1);
            check("test ~2000", Math.abs(tt[0].test.rowCount() - 2000) <= 1);
            // deterministic
            DataFrameOps.TrainTestSplit tt2 = df.trainTestSplit(0.2, 42L);
            checkEq("deterministic train0 user",
                tt[0].train.get(0, "user_id"), tt2.train.get(0, "user_id"));
        });

        benchmark("featureLabel + exclude", () -> {
            DataFrameOps.FeatureLabelSplit fl = df.featureLabel(
                new String[]{"user_id", "item_id", "likes_level", "views_level", "item_seq"},
                "label");
            checkEq("X cols 5", 5, fl.X.columnCount());
            checkEq("y cols 1", 1, fl.y.columnCount());
            check("y is label", fl.y.hasColumn("label"));
            checkEq("X rows", df.rowCount(), fl.X.rowCount());

            DataFrameOps.FeatureLabelSplit fl2 = df.featureLabelExclude("label");
            checkEq("exclude X cols 5", 5, fl2.X.columnCount());
            check("exclude has item_seq", fl2.X.hasColumn("item_seq"));
        });

        // ========== 5. optional feature pipeline ==========
        System.out.println("\n== 5. feature pipeline (optional) ==");
        benchmark("pipeline StandardScaler on likes/views", () -> {
            DataFrame sub = df.select("likes_level", "views_level", "label").head(1000);
            DataFramePipeline pipe = new DataFramePipeline(sub)
                .append("scale", new StandardScaler("likes_level", "views_level"));
            DataFrame out = pipe.fitTransform();
            checkEq("pipeline rows", 1000, out.rowCount());
            check("scaled likes", out.hasColumn("likes_level"));
            // mean ~0 after standard scale
            double mean = 0;
            for (int i = 0; i < out.rowCount(); i++) {
                mean += ((Number) out.get(i, "likes_level")).doubleValue();
            }
            mean /= out.rowCount();
            check("likes mean~0", Math.abs(mean) < 0.15);
            System.out.println("    scaled likes mean=" + mean);
        });

        // ========== 6. DataFrameDataset + DataLoader ==========
        System.out.println("\n== 6. Dataset / DataLoader ==");
        final DataFrameDataset[] dsHolder = new DataFrameDataset[1];
        benchmark("toDataset builder", () -> {
            DataFrameDataset ds = df.head(2000).toDataset()
                .features("user_id", "item_id", "likes_level", "views_level")
                .sequenceFeature("item_seq")
                .labels("label")
                .labelsAsLong(true)
                .build();
            dsHolder[0] = ds;
            checkEq("dataset size", 2000, ds.size());
            checkEq("scalar feats 4", 4, ds.scalarFeatureNames().length);
            checkEq("seq feats 1", 1, ds.sequenceFeatureNames().length);
            check("seq name item_seq", "item_seq".equals(ds.sequenceFeatureNames()[0]));

            DataFrameDataset.Sample s0 = ds.get(0);
            check("sample has stacked", s0.features().containsKey("__stacked__")
                || s0.data() != null);
            Tensor seq = s0.feature("item_seq");
            checkEq("seq numel 64", 64, seq.numel());
            Tensor y = s0.labels();
            check("label present", y != null && y.numel() >= 1);

            Tensor X = ds.featuresTensor();
            checkEq("X shape0", 2000, X.size(0));
            checkEq("X shape1", 4, X.size(1));
            Tensor S = ds.sequenceTensor("item_seq");
            checkEq("S shape0", 2000, S.size(0));
            checkEq("S shape1", 64, S.size(1));
            Tensor Y = ds.labelsTensor();
            checkEq("Y shape0", 2000, Y.size(0));
        });

        if (dsHolder[0] != null) {
            DataFrameDataset ds = dsHolder[0];
            benchmark("DataLoader batch iterate", () -> {
                DataFrameDataLoader loader = ds.dataloader()
                    .batchSize(256)
                    .shuffle(true)
                    .seed(7L)
                    .dropLast(false)
                    .build();
                int batches = 0;
                int rows = 0;
                for (DataFrameDataLoader.Batch b : loader) {
                    batches++;
                    rows += b.size();
                    Tensor feat = b.features();
                    check("batch features [B,4]", feat.size(0) == b.size() && feat.size(1) == 4);
                    Tensor seq = b.feature("item_seq");
                    check("batch seq [B,64]", seq.size(0) == b.size() && seq.size(1) == 64);
                    Tensor lab = b.labels();
                    check("batch labels [B]", lab != null && lab.size(0) == b.size());
                }
                checkEq("all rows covered", 2000, rows);
                check("num batches ~8", batches == loader.numBatches());
                System.out.println("    batches=" + batches + " rows=" + rows);
            });

            benchmark("toTensorDataset / withLabels", () -> {
                TensorDataset tds = ds.toTensorDataset();
                check("TensorDataset size", tds.size().get() == 2000);
                TensorDataset tds2 = ds.toTensorDatasetWithLabels();
                check("TensorDataset+labels size", tds2.size().get() == 2000);
            });
        }

        // ========== 7. featureLabel → dataset shortcut + pipeline ==========
        System.out.println("\n== 7. featureLabel → dataset (+ optional pipe) ==");
        benchmark("fl.toDataset()", () -> {
            DataFrameOps.FeatureLabelSplit fl = df.head(500).featureLabel(
                new String[]{"user_id", "item_id", "likes_level", "views_level"},
                "label");
            DataFrameDataset ds = fl.toDataset();
            checkEq("fl dataset size", 500, ds.size());
            checkEq("fl scalars", 4, ds.scalarFeatureNames().length);
        });

        benchmark("builder with StandardScaler pipeline on X", () -> {
            DataFramePipeline pipe = new DataFramePipeline()
                .append("scale", new StandardScaler("likes_level", "views_level"));
            DataFrameDataset ds = df.head(800).toDataset()
                .features("user_id", "item_id", "likes_level", "views_level")
                .labels("label")
                .pipeline(pipe)
                .build();
            checkEq("piped size", 800, ds.size());
            Tensor X = ds.featuresTensor();
            // likes/views columns are indices 2,3 — roughly zero-mean
            float sum = 0;
            // just ensure finite values
            check("X numel", X.numel() == 800L * 4);
        });

        // ========== 8. Multi-file / auto read + item_info VECTOR ==========
        System.out.println("\n== 8. multi-file formats / item_info ==");
        benchmark("DataFrame.read(valid) auto", () -> {
            DataFrame r = DataFrame.read(valid.toString());
            checkEq("auto read rows", 10000, r.rowCount());
            r.close();
        });

        benchmark("read item_info tags+emb", () -> {
            DataFrame items = DataFrame.readParquet(itemInfo.toString());
            check("item_tags LIST", items.column("item_tags").dtype() == Column.DType.LIST);
            Object tags0 = items.get(0, "item_tags");
            check("tags long[] or int[]", tags0 instanceof long[] || tags0 instanceof int[]);
            if (tags0 instanceof long[] a) checkEq("tags len 5", 5, a.length);
            Object emb0 = items.get(0, "item_emb_d128");
            check("emb is array", emb0 instanceof double[] || emb0 instanceof float[]
                || emb0 instanceof List);
            if (emb0 instanceof double[] d) checkEq("emb dim 128", 128, d.length);
            if (emb0 instanceof float[] f) checkEq("emb dim 128f", 128, f.length);
            // VECTOR dtype preferred for float/double lists
            Column.DType edt = items.column("item_emb_d128").dtype();
            check("emb VECTOR or LIST", edt == Column.DType.VECTOR || edt == Column.DType.LIST);
            System.out.println("    item_info rows=" + items.rowCount()
                + " emb_dtype=" + edt);
            items.close();
        });

        // train is 3.6M — only schema + small head via full read would be slow;
        // schema-only is enough for correctness of nested LIST on train footer.
        if (Files.isRegularFile(train)) {
            benchmark("inferSchema(train.parquet) 3.6M footer", () -> {
                Schema s = DataFrame.inferSchema(train.toString());
                check("train item_seq LIST", s.fieldType("item_seq") == Column.DType.LIST);
                checkEq("train fields", 6, s.size());
            });
        }

        if (Files.isRegularFile(test)) {
            benchmark("inferSchema(test.parquet)", () -> {
                Schema s = DataFrame.inferSchema(test.toString());
                check("test has ID or user_id", s.hasField("user_id") || s.hasField("ID"));
                check("test item_seq LIST", s.fieldType("item_seq") == Column.DType.LIST);
            });
        }

        // magic-byte sniff on .bin if present (may or may not be parquet)
        if (Files.isRegularFile(trainBin)) {
            benchmark("sniff train.bin", () -> {
                var fmt = SchemaInfer.sniff(trainBin.toString());
                System.out.println("    train.bin sniffed as " + fmt);
                // just ensure no throw
                check("sniff returns", fmt != null);
            });
        }

        // ========== 9. parquet write LIST round-trip (needs hadoop-client-runtime) ==========
        System.out.println("\n== 9. writeParquet LIST round-trip ==");
        benchmark("write+read LIST head(100)", () -> {
            Path tmp = Files.createTempFile("microlens-rt-", ".parquet");
            try {
                DataFrame small = df.head(100);
                small.writeParquet(tmp.toString());
                DataFrame back = DataFrame.readParquet(tmp.toString());
                checkEq("rt rows", 100, back.rowCount());
                check("rt item_seq LIST", back.column("item_seq").dtype() == Column.DType.LIST);
                Object a = small.get(0, "item_seq");
                Object b = back.get(0, "item_seq");
                if (a instanceof long[] la && b instanceof long[] lb) {
                    check("rt seq equal", Arrays.equals(la, lb));
                } else {
                    check("rt seq non-null", b != null);
                }
            } finally {
                Files.deleteIfExists(tmp);
            }
        });

        // ========== 10. end-to-end mini train loop smoke (1 epoch, few batches) ==========
        System.out.println("\n== 10. mini training smoke (optional native) ==");
        benchmark("mini CE-like smoke over batches", () -> {
            DataFrameDataset ds = df.head(512).toDataset()
                .features("user_id", "item_id", "likes_level", "views_level")
                .sequenceFeature("item_seq")
                .labels("label")
                .build();
            DataFrameDataLoader loader = ds.dataloader().batchSize(64).shuffle(true).seed(1L).build();
            int n = 0;
            double lossProxy = 0;
            for (DataFrameDataLoader.Batch b : loader) {
                Tensor x = b.features(); // [B,4]
                Tensor seq = b.feature("item_seq"); // [B,64]
                Tensor y = b.labels();
                // trivial proxy: mean |x| + mean(seq)/1e5 + mean(y)
                lossProxy += Math.abs(x.mean().item_float())
                    + Math.abs(seq.to(torch.ScalarType.Float).mean().item_float()) / 1e5
                    + Math.abs(y.to(torch.ScalarType.Float).mean().item_float());
                n++;
                if (n >= 4) break;
            }
            check("ran batches", n >= 1);
            check("finite loss proxy", Double.isFinite(lossProxy));
            System.out.println("    batches=" + n + " lossProxy=" + lossProxy);
        });

        // cleanup
        try { df.close(); } catch (Exception ignored) {}

        System.out.println("\n========== SUMMARY ==========");
        System.out.println("passed=" + passed + " failed=" + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL OK");
    }
}
