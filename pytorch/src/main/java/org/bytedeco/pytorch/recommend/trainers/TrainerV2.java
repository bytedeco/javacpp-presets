/*
 * Ported from torchSa: torchrec/train/TrainerV2.scala
 *
 * Trainer V2 — MicroLens / Transformer_DCN benchmark with:
 *   - DataFrame.readParquet for train / valid / item_info
 *   - df.toDataset().features(...).sequenceFeature(...).labels(...).build()
 *   - pure-Java DataFrameDataLoader for named multi-feature batches
 *   - Tqdm progress bars
 *   - per-batch PointerScope so JavaCPP native Tensor wrappers are released
 *   - our recommend Trainer helpers (DeviceSupport, Adam warm-up pattern)
 *
 * Args:
 *   0 dataRoot
 *   1 batchSize        (default 256)
 *   2 numEpochs        (default 1 for smoke; use 10 for full)
 *   3 learningRate     (default 5e-4)
 *   4 maxTrainBatches  (default 0 = all; smoke: 20)
 *   5 maxValidBatches  (default 0 = all)
 *   6 logEvery         (default 10)
 *   7 trainSampleFrac  (default 0.01 = 1% of train; use 1.0 for full)
 *   8 profileFirstN    (default 5)
 *
 * Run example (from pytorch module):
 *   mvn -q exec:java -Dexec.mainClass=\
 *     org.bytedeco.pytorch.utils.recommend.trainers.TrainerV2 \
 *     -Dexec.args="/path/to/MicroLens_1M_x1 128 1 5e-4 20 10 5 0.01 3"
 */
package org.bytedeco.pytorch.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataLoader;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataset;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.metrics.RankingMetrics;
import org.bytedeco.pytorch.recommend.models.ranking.TransformerDCN;
import org.bytedeco.pytorch.plot.tqdm.Tqdm;
import org.bytedeco.pytorch.plot.tqdm.TqdmBar;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public final class TrainerV2 {

    private static final int PRETRAIN_DIM = 128;
    private static final int TAGS_LEN = 5;
    private static final int MAX_SEQ_LEN = 64;

    private TrainerV2() {}

    // ---- item_info tables ----------------------------------------------------

    /** Flattened item tables keyed by row index == item_id (row 0 = padding). */
    public static final class ItemTables {
        public final int numItems;
        public final float[] pretrainedEmbFlat;
        public final long[] itemTagsFlat;

        public ItemTables(int numItems, float[] pretrainedEmbFlat, long[] itemTagsFlat) {
            this.numItems = numItems;
            this.pretrainedEmbFlat = pretrainedEmbFlat;
            this.itemTagsFlat = itemTagsFlat;
        }
    }

    /**
     * Load item_info.parquet via DataFrame and pack:
     *   - item_emb_d128 → float[numItems * 128]
     *   - item_tags     → long [numItems * 5]
     */
    public static ItemTables loadItemTables(String itemPath) throws Exception {
        System.out.print("Loading item_info via DataFrame: " + itemPath + " ... ");
        System.out.flush();
        DataFrame df = DataFrame.readParquet(itemPath);
        int n = df.rowCount();
        System.out.printf("%,d items%n", n);

        float[] embFlat = new float[n * PRETRAIN_DIM];
        long[] tagsFlat = new long[n * TAGS_LEN];

        org.bytedeco.pytorch.dataframe.Column embCol =
                df.hasColumn("item_emb_d128") ? df.column("item_emb_d128") : null;
        org.bytedeco.pytorch.dataframe.Column tagsCol =
                df.hasColumn("item_tags") ? df.column("item_tags") : null;

        for (int i = 0; i < n; i++) {
            if (embCol != null) {
                fillFloatRow(embFlat, i * PRETRAIN_DIM, PRETRAIN_DIM, embCol.get(i));
            }
            if (tagsCol != null) {
                fillLongRow(tagsFlat, i * TAGS_LEN, TAGS_LEN, tagsCol.get(i));
            }
        }
        return new ItemTables(n, embFlat, tagsFlat);
    }

    private static void fillLongRow(long[] dest, int off, int dim, Object v) {
        if (v == null) return;
        if (v instanceof long[]) {
            long[] a = (long[]) v;
            System.arraycopy(a, 0, dest, off, Math.min(dim, a.length));
        } else if (v instanceof int[]) {
            int[] a = (int[]) v;
            for (int k = 0; k < Math.min(dim, a.length); k++) dest[off + k] = a[k];
        } else if (v instanceof double[]) {
            double[] a = (double[]) v;
            for (int k = 0; k < Math.min(dim, a.length); k++) dest[off + k] = (long) a[k];
        } else if (v instanceof float[]) {
            float[] a = (float[]) v;
            for (int k = 0; k < Math.min(dim, a.length); k++) dest[off + k] = (long) a[k];
        } else if (v instanceof List) {
            List<?> list = (List<?>) v;
            int m = Math.min(dim, list.size());
            for (int k = 0; k < m; k++) {
                Object e = list.get(k);
                if (e instanceof Number) dest[off + k] = ((Number) e).longValue();
            }
        } else if (v instanceof Number) {
            dest[off] = ((Number) v).longValue();
        }
    }

    private static void fillFloatRow(float[] dest, int off, int dim, Object v) {
        if (v == null) return;
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            System.arraycopy(a, 0, dest, off, Math.min(dim, a.length));
        } else if (v instanceof double[]) {
            double[] a = (double[]) v;
            for (int k = 0; k < Math.min(dim, a.length); k++) dest[off + k] = (float) a[k];
        } else if (v instanceof long[]) {
            long[] a = (long[]) v;
            for (int k = 0; k < Math.min(dim, a.length); k++) dest[off + k] = (float) a[k];
        } else if (v instanceof int[]) {
            int[] a = (int[]) v;
            for (int k = 0; k < Math.min(dim, a.length); k++) dest[off + k] = (float) a[k];
        } else if (v instanceof List) {
            List<?> list = (List<?>) v;
            int m = Math.min(dim, list.size());
            for (int k = 0; k < m; k++) {
                Object e = list.get(k);
                if (e instanceof Number) dest[off + k] = ((Number) e).floatValue();
            }
        } else if (v instanceof Number) {
            dest[off] = ((Number) v).floatValue();
        }
    }

    // ---- DataFrame → Dataset / DataLoader ------------------------------------

    public static DataFrameDataset buildDataset(
            String path, String name, double sampleFrac, long sampleSeed) throws Exception {
        System.out.print("Loading " + name + " via DataFrame: " + path + " ... ");
        System.out.flush();
        DataFrame raw = DataFrame.readParquet(path);
        int fullN = raw.rowCount();
        DataFrame df;
        if (sampleFrac < 1.0 && sampleFrac > 0.0) {
            DataFrame sampled = raw.sampleFrac(sampleFrac, sampleSeed);
            System.out.printf("%,d rows → sampleFrac=%s → %,d rows%n",
                    fullN, sampleFrac, sampled.rowCount());
            df = sampled;
        } else {
            System.out.printf("%,d rows (full)%n", fullN);
            df = raw;
        }
        df.printSchema();
        df.show(3);

        // labelsAsLong(false) → float labels for BCEWithLogitsLoss
        return df.toDataset()
                .features("user_id", "item_id", "likes_level", "views_level")
                .sequenceFeature("item_seq")
                .labels("label")
                .labelsAsLong(false)
                .build();
    }

    public static DataFrameDataLoader buildLoader(
            DataFrameDataset ds, int batchSize, boolean shuffle, long seed) {
        return ds.dataloader()
                .batchSize(batchSize)
                .shuffle(shuffle)
                .dropLast(false)
                .seed(seed)
                .build();
    }

    // ---- Batch conversion ----------------------------------------------------

    /**
     * Named features from DataFrameDataLoader → tensors for TransformerDCN.forward:
     *   item_seq, item_id, likes_level, views_level, mask, label
     */
    public static final class ModelBatch {
        public final Tensor itemSeq;    // [B,S] long
        public final Tensor itemIds;    // [B] long
        public final Tensor likesLevel; // [B] long
        public final Tensor viewsLevel; // [B] long
        public final Tensor mask;       // [B,S] float
        public final Tensor label;      // [B] float
        public final int batchSize;

        public ModelBatch(Tensor itemSeq, Tensor itemIds, Tensor likesLevel, Tensor viewsLevel,
                          Tensor mask, Tensor label, int batchSize) {
            this.itemSeq = itemSeq;
            this.itemIds = itemIds;
            this.likesLevel = likesLevel;
            this.viewsLevel = viewsLevel;
            this.mask = mask;
            this.label = label;
            this.batchSize = batchSize;
        }

        public ModelBatch to(Device device) {
            ScalarType longT = ScalarType.Long;
            ScalarType floatT = ScalarType.Float;
            return new ModelBatch(
                    itemSeq.to(device, longT),
                    itemIds.to(device, longT),
                    likesLevel.to(device, longT),
                    viewsLevel.to(device, longT),
                    mask.to(device, floatT),
                    label.to(device, floatT),
                    batchSize);
        }
    }

    public static ModelBatch toModelBatch(DataFrameDataLoader.Batch b, Device device) {
        Tensor itemSeq = b.feature("item_seq").to(ScalarType.Long).contiguous();
        Tensor itemIds = b.feature("item_id").to(ScalarType.Long).contiguous();
        Tensor likesLevel = b.feature("likes_level").to(ScalarType.Long).contiguous();
        Tensor viewsLevel = b.feature("views_level").to(ScalarType.Long).contiguous();
        // mask = (item_seq > 0).float()
        Tensor mask = itemSeq.gt(new Scalar(0L)).to(ScalarType.Float).contiguous();
        Tensor labelRaw = b.labels();
        Tensor label;
        if (labelRaw == null) {
            label = torch.zeros(new long[]{b.size()},
                    new TensorOptions().dtype(
                            new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Float)));
        } else {
            label = labelRaw.to(ScalarType.Float).reshape(b.size()).contiguous();
        }
        ModelBatch batch = new ModelBatch(itemSeq, itemIds, likesLevel, viewsLevel,
                mask, label, b.size());
        return batch.to(device);
    }

    // ---- tqdm helpers --------------------------------------------------------

    private static Map<String, Object> postfix(double loss, Double aucOpt, double sps) {
        Map<String, Object> m = new HashMap<>();
        m.put("loss", String.format("%.4f", loss));
        if (aucOpt != null) m.put("auc", String.format("%.3f", aucOpt));
        m.put("sps", String.format("%,d", (int) sps));
        return m;
    }

    private static <T> TqdmBar<T> wrapBar(Iterator<T> it, int total, String desc, String colour) {
        return Tqdm.of(it, total)
                .setDescription(desc)
                .setUnit("batch")
                .colour(colour)
                .setMinInterval(0.2)
                .setAscii(false);
    }

    private static double msSince(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }

    private static String resolveDeviceString() {
        String backend = DeviceSupport.backend();
        System.out.println("  Device       : " + backend);
        return backend;
    }

    private static void reportModelBudget(TransformerDCN model, int batchSize) {
        TensorVector params = model.parameters();
        long nParams = 0L;
        long n = params.size();
        for (long i = 0; i < n; i++) {
            nParams += params.get(i).__dispatch_numel();
        }
        double weightMb = nParams * 4.0 / (1024 * 1024);
        double adamMb = weightMb * 3.0;
        long d = model.dcnInDim();
        long crossParams = 3L * d * d + 3L * d;
        double crossMb = crossParams * 4.0 / (1024 * 1024);
        double actMb = batchSize * d * 4L * 8L / (1024.0 * 1024.0);
        System.out.printf("  Params         : %,d  (~%.0f MB fp32 weights)%n", nParams, weightMb);
        System.out.printf("  Adam state est : ~%.0f MB (weights + m + v)%n", adamMb);
        System.out.printf("  CrossNet alone : d=%d  ~%,d params (~%.0f MB)  act@bs=%d ~%.0f MB%n",
                d, crossParams, crossMb, batchSize, actMb);
        System.out.println("  NOTE           : per-batch PointerScope frees collate tensors;");
        System.out.println("                   RankingMetrics.Accumulator uses primitive arrays.");
    }

    /**
     * Warm up Adam state tensors OUTSIDE any PointerScope.
     * Adam lazily allocates exp_avg / exp_avg_sq on first step(); if that happens
     * inside a per-batch scope, scope.close() frees optimizer state → next step SIGSEGV.
     */
    private static void warmAdamState(
            TransformerDCN model, Adam optimizer, Device device, int batchSize) {
        System.out.print("Warming Adam state (outside PointerScope) ... ");
        System.out.flush();
        long B = Math.min(batchSize, 4);
        long S = MAX_SEQ_LEN;
        TensorOptions longOpts = new TensorOptions()
                .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Long));
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Float));
        Tensor itemSeq = torch.zeros(new long[]{B, S}, longOpts);
        Tensor itemIds = torch.zeros(new long[]{B}, longOpts);
        Tensor likes = torch.zeros(new long[]{B}, longOpts);
        Tensor views = torch.zeros(new long[]{B}, longOpts);
        Tensor mask = torch.ones(new long[]{B, S}, floatOpts);
        Tensor label = torch.zeros(new long[]{B}, floatOpts);
        ModelBatch batch = new ModelBatch(itemSeq, itemIds, likes, views, mask, label, (int) B)
                .to(device);
        model.train(true);
        optimizer.zero_grad();
        Tensor out = model.forward(batch.itemSeq, batch.itemIds, batch.mask,
                batch.likesLevel, batch.viewsLevel);
        Tensor loss = model.computeLoss(batch.label, out);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        System.out.println("done.");
    }

    // ---- main ----------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        String dataRoot = args.length > 0
                ? args[0]
                : "/Users/muller/Documents/code/cpp/VideoMMCTR/data/MicroLens_1M_x1";
        int batchSize = args.length > 1 ? Integer.parseInt(args[1]) : 256;
        int numEpochs = args.length > 2 ? Integer.parseInt(args[2]) : 1;
        double learningRate = args.length > 3 ? Double.parseDouble(args[3]) : 5e-4;
        int maxTrainBatches = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        int maxValidBatches = args.length > 5 ? Integer.parseInt(args[5]) : 0;
        int logEvery = args.length > 6 ? Math.max(1, Integer.parseInt(args[6])) : 10;
        double trainSampleFrac = args.length > 7 ? Double.parseDouble(args[7]) : 0.01;
        int profileFirstN = args.length > 8 ? Integer.parseInt(args[8]) : 5;

        System.out.println("=".repeat(60));
        System.out.println(" Transformer_DCN - MicroLens Training V2 (DataFrame + tqdm)");
        System.out.println("=".repeat(60));
        System.out.println("  Data root    : " + dataRoot);
        System.out.println("  Batch size   : " + batchSize);
        System.out.println("  Epochs       : " + numEpochs);
        System.out.println("  Learning rate: " + learningRate);
        System.out.println("  Max train batches (0=all): " + maxTrainBatches);
        System.out.println("  Max valid batches (0=all): " + maxValidBatches);
        System.out.println("  Log every    : " + logEvery + " batch(es)  (batch-AUC only on log steps)");
        System.out.println("  Train sample : " + (trainSampleFrac * 100) + "% of full train set");
        System.out.println("  Profile first: " + profileFirstN + " train batches (ms breakdown)");

        Loader.load(org.bytedeco.pytorch.presets.torch.class);
        torch.manual_seed(20242025L);
        String deviceStr = resolveDeviceString();
        Device device = new Device(deviceStr);
        System.out.println();

        String trainPath = dataRoot + "/train.parquet";
        String validPath = dataRoot + "/valid.parquet";
        String itemPath = dataRoot + "/item_info.parquet";

        // 1) item tables
        ItemTables itemTables = loadItemTables(itemPath);

        // 2) DataFrame → Dataset → DataLoader
        DataFrameDataset trainDs = buildDataset(trainPath, "train", trainSampleFrac, 20242025L);
        DataFrameDataset validDs = buildDataset(validPath, "valid", 1.0, 0L);

        DataFrameDataLoader trainLoader = buildLoader(trainDs, batchSize, true, 20242025L);
        DataFrameDataLoader validLoader = buildLoader(validDs, batchSize, false, 0L);

        int trainTotalBatches = maxTrainBatches > 0
                ? Math.min(maxTrainBatches, trainLoader.numBatches())
                : trainLoader.numBatches();
        int validTotalBatches = maxValidBatches > 0
                ? Math.min(maxValidBatches, validLoader.numBatches())
                : validLoader.numBatches();

        System.out.printf("  Train samples     : %,d%n", trainDs.size());
        System.out.printf("  Valid samples     : %,d%n", validDs.size());
        System.out.printf("  Train steps/epoch : %,d  (~%,d samples)%n",
                trainTotalBatches, (long) trainTotalBatches * batchSize);
        System.out.printf("  Valid steps       : %,d%n", validTotalBatches);
        System.out.println();

        System.out.print("Building model ... ");
        System.out.flush();
        long nItems = itemTables.numItems;
        TransformerDCN model = new TransformerDCN(
                /*itemVocabSize*/ Math.max(nItems, 91718L),
                /*embDim*/ 64L,
                /*pretrainDim*/ PRETRAIN_DIM,
                /*embDimPretrain*/ 128L,
                /*likesVocabSize*/ 11L,
                /*viewsVocabSize*/ 11L,
                /*tagsVocabSize*/ 11740L,
                /*numItems*/ nItems,
                /*tagsLen*/ TAGS_LEN,
                itemTables.pretrainedEmbFlat,
                itemTables.itemTagsFlat,
                /*numHeads*/ 1L,
                /*transformerLayers*/ 2,
                /*transformerDropout*/ 0.2,
                /*dimFeedforward*/ 256L,
                /*firstKCols*/ 16,
                /*concatMaxPool*/ true,
                /*dcnCrossLayers*/ 3,
                new long[]{1024L, 512L, 256L},
                new long[]{64L, 32L},
                /*netDropout*/ 0.2,
                deviceStr);
        model.to(device, false);
        model.train(true);
        System.out.println("done.");

        model.summary();
        System.out.println("  Expected dcnInDim ~ 9088");
        reportModelBudget(model, batchSize);
        System.out.println();

        // Optimizer OUTSIDE PointerScope
        Adam optimizer = new Adam(model.parameters(), new AdamOptions(learningRate));
        warmAdamState(model, optimizer, device, batchSize);
        System.out.println();

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // ---------- TRAIN ----------
            model.train(true);
            long trainExpected = maxTrainBatches > 0
                    ? (long) maxTrainBatches * batchSize
                    : trainDs.size();
            RankingMetrics.Accumulator trainAcc = new RankingMetrics.Accumulator(
                    (int) Math.min(Integer.MAX_VALUE, Math.max(4096L, trainExpected)));
            int batchCount = 0;
            long samplesDone = 0L;
            long t0 = System.nanoTime();

            Tqdm.write(String.format(
                    "--- Epoch %d/%d TRAIN  batch_size=%d  steps=%,d ---",
                    epoch + 1, numEpochs, batchSize, trainTotalBatches));

            Iterator<DataFrameDataLoader.Batch> trainIt = trainLoader.iterator();
            TqdmBar<DataFrameDataLoader.Batch> trainBar =
                    wrapBar(trainIt, trainTotalBatches, "E" + (epoch + 1) + " train", "green");
            try {
                while (trainBar.hasNext()
                        && (maxTrainBatches <= 0 || batchCount < maxTrainBatches)) {
                    boolean doProfile = batchCount < profileFirstN;
                    long tBatch0 = System.nanoTime();

                    // CRITICAL: collate + model intermediates inside PointerScope
                    PointerScope scope = new PointerScope();
                    double lossVal = 0.0;
                    int actualBs = 0;
                    Double aucOpt = null;
                    try {
                        long tData = System.nanoTime();
                        DataFrameDataLoader.Batch dfBatch = trainBar.next();
                        ModelBatch batch = toModelBatch(dfBatch, device);
                        actualBs = batch.batchSize;
                        double msData = msSince(tData);

                        long tFwd = System.nanoTime();
                        optimizer.zero_grad();
                        Tensor out = model.forward(
                                batch.itemSeq, batch.itemIds, batch.mask,
                                batch.likesLevel, batch.viewsLevel);
                        Tensor loss = model.computeLoss(batch.label, out);
                        if (doProfile) {
                            loss.item_double();
                        }
                        double msFwd = msSince(tFwd);

                        long tBwd = System.nanoTime();
                        loss.backward();
                        if (doProfile) {
                            loss.item_double();
                        }
                        double msBwd = msSince(tBwd);

                        long tStep = System.nanoTime();
                        optimizer.step();
                        lossVal = loss.item_double();
                        double msStep = msSince(tStep);

                        long tMet = System.nanoTime();
                        trainAcc.update(out, batch.label, lossVal, actualBs);

                        batchCount++;
                        samplesDone += actualBs;

                        boolean shouldLog = batchCount == 1
                                || batchCount == trainTotalBatches
                                || batchCount % logEvery == 0;
                        if (shouldLog) {
                            aucOpt = RankingMetrics.fromLogits(out, batch.label).auc;
                        }
                        double msMet = msSince(tMet);

                        if (doProfile) {
                            double msTot = msSince(tBatch0);
                            Tqdm.write(String.format(
                                    "  [profile b%02d] data=%.0fms  fwd=%.0fms  bwd=%.0fms  "
                                            + "step=%.0fms  met=%.0fms  total=%.0fms  bs=%d",
                                    batchCount, msData, msFwd, msBwd, msStep, msMet, msTot, actualBs));
                        }
                    } finally {
                        scope.close();
                    }

                    double elapsed = (System.nanoTime() - t0) / 1e9;
                    double sps = elapsed > 0 ? samplesDone / elapsed : 0.0;
                    trainBar.set_postfix(postfix(lossVal, aucOpt, sps));
                }
            } finally {
                trainBar.close();
            }

            double[] tr = trainAcc.result();
            double trainElapsed = (System.nanoTime() - t0) / 1e9;
            Tqdm.write(String.format(
                    "[E%d] TRAIN done  n=%,d  batches=%,d  logloss=%.5f  auc=%.4f  acc=%.4f  "
                            + "%,d samples/s  %.1fs",
                    epoch + 1, (int) tr[3], batchCount, tr[0], tr[1], tr[2],
                    trainElapsed > 0 ? (int) (tr[3] / trainElapsed) : 0, trainElapsed));
            trainAcc.clear();

            // ---------- VALID ----------
            model.eval();
            long validExpected = maxValidBatches > 0
                    ? (long) maxValidBatches * batchSize
                    : validDs.size();
            RankingMetrics.Accumulator validAcc = new RankingMetrics.Accumulator(
                    (int) Math.min(Integer.MAX_VALUE, Math.max(4096L, validExpected)));
            int vCount = 0;
            long vSamples = 0L;
            long tv0 = System.nanoTime();

            Tqdm.write(String.format(
                    "--- Epoch %d/%d VALID  steps=%,d ---",
                    epoch + 1, numEpochs, validTotalBatches));

            Iterator<DataFrameDataLoader.Batch> validIt = validLoader.iterator();
            TqdmBar<DataFrameDataLoader.Batch> validBar =
                    wrapBar(validIt, validTotalBatches, "E" + (epoch + 1) + " valid", "cyan");
            try {
                while (validBar.hasNext()
                        && (maxValidBatches <= 0 || vCount < maxValidBatches)) {
                    PointerScope scope = new PointerScope();
                    double lossVal = 0.0;
                    int bs = 0;
                    try {
                        DataFrameDataLoader.Batch dfBatch = validBar.next();
                        ModelBatch batch = toModelBatch(dfBatch, device);
                        bs = batch.batchSize;
                        Tensor out = model.forward(
                                batch.itemSeq, batch.itemIds, batch.mask,
                                batch.likesLevel, batch.viewsLevel);
                        Tensor loss = model.computeLoss(batch.label, out);
                        lossVal = loss.item_double();
                        validAcc.update(out, batch.label, lossVal, bs);
                        vCount++;
                        vSamples += bs;
                    } finally {
                        scope.close();
                    }
                    double elapsed = (System.nanoTime() - tv0) / 1e9;
                    double sps = elapsed > 0 ? vSamples / elapsed : 0.0;
                    validBar.set_postfix(postfix(lossVal, null, sps));
                }
            } finally {
                validBar.close();
            }

            double[] vr = validAcc.result();
            if (vr[3] > 0) {
                double validElapsed = (System.nanoTime() - tv0) / 1e9;
                Tqdm.write(String.format(
                        "[E%d] VALID done  n=%,d  batches=%,d  logloss=%.5f  auc=%.4f  acc=%.4f  "
                                + "%,d samples/s  %.1fs",
                        epoch + 1, (int) vr[3], vCount, vr[0], vr[1], vr[2],
                        validElapsed > 0 ? (int) (vr[3] / validElapsed) : 0, validElapsed));
            }
            validAcc.clear();
        }

        System.out.println("Training complete (V2 / DataFrame + tqdm + PointerScope + TransformerDCN).");
    }
}
