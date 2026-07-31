/*
 * Real-data OneRec SFT benchmark: MicroLens
 *   item_emb_d128.npy → RQ-KMeans SIDs → SID sequences → OneRec / OneRecV2 / OpenOneRec NTP
 *
 * Also optional short GRPO post-training pass with constrained beam.
 *
 * Args:
 *   0 dataRoot        (default MicroLens_1M_x1)
 *   1 device          (cpu|mps|cuda, default auto)
 *   2 model           (onerec|onerecv2|openonerec)
 *   3 batchSize       (default 64)
 *   4 maxTrainBatches (default 50; 0=all of sampled set)
 *   5 maxValidBatches (default 20)
 *   6 trainRows       (default 20000 subsample before batching)
 *   7 validRows       (default 4000)
 *   8 numLevels       (default 3)
 *   9 codebookSize    (default 128)
 *  10 histItems       (default 8)
 *  11 dModel          (default 256)
 *  12 nLayers         (default 4)  — for V2: genLayers; histLayers=1
 *  13 lr              (default 1e-3)
 *  14 grpoSteps       (default 0 = skip RL)
 *  15 rqIters         (default 15)
 *
 * Example:
 *   java ... OneRecSFTBenchmark /path/MicroLens_1M_x1 mps onerecv2 64 40 15 10000 2000
 */
package org.bytedeco.pytorch.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.data.Batch;
import org.bytedeco.pytorch.recommend.data.DataLoader;
import org.bytedeco.pytorch.recommend.data.SIDSequenceDataset;
import org.bytedeco.pytorch.recommend.models.generative.ConstrainedBeamSearch;
import org.bytedeco.pytorch.recommend.models.generative.OneRec;
import org.bytedeco.pytorch.recommend.models.generative.OneRecV2;
import org.bytedeco.pytorch.recommend.models.generative.OpenOneRec;
import org.bytedeco.pytorch.recommend.models.generative.SemanticID;
import org.bytedeco.pytorch.plot.tqdm.Tqdm;
import org.bytedeco.pytorch.plot.tqdm.TqdmBar;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public final class OneRecSFTBenchmark {

    private OneRecSFTBenchmark() {}

    public static void main(String[] args) throws Exception {
        String dataRoot = args.length > 0 ? args[0]
                : "/Users/muller/Documents/code/cpp/VideoMMCTR/data/MicroLens_1M_x1";
        String deviceArg = args.length > 1 ? args[1] : "auto";
        String modelName = args.length > 2 ? args[2].toLowerCase() : "onerec";
        int batchSize = args.length > 3 ? Integer.parseInt(args[3]) : 64;
        int maxTrainBatches = args.length > 4 ? Integer.parseInt(args[4]) : 50;
        int maxValidBatches = args.length > 5 ? Integer.parseInt(args[5]) : 20;
        int trainRows = args.length > 6 ? Integer.parseInt(args[6]) : 20_000;
        int validRows = args.length > 7 ? Integer.parseInt(args[7]) : 4_000;
        int numLevels = args.length > 8 ? Integer.parseInt(args[8]) : 3;
        int codebookSize = args.length > 9 ? Integer.parseInt(args[9]) : 128;
        int histItems = args.length > 10 ? Integer.parseInt(args[10]) : 8;
        int dModel = args.length > 11 ? Integer.parseInt(args[11]) : 256;
        int nLayers = args.length > 12 ? Integer.parseInt(args[12]) : 4;
        double lr = args.length > 13 ? Double.parseDouble(args[13]) : 1e-3;
        int grpoSteps = args.length > 14 ? Integer.parseInt(args[14]) : 0;
        int rqIters = args.length > 15 ? Integer.parseInt(args[15]) : 15;

        System.out.println("=".repeat(64));
        System.out.println(" OneRec SFT Benchmark — MicroLens real data");
        System.out.println(" RQ-KMeans → SID sequences → " + modelName + " NTP"
                + (grpoSteps > 0 ? " → GRPO" : ""));
        System.out.println("=".repeat(64));
        System.out.println("  dataRoot     : " + dataRoot);
        System.out.println("  model        : " + modelName);
        System.out.println("  batchSize    : " + batchSize);
        System.out.println("  train/valid  : rows=" + trainRows + "/" + validRows
                + "  maxBatches=" + maxTrainBatches + "/" + maxValidBatches);
        System.out.println("  SID          : L=" + numLevels + " K=" + codebookSize);
        System.out.println("  histItems    : " + histItems);
        System.out.println("  dModel/layers: " + dModel + " / " + nLayers);
        System.out.println("  lr           : " + lr);
        System.out.println("  grpoSteps    : " + grpoSteps);
        System.out.println("  rqIters      : " + rqIters);

        Loader.load(org.bytedeco.pytorch.presets.torch.class);
        if ("cpu".equalsIgnoreCase(deviceArg)) {
            DeviceSupport.setDevice(DeviceSupport.DeviceType.CPU);
        } else if ("mps".equalsIgnoreCase(deviceArg)) {
            DeviceSupport.setDevice(DeviceSupport.DeviceType.MPS);
        } else if ("cuda".equalsIgnoreCase(deviceArg)) {
            DeviceSupport.setDevice(DeviceSupport.DeviceType.CUDA);
        }
        String device = DeviceSupport.backend();
        System.out.println("  device       : " + device);
        torch.manual_seed(20242025L);

        // ---- data: RQ-KMeans SID on real item emb + seq ----
        SIDSequenceDataset.MicroLensSplit split = SIDSequenceDataset.loadMicroLens(
                Path.of(dataRoot), numLevels, codebookSize, histItems,
                trainRows, validRows, rqIters, 20242025L);

        int maxSeqLen = split.train.maxSeqLen();
        long nHeads = Math.min(8, dModel >= 8 ? 8 : 4);
        while (dModel % nHeads != 0) nHeads--;

        System.out.print("Building model (" + modelName + ") ... ");
        System.out.flush();
        Module model;
        switch (modelName) {
            case "onerecv2":
            case "v2":
            case "lazy":
                OneRecV2 v2 = new OneRecV2(numLevels, codebookSize, dModel, nHeads,
                        /*histLayers*/ 1, /*genLayers*/ nLayers, maxSeqLen, 0.1, true, device);
                v2.summary();
                model = v2;
                break;
            case "openonerec":
            case "open":
            case "industrial":
                // itemic markers off for SIDSequenceDataset compatibility
                // (SemanticID.encode offset == OpenOneRec codeOffset when markers=false).
                // Enable markers once data pipeline emits SID_BEGIN/END spans.
                OpenOneRec open = new OpenOneRec(numLevels, codebookSize, dModel, nHeads,
                        nLayers, maxSeqLen, 0.1, true, /*itemic*/ false, /*aux*/ true, device);
                open.summary();
                model = open;
                break;
            case "onerec":
            default:
                OneRec one = new OneRec(numLevels, codebookSize, dModel, nHeads,
                        nLayers, maxSeqLen, 0.1, true, device);
                one.summary();
                model = one;
                break;
        }
        System.out.println("done.");

        Adam optimizer = new Adam(model.parameters(), new AdamOptions(lr));
        // warm adam
        System.out.print("Warming Adam ... ");
        System.out.flush();
        {
            Batch warm = split.train.getBatch(0);
            // make a tiny batch by stacking one row expanded — use DataLoader
            Iterable<Batch> warmIt = DataLoader.batches(split.train, Math.min(4, batchSize),
                    false, false, device);
            Batch w = warmIt.iterator().next();
            model.train(true);
            optimizer.zero_grad();
            Tensor loss = computeLoss(model, w.tokens);
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
            System.out.printf("ok  first_loss=%.4f%n", TensorHelpers.itemSafe(loss));
        }

        GenerativeTrainer trainer = new GenerativeTrainer(model)
                .learningRate((float) lr)
                .device(device)
                .verbose(true)
                .withOptimizer(optimizer)
                .withTrie(split.trie)
                .reportTokenAccuracy(true)
                .numEpochs(1)
                .maximizeMetric(false);

        // ---- SFT train loop with tqdm + PointerScope ----
        System.out.println("--- SFT TRAIN ---");
        model.train(true);
        Iterable<Batch> trainLoader = DataLoader.batches(split.train, batchSize, true, false, device);
        int trainTotal = maxTrainBatches > 0 ? maxTrainBatches
                : (int) Math.ceil(split.train.sizeLong() / (double) batchSize);
        double lossSum = 0;
        int n = 0;
        long t0 = System.nanoTime();
        long samples = 0;
        Iterator<Batch> trainIt = trainLoader.iterator();
        TqdmBar<Batch> bar = Tqdm.of(trainIt, trainTotal)
                .setDescription("SFT " + modelName)
                .setUnit("batch")
                .colour("green")
                .setMinInterval(0.15);
        try {
            while (bar.hasNext() && (maxTrainBatches <= 0 || n < maxTrainBatches)) {
                double lv = 0;
                try (PointerScope scope = new PointerScope()) {
                    Batch batch = bar.next();
                    Float v = trainer.trainStep(batch);
                    if (v != null) {
                        lv = v;
                        lossSum += v;
                        n++;
                        samples += batch.tokens != null ? batch.tokens.size(0) : batchSize;
                    }
                }
                Map<String, Object> pf = new HashMap<>();
                pf.put("loss", String.format("%.4f", lv));
                double elapsed = (System.nanoTime() - t0) / 1e9;
                pf.put("sps", String.format("%,d", elapsed > 0 ? (int) (samples / elapsed) : 0));
                bar.set_postfix(pf);
            }
        } finally {
            bar.close();
        }
        double trainSec = (System.nanoTime() - t0) / 1e9;
        System.out.printf("SFT TRAIN done  mean_loss=%.4f  steps=%d  samples=%,d  %.1fs  (%.1f sps)%n",
                n > 0 ? lossSum / n : 0, n, samples, trainSec,
                trainSec > 0 ? samples / trainSec : 0);

        // ---- valid ----
        System.out.println("--- SFT VALID ---");
        Iterable<Batch> validLoader = DataLoader.batches(split.valid, batchSize, false, false, device);
        // limit valid batches
        java.util.List<Batch> validLimited = new java.util.ArrayList<>();
        int vc = 0;
        for (Batch b : validLoader) {
            validLimited.add(b);
            vc++;
            if (maxValidBatches > 0 && vc >= maxValidBatches) break;
        }
        Map<String, Float> valMetrics = trainer.evaluate(validLimited);
        System.out.printf("SFT VALID  loss=%.4f  token_acc=%.4f%n",
                valMetrics.getOrDefault("loss", 0f),
                valMetrics.getOrDefault("token_acc", 0f));

        // ---- constrained beam generation smoke on a few val rows ----
        System.out.println("--- CONSTRAINED BEAM GENERATE (4 users) ---");
        model.eval();
        int shown = 0;
        for (Batch b : validLimited) {
            if (b.tokens == null) continue;
            int B = (int) Math.min(4, b.tokens.size(0));
            for (int i = 0; i < B && shown < 4; i++) {
                Tensor row = b.tokens.select(0, i).unsqueeze(0);
                long[] flat = TensorHelpers.toLongArray(
                        row.reshape(-1).to(org.bytedeco.pytorch.global.torch.ScalarType.Long)
                                .cpu().contiguous());
                int len = flat.length;
                while (len > numLevels + 1 && flat[len - 1] == SemanticID.PAD) len--;
                if (len <= numLevels) continue;
                int prefLen = len - numLevels;
                int[] pref = new int[prefLen];
                int[] target = new int[numLevels];
                for (int k = 0; k < prefLen; k++) pref[k] = (int) flat[k];
                for (int k = 0; k < numLevels; k++) target[k] = (int) flat[prefLen + k];
                Tensor ctx = TensorHelpers.tensor(pref, 1L, (long) prefLen)
                        .toType(org.bytedeco.pytorch.global.torch.ScalarType.Long);
                try {
                    ctx = ctx.to(new Device(device),
                            org.bytedeco.pytorch.global.torch.ScalarType.Long);
                } catch (Throwable ignored) {}
                int[] gen = ConstrainedBeamSearch.generateOne(
                        model, ctx, split.trie, /*beam*/ 4, numLevels, device);
                boolean hit = java.util.Arrays.equals(gen, target);
                boolean legal = split.trie.contains(gen);
                System.out.printf("  u%d gen=%s tgt=%s hit=%s legal=%s%n",
                        shown, java.util.Arrays.toString(gen), java.util.Arrays.toString(target),
                        hit, legal);
                shown++;
            }
            if (shown >= 4) break;
        }

        // ---- optional GRPO ----
        if (grpoSteps > 0) {
            System.out.println("--- GRPO POST-TRAIN (" + grpoSteps + " steps) ---");
            // reference = policy itself without updates would need a clone; skip KL if same
            GRPOTrainer grpo = new GRPOTrainer(model, /*ref*/ null, split.trie, numLevels)
                    .device(device)
                    .learningRate((float) (lr * 0.1))
                    .groupSize(4)
                    .beamSize(6)
                    .klCoeff(0.0f)
                    .rewardFn(GRPOTrainer.rankAwareReward(split.trie))
                    .verbose(true)
                    .withOptimizer(new Adam(model.parameters(), new AdamOptions(lr * 0.1)));
            Map<String, Float> grpoMetrics = grpo.fit(validLimited, grpoSteps);
            System.out.printf("GRPO  loss=%.4f  reward=%.4f  hit=%.4f%n",
                    grpoMetrics.getOrDefault("loss", 0f),
                    grpoMetrics.getOrDefault("reward_mean", 0f),
                    grpoMetrics.getOrDefault("hit_rate", 0f));
        }

        System.out.println("OneRec SFT benchmark complete.");
    }

    private static Tensor computeLoss(Module model, Tensor tokens) {
        if (model instanceof OneRec) return ((OneRec) model).computeLoss(tokens);
        if (model instanceof OneRecV2) return ((OneRecV2) model).computeLoss(tokens);
        if (model instanceof OpenOneRec) return ((OpenOneRec) model).computeLoss(tokens);
        throw new IllegalArgumentException("unknown model");
    }
}
