/*
 * OneRec smoke demo — synthetic Semantic-ID sequences, train a few steps, generate.
 *
 * Mirrors the MiniOneRec SFT loop (history SIDs → next-item SID) without needing
 * a full industrial corpus. Validates:
 *   SemanticID encode/trie, OneRec NTP train, GenerativeTrainer, constrained decode.
 *
 * Run:
 *   java ... org.bytedeco.pytorch.utils.recommend.trainers.OneRecDemo \
 *     [device] [batchSize] [steps] [numItems] [histLen]
 *
 * Reference:
 *   - OneRec https://arxiv.org/abs/2502.18965
 *   - MiniOneRec https://github.com/AkaliKong/MiniOneRec
 *   - OpenOneRec https://github.com/Kuaishou-OneRec/OpenOneRec
 */
package org.bytedeco.pytorch.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.data.Batch;
import org.bytedeco.pytorch.recommend.models.generative.OneRec;
import org.bytedeco.pytorch.recommend.models.generative.SemanticID;
import org.bytedeco.pytorch.plot.tqdm.Tqdm;
import org.bytedeco.pytorch.plot.tqdm.TqdmBar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class OneRecDemo {

    private OneRecDemo() {}

    /** Synthetic catalog: numItems random L-level SIDs over codebook K. */
    static final class Catalog {
        final int numLevels;
        final int codebookSize;
        final int[][] itemSids; // [numItems][L]
        final SemanticID.Trie trie;

        Catalog(int numItems, int numLevels, int codebookSize, long seed) {
            this.numLevels = numLevels;
            this.codebookSize = codebookSize;
            this.itemSids = new int[numItems][numLevels];
            this.trie = new SemanticID.Trie(numLevels, codebookSize);
            Random rng = new Random(seed);
            // ensure uniqueness of SID paths (retry on collision)
            java.util.Set<String> seen = new java.util.HashSet<>();
            for (int i = 0; i < numItems; i++) {
                int tries = 0;
                while (true) {
                    int[] codes = new int[numLevels];
                    StringBuilder key = new StringBuilder();
                    for (int l = 0; l < numLevels; l++) {
                        codes[l] = rng.nextInt(codebookSize);
                        key.append(codes[l]).append(',');
                    }
                    if (seen.add(key.toString()) || tries++ > 50) {
                        itemSids[i] = codes;
                        trie.insertCodes(codes);
                        break;
                    }
                }
            }
        }
    }

    /** Build one NTP batch: prefix = BOS + hist SIDs, full seq adds target item SID (+ optional EOS). */
    static Batch makeBatch(Catalog cat, int batchSize, int histLen, boolean addEos, String device, Random rng) {
        int L = cat.numLevels;
        int K = cat.codebookSize;
        int targetCodes = L;
        int histCodes = histLen * L;
        int seqLen = 1 + histCodes + targetCodes + (addEos ? 1 : 0); // BOS + hist + target [+ EOS]

        int[] flat = new int[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++) {
            int base = b * seqLen;
            flat[base] = SemanticID.BOS;
            int p = 1;
            // history items
            for (int h = 0; h < histLen; h++) {
                int item = rng.nextInt(cat.itemSids.length);
                int[] sid = cat.itemSids[item];
                for (int l = 0; l < L; l++) {
                    flat[base + p++] = SemanticID.encode(l, sid[l], K);
                }
            }
            // target item
            int tgt = rng.nextInt(cat.itemSids.length);
            int[] tsid = cat.itemSids[tgt];
            for (int l = 0; l < L; l++) {
                flat[base + p++] = SemanticID.encode(l, tsid[l], K);
            }
            if (addEos) flat[base + p] = SemanticID.EOS;
        }

        Tensor tokens = TensorHelpers.tensor(flat, batchSize, seqLen).toType(ScalarType.Long);
        if (device != null && !"cpu".equals(device)) {
            tokens = tokens.to(new Device(device), ScalarType.Long);
        }
        return new Batch(
                Collections.emptyMap(), Collections.emptyMap(), Collections.emptyMap(),
                null, tokens, null, null, null,
                Collections.emptyMap(), null, null);
    }

    static List<Batch> makeEpoch(Catalog cat, int numBatches, int batchSize, int histLen,
                                 boolean addEos, String device, long seed) {
        Random rng = new Random(seed);
        List<Batch> out = new ArrayList<>(numBatches);
        for (int i = 0; i < numBatches; i++) {
            out.add(makeBatch(cat, batchSize, histLen, addEos, device, rng));
        }
        return out;
    }

    public static void main(String[] args) {
        String device = args.length > 0 ? args[0] : DeviceSupport.backend();
        int batchSize = args.length > 1 ? Integer.parseInt(args[1]) : 32;
        int steps = args.length > 2 ? Integer.parseInt(args[2]) : 40;
        int numItems = args.length > 3 ? Integer.parseInt(args[3]) : 500;
        int histLen = args.length > 4 ? Integer.parseInt(args[4]) : 5;

        final int numLevels = 3;
        final int codebookSize = 64; // small for smoke (industrial OneRec often uses 256+)
        final int dModel = 128;
        final int nLayers = 2;
        final int nHeads = 4;
        final boolean addEos = false;

        System.out.println("=".repeat(60));
        System.out.println(" OneRec Demo — Kuaishou generative rec (SID + NTP)");
        System.out.println("=".repeat(60));
        System.out.println("  device     : " + device);
        System.out.println("  batchSize  : " + batchSize);
        System.out.println("  steps      : " + steps);
        System.out.println("  numItems   : " + numItems);
        System.out.println("  histLen    : " + histLen);
        System.out.println("  SID        : L=" + numLevels + " K=" + codebookSize
                + " vocab=" + SemanticID.vocabSize(numLevels, codebookSize));

        Loader.load(org.bytedeco.pytorch.presets.torch.class);
        // Force device before model build
        if ("cpu".equalsIgnoreCase(device)) {
            DeviceSupport.setDevice(DeviceSupport.DeviceType.CPU);
        } else if ("mps".equalsIgnoreCase(device)) {
            DeviceSupport.setDevice(DeviceSupport.DeviceType.MPS);
        } else if ("cuda".equalsIgnoreCase(device)) {
            DeviceSupport.setDevice(DeviceSupport.DeviceType.CUDA);
        }
        device = DeviceSupport.backend();
        torch.manual_seed(42L);

        System.out.print("Building synthetic SID catalog ... ");
        System.out.flush();
        Catalog catalog = new Catalog(numItems, numLevels, codebookSize, 42L);
        System.out.println("trie size=" + catalog.trie.size());

        int maxSeqLen = 1 + histLen * numLevels + numLevels + 2; // BOS+hist+target+slop
        System.out.print("Building OneRec ... ");
        System.out.flush();
        OneRec model = new OneRec(
                numLevels, codebookSize, dModel, nHeads, nLayers,
                Math.max(maxSeqLen, 128), 0.1, true, device);
        model.summary();

        GenerativeTrainer trainer = new GenerativeTrainer(model)
                .learningRate(1e-3f)
                .numEpochs(1)
                .device(device)
                .verbose(true)
                .withTrie(catalog.trie)
                .reportTokenAccuracy(true);
        // Warm Adam outside PointerScope
        Adam opt = new Adam(model.parameters(), new AdamOptions(1e-3));
        trainer.withOptimizer(opt);
        {
            Batch warm = makeBatch(catalog, Math.min(4, batchSize), histLen, addEos, device, new Random(0));
            model.train(true);
            opt.zero_grad();
            Tensor loss = model.computeLoss(warm.tokens);
            loss.backward();
            opt.step();
            opt.zero_grad();
            System.out.println("Adam warm-up done. first_loss=" + String.format("%.4f",
                    TensorHelpers.itemSafe(loss)));
        }

        List<Batch> trainSet = makeEpoch(catalog, steps, batchSize, histLen, addEos, device, 7L);
        List<Batch> validSet = makeEpoch(catalog, Math.max(5, steps / 4), batchSize, histLen, addEos, device, 99L);

        System.out.println("--- TRAIN (" + steps + " steps) ---");
        model.train(true);
        long t0 = System.nanoTime();
        double lossSum = 0.0;
        int n = 0;
        Iterator<Batch> it = trainSet.iterator();
        TqdmBar<Batch> bar = Tqdm.of(it, steps)
                .setDescription("OneRec train")
                .setUnit("batch")
                .colour("green")
                .setMinInterval(0.15);
        try {
            int step = 0;
            while (bar.hasNext() && step < steps) {
                PointerScope scope = new PointerScope();
                double lv = 0.0;
                try {
                    Batch batch = bar.next();
                    Float v = trainer.trainStep(batch);
                    if (v != null) {
                        lv = v;
                        lossSum += v;
                        n++;
                    }
                    step++;
                } finally {
                    scope.close();
                }
                Map<String, Object> pf = new HashMap<>();
                pf.put("loss", String.format("%.4f", lv));
                bar.set_postfix(pf);
            }
        } finally {
            bar.close();
        }
        double trainSec = (System.nanoTime() - t0) / 1e9;
        System.out.printf("TRAIN done  mean_loss=%.4f  steps=%d  %.1fs  (%.1f steps/s)%n",
                n > 0 ? lossSum / n : 0.0, n, trainSec, n / Math.max(trainSec, 1e-6));

        System.out.println("--- VALID ---");
        Map<String, Float> metrics = trainer.evaluate(validSet);
        System.out.printf("VALID  loss=%.4f  token_acc=%.4f%n",
                metrics.getOrDefault("loss", 0f),
                metrics.getOrDefault("token_acc", 0f));

        // Constrained generation smoke
        System.out.println("--- GENERATE (constrained, 4 users) ---");
        model.eval();
        Batch genBatch = makeBatch(catalog, 4, histLen, false, device, new Random(123));
        // use only prefix (drop last L target tokens) for generation context
        Tensor full = genBatch.tokens;
        long T = full.size(1);
        Tensor prefix = full.narrow(1, 0, T - numLevels);
        SemanticID.ConstrainedDecoder[] decoders = new SemanticID.ConstrainedDecoder[4];
        for (int i = 0; i < 4; i++) decoders[i] = new SemanticID.ConstrainedDecoder(catalog.trie);
        Tensor gen = model.generateItem(prefix, decoders);
        long[] flat = TensorHelpers.toLongArray(gen.cpu().toType(ScalarType.Long).contiguous());
        for (int b = 0; b < 4; b++) {
            int[] toks = new int[numLevels];
            boolean valid = true;
            for (int l = 0; l < numLevels; l++) {
                toks[l] = (int) flat[b * numLevels + l];
                int[] dec = SemanticID.decode(toks[l], numLevels, codebookSize);
                if (dec == null || dec[0] != l) valid = false;
            }
            boolean inTrie = catalog.trie.contains(toks);
            System.out.printf("  user%d SID tokens=%s  level_ok=%s  in_catalog=%s%n",
                    b, java.util.Arrays.toString(toks), valid, inTrie);
        }

        System.out.println("OneRec demo complete.");
    }
}
