/*
 * Smoke: CriteoDataset (native Dataset) → DeepFM / DCNv2 one-step train,
 * plus BatchCollator modes (FLAT / MULTI_HOT / PADDED_SEQUENCE / HYBRID).
 *
 *   java -cp ... samples.recommend.SmokeCriteoRanking
 */
package samples.recommend;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.dataloader.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.data.ExampleVectorIterator;
import org.bytedeco.pytorch.data.dataloader.RandomDataLoader;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.Recommend;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.losses.Losses;
import org.bytedeco.pytorch.utils.recommend.data.Batch;
import org.bytedeco.pytorch.utils.recommend.data.BatchCollator;
import org.bytedeco.pytorch.utils.recommend.data.CriteoDataset;
import org.bytedeco.pytorch.utils.recommend.data.DataGenerator;
import org.bytedeco.pytorch.utils.recommend.data.DataLoader;
import org.bytedeco.pytorch.utils.recommend.data.RecommendDataset;
import org.bytedeco.pytorch.utils.recommend.data.SequenceDataset;
import org.bytedeco.pytorch.utils.recommend.data.TensorDataset;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DCNv2;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DeepFM;
import org.bytedeco.pytorch.dataframe.dataset.NativeBatchSupport;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class SmokeCriteoRanking {

    static int pass = 0;
    static int fail = 0;

    public static void main(String[] args) {
        Recommend.loadNative();
        // Pin CPU before any model/tensor alloc — avoids MPS Dropout SIGSEGV and
        // CrossNet device drift in this single-process smoke.
        DeviceSupport.setDevice(DeviceSupport.DeviceType.CPU);

        System.out.println("==========================================");
        System.out.println("  Smoke: Criteo → DeepFM / DCNv2 + Collator");
        System.out.println("  backend=" + DeviceSupport.backend());
        System.out.println("==========================================");

        // Always use synthetic for deterministic offline smoke (small n).
        int n = 2_048;
        for (String a : args) {
            if (a.startsWith("--n=")) {
                try { n = Integer.parseInt(a.substring(4)); } catch (Exception ignored) {}
            }
        }

        CriteoDataset.Split split = CriteoDataset.generateSynthetic(0.8f, n, 42);
        TensorDataset train = split.train;
        System.out.println("[INFO] train size=" + train.sizeLong()
                + " (native Dataset? " + (train instanceof org.bytedeco.pytorch.data.Dataset) + ")"
                + " sparse=" + train.sparseOrder().size()
                + " dense=" + train.denseOrder().size());

        check("extends native Dataset", train instanceof org.bytedeco.pytorch.data.Dataset);
        check("sizeLong > 0", train.sizeLong() > 0);

        // Feature list from Criteo sparse_* keys
        List<Feature> sparseFeats = new ArrayList<>();
        for (String name : train.sparseOrder()) {
            sparseFeats.add(Features.sparse(name, CriteoDataset.sparseVocab(), 8));
        }
        check("sparse feature count == 26", sparseFeats.size() == CriteoDataset.numSparse());

        // DCNv2 first (avoids any leftover autograd state from DeepFM on short smokes)
        smokeDCNv2(train, sparseFeats);
        smokeDeepFM(train, sparseFeats);
        smokeNativeDataLoader(train);
        smokeBatchCollator(train);
        smokeSequenceCollator();

        System.out.println("------------------------------------------");
        System.out.println("Summary: PASS=" + pass + " FAIL=" + fail);
        if (fail > 0) System.exit(1);
    }

    private static void smokeDeepFM(TensorDataset train, List<Feature> sparseFeats) {
        try {
            // deep = fm = all sparse (classic DeepFM)
            // dropout=0 avoids intermittent native DropoutImpl SIGSEGV in short smokes
            DeepFM model = new DeepFM(sparseFeats, sparseFeats, 8, new long[]{64L, 32L}, 0.0f,
                    DeviceSupport.backend());
            model.train(true);

            AdamOptions optOpts = new AdamOptions(1e-3);
            Adam optim = new Adam(model.parameters(), optOpts);

            Iterator<Batch> it = DataLoader.batches(train, 64, true, true).iterator();
            Batch batch = it.next();
            Map<String, Tensor> sparse = batch.sparseFeatures;
            Tensor labels = batch.labels;

            optim.zero_grad();
            Tensor logits = model.forward(sparse);
            // logits [B,1], labels [B] or [B,1]
            Tensor y = labels.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float)
                    .reshape(-1L, 1L);
            Tensor loss = new Losses.BCEWithLogitsLoss().apply(
                    logits.reshape(-1L), y.reshape(-1L));
            loss.backward();
            optim.step();

            double lv = TensorHelpers.itemSafe(loss);
            if (Double.isNaN(lv) || Double.isInfinite(lv)) {
                throw new IllegalStateException("DeepFM loss NaN/Inf: " + lv);
            }
            System.out.println("[PASS] DeepFM one-step loss=" + lv
                    + " logits_shape=" + java.util.Arrays.toString(logits.shape()));
            pass++;
        } catch (Throwable t) {
            System.out.println("[FAIL] DeepFM: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }
    }

    private static void smokeDCNv2(TensorDataset train, List<Feature> sparseFeats) {
        try {
            // DeviceSupport is pinned to CPU in main(); CrossNetV2 now uses Linear bias
            // so weights move with the module. dropout=0 for short-smoke stability.
            String device = DeviceSupport.backend();
            DCNv2 model = new DCNv2(sparseFeats, 8, 2, false, 4,
                    new long[]{64L, 32L}, 0.0f, device);
            model.train(true);

            Adam optim = new Adam(model.parameters(), new AdamOptions(1e-3));
            Batch batch = DataLoader.batches(train, 64, true, true, device).iterator().next();

            optim.zero_grad();
            Tensor logits = model.forward(batch.sparseFeatures);
            Tensor y = batch.labels.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float)
                    .reshape(-1L);
            Tensor pred = logits.reshape(-1L);
            Tensor loss = new Losses.BCEWithLogitsLoss().apply(pred, y);
            loss.backward();
            optim.step();

            double lv = TensorHelpers.itemSafe(loss);
            if (Double.isNaN(lv) || Double.isInfinite(lv)) {
                throw new IllegalStateException("DCNv2 loss NaN/Inf: " + lv);
            }
            System.out.println("[PASS] DCNv2 one-step loss=" + lv
                    + " logits_shape=" + java.util.Arrays.toString(logits.shape())
                    + " device=" + device);
            pass++;
        } catch (Throwable t) {
            System.out.println("[FAIL] DCNv2: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }
    }

    private static void smokeNativeDataLoader(TensorDataset train) {
        try {
            // Prove native RandomDataLoader accepts RecommendDataset (extends Dataset)
            RandomDataLoader loader = train.randomDataLoader(32);
            ExampleVectorIterator begin = loader.begin();
            ExampleVectorIterator end = loader.end();
            if (begin.equals(end)) {
                throw new IllegalStateException("empty native loader");
            }
            ExampleVector raw = begin.access();
            Example stacked = NativeBatchSupport.stack(raw);
            long B = stacked.data().size(0);
            if (B <= 0) throw new IllegalStateException("stacked batch size 0");
            System.out.println("[PASS] native RandomDataLoader batch data_shape="
                    + java.util.Arrays.toString(stacked.data().shape())
                    + " target_shape=" + java.util.Arrays.toString(stacked.target().shape()));
            pass++;
        } catch (Throwable t) {
            System.out.println("[FAIL] native DataLoader: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }
    }

    private static void smokeBatchCollator(TensorDataset train) {
        try {
            List<Batch> rows = new ArrayList<>();
            for (int i = 0; i < 8; i++) rows.add(train.getBatch(i));

            // FLAT_SCALARS
            BatchCollator flat = new BatchCollator(
                    BatchCollator.Options.defaults().mode(BatchCollator.Mode.FLAT_SCALARS));
            BatchCollator.Collated c1 = flat.collate(rows);
            if (c1.example == null || c1.example.data().size(0) != 8) {
                throw new IllegalStateException("FLAT example batch dim != 8");
            }
            System.out.println("[PASS] Collator FLAT data="
                    + java.util.Arrays.toString(c1.example.data().shape())
                    + " target=" + java.util.Arrays.toString(c1.example.target().shape()));
            pass++;

            // MULTI_HOT on first 3 sparse fields
            List<String> keys = train.sparseOrder();
            BatchCollator.Options mhOpts = BatchCollator.Options.defaults()
                    .mode(BatchCollator.Mode.MULTI_HOT)
                    .multiHot(
                            new BatchCollator.MultiHotSpec(keys.get(0), 128),
                            new BatchCollator.MultiHotSpec(keys.get(1), 128),
                            new BatchCollator.MultiHotSpec(keys.get(2), 64));
            BatchCollator.Collated c2 = new BatchCollator(mhOpts).collate(rows);
            if (c2.multiHotFeatures.size() != 3) {
                throw new IllegalStateException("expected 3 multi-hot fields");
            }
            Tensor mh0 = c2.multiHotFeatures.get(keys.get(0));
            if (mh0.size(0) != 8 || mh0.size(1) != 128) {
                throw new IllegalStateException("multi-hot shape " + java.util.Arrays.toString(mh0.shape()));
            }
            System.out.println("[PASS] Collator MULTI_HOT fields=" + c2.multiHotFeatures.size()
                    + " ex=" + java.util.Arrays.toString(c2.example.data().shape()));
            pass++;

            // STACKED_FEATURES
            BatchCollator.Collated c3 = new BatchCollator(
                    BatchCollator.Options.defaults().mode(BatchCollator.Mode.STACKED_FEATURES))
                    .collate(rows);
            if (c3.batch.numSamples() != 8) {
                throw new IllegalStateException("stacked numSamples != 8");
            }
            if (c3.example != null) {
                throw new IllegalStateException("STACKED should not produce Example");
            }
            System.out.println("[PASS] Collator STACKED numSamples=" + c3.batch.numSamples()
                    + " sparseKeys=" + c3.batch.sparseFeatures.size());
            pass++;
        } catch (Throwable t) {
            System.out.println("[FAIL] BatchCollator: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }
    }

    private static void smokeSequenceCollator() {
        try {
            SequenceDataset seqDs = DataGenerator.generateSequenceData(64, 12, 1000, 7);
            List<Batch> rows = new ArrayList<>();
            // make variable lengths by truncating tokens view — use as-is with PADDED mode
            for (int i = 0; i < 5; i++) rows.add(seqDs.getBatch(i));

            BatchCollator.Options opts = BatchCollator.Options.defaults()
                    .mode(BatchCollator.Mode.PADDED_SEQUENCE)
                    .sequences(new BatchCollator.SequenceSpec("item_seq", 16, 0L, false, true));
            BatchCollator.Collated c = new BatchCollator(opts).collate(rows);
            Tensor padded = c.batch.sequenceFeatures.get("item_seq");
            Tensor mask = c.sequenceMasks.get("item_seq");
            if (padded == null || padded.size(0) != 5 || padded.size(1) != 16) {
                throw new IllegalStateException("padded shape "
                        + (padded == null ? "null" : java.util.Arrays.toString(padded.shape())));
            }
            if (mask == null || mask.size(1) != 16) {
                throw new IllegalStateException("mask missing/bad shape");
            }

            // HYBRID
            BatchCollator.Options hybrid = BatchCollator.Options.defaults()
                    .mode(BatchCollator.Mode.HYBRID)
                    .sequences(new BatchCollator.SequenceSpec("item_seq", 16));
            BatchCollator.Collated h = new BatchCollator(hybrid).collate(rows);
            if (h.example == null || h.example.data().size(0) != 5) {
                throw new IllegalStateException("HYBRID example bad");
            }
            System.out.println("[PASS] Collator PADDED/HYBRID padded="
                    + java.util.Arrays.toString(padded.shape())
                    + " hybrid_data=" + java.util.Arrays.toString(h.example.data().shape()));
            pass++;
        } catch (Throwable t) {
            System.out.println("[FAIL] Sequence collator: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }
    }

    private static void check(String name, boolean ok) {
        if (ok) {
            System.out.println("[PASS] " + name);
            pass++;
        } else {
            System.out.println("[FAIL] " + name);
            fail++;
        }
    }
}
