/*
 * IndustryTrainBenchmarks — mini end-to-end train loops per industry domain.
 *
 * Domains:
 *   news     — NRMS on MindDataset (synthetic fallback)
 *   pharma   — DeepDTA on DavisKibaDataset (synthetic fallback)
 *   ecommerce— ESCM2 / DBMTL on synthetic multi-task CTR/CVR features
 *   fintech  — TabTransformer on synthetic tabular risk
 *   shortvideo — WLR watch-time weighted BCE
 *   avazu    — existing AvazuDataset + MultiDomainCTR / WLR-style tower
 *
 * Run all:
 *   java org.bytedeco.pytorch.utils.recommend.benchmarks.IndustryTrainBenchmarks
 * Run one:
 *   java ... IndustryTrainBenchmarks news
 *   java ... IndustryTrainBenchmarks pharma
 *
 * Each benchmark:
 *   1) load/generate data
 *   2) build model
 *   3) a few Adam steps
 *   4) print loss trajectory (must decrease or stay finite)
 */
package org.bytedeco.pytorch.recommend.benchmarks;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.data.AvazuDataset;
import org.bytedeco.pytorch.recommend.data.industry.DavisKibaDataset;
import org.bytedeco.pytorch.recommend.data.industry.MindDataset;
import org.bytedeco.pytorch.recommend.models.ecommerce.DBMTL;
import org.bytedeco.pytorch.recommend.models.ecommerce.ESCM2;
import org.bytedeco.pytorch.recommend.models.ecommerce.MultiDomainCTR;
import org.bytedeco.pytorch.recommend.models.fintech.TabTransformer;
import org.bytedeco.pytorch.recommend.models.news.NRMS;
import org.bytedeco.pytorch.recommend.models.pharma.DeepDTA;
import org.bytedeco.pytorch.recommend.models.pharma.GraphDrugEncoder;
import org.bytedeco.pytorch.recommend.models.shortvideo.WLR;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class IndustryTrainBenchmarks {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private static final String DEVICE = "cpu";
    private static final float LR = 1e-3f;
    private static final int STEPS = 5;
    private static final int BATCH = 32;

    private IndustryTrainBenchmarks() {}

    public static void main(String[] args) {
        DeviceSupport.setDevice(DeviceSupport.DeviceType.CPU);
        String which = (args != null && args.length > 0) ? args[0].toLowerCase() : "all";
        int failed = 0;
        failed += runIf(which, "news", IndustryTrainBenchmarks::benchNews);
        failed += runIf(which, "pharma", IndustryTrainBenchmarks::benchPharma);
        failed += runIf(which, "ecommerce", IndustryTrainBenchmarks::benchEcommerce);
        failed += runIf(which, "fintech", IndustryTrainBenchmarks::benchFintech);
        failed += runIf(which, "shortvideo", IndustryTrainBenchmarks::benchShortVideo);
        failed += runIf(which, "avazu", IndustryTrainBenchmarks::benchAvazu);
        failed += runIf(which, "graphdrug", IndustryTrainBenchmarks::benchGraphDrug);

        System.out.println("============================================================");
        if (failed > 0) {
            System.out.println("Industry benchmarks FAILED: " + failed);
            System.exit(1);
        }
        System.out.println("Industry benchmarks ALL PASSED");
    }

    private static int runIf(String which, String name, RunnableFn fn) {
        if (!"all".equals(which) && !which.equals(name)) return 0;
        try {
            System.out.println("\n########## BENCH " + name.toUpperCase() + " ##########");
            fn.run();
            System.out.println("[PASS] " + name);
            return 0;
        } catch (Throwable t) {
            System.out.println("[FAIL] " + name + ": " + t.getMessage());
            t.printStackTrace(System.out);
            return 1;
        }
    }

    @FunctionalInterface
    private interface RunnableFn { void run() throws Exception; }

    // ---- news: NRMS + MindDataset ----
    private static void benchNews() {
        MindDataset.Split data = MindDataset.load(512, 7);
        System.out.println("  data n=" + data.size() + " synthetic=" + data.synthetic
                + " vocab=" + data.vocabSize);
        // smaller model for speed
        NRMS model = new NRMS(data.vocabSize, 32, 4, 16, 0.0f, DEVICE);
        Optimizer opt = adam(model);

        float first = Float.NaN, last = Float.NaN;
        int n = (int) data.size();
        for (int step = 0; step < STEPS; step++) {
            int start = (step * BATCH) % Math.max(1, n - BATCH);
            Tensor hist = data.historyTokenIds.narrow(0, start, BATCH);
            Tensor cand = data.candidateTokenIds.narrow(0, start, BATCH);
            Tensor lab = data.labels.narrow(0, start, BATCH); // [B, C]

            opt.zero_grad();
            Tensor scores = model.forward(hist, cand); // [B, C]
            Tensor p = scores.sigmoid();
            Tensor loss = bce(p.reshape(BATCH * (int) p.size(1)),
                    lab.reshape(BATCH * (int) lab.size(1)));
            loss.backward();
            opt.step();
            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            System.out.printf("  step %d loss=%.6f%n", step, v);
        }
        assertFinite(first, last);
    }

    // ---- pharma: DeepDTA + Davis/KIBA ----
    private static void benchPharma() {
        DavisKibaDataset.Split data = DavisKibaDataset.load(
                DavisKibaDataset.Source.DAVIS, 512, 11);
        System.out.println("  data n=" + data.size() + " synthetic=" + data.synthetic
                + " name=" + data.name);
        DeepDTA model = new DeepDTA(data.drugVocabSize, data.proteinVocabSize,
                32, 8, new int[]{3, 4}, 32, new long[]{64L, 32L}, DEVICE);
        Optimizer opt = adam(model);
        float first = Float.NaN, last = Float.NaN;
        int n = (int) data.size();
        for (int step = 0; step < STEPS; step++) {
            int start = (step * BATCH) % Math.max(1, n - BATCH);
            Tensor drug = data.drugTokens.narrow(0, start, BATCH);
            Tensor prot = data.proteinTokens.narrow(0, start, BATCH);
            Tensor y = data.affinity.narrow(0, start, BATCH);
            opt.zero_grad();
            Tensor pred = model.forward(drug, prot);
            Tensor loss = DeepDTA.mseLoss(pred, y);
            loss.backward();
            opt.step();
            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            System.out.printf("  step %d mse=%.6f%n", step, v);
        }
        assertFinite(first, last);
    }

    // ---- ecommerce: ESCM2 + DBMTL ----
    private static void benchEcommerce() {
        List<Feature> feats = sparseFeats(6, 200, 8);
        ESCM2 escm2 = new ESCM2(feats, new long[]{64L, 32L}, 3, true, true, DEVICE);
        DBMTL dbmtl = new DBMTL(feats, new long[]{64L, 32L}, new long[]{32L},
                3, true, false, true, DEVICE);

        runEcommerceModel("ESCM2", escm2, feats, true);
        runEcommerceModel("DBMTL", dbmtl, feats, false);
    }

    private static void runEcommerceModel(String name, Module model, List<Feature> feats,
                                          boolean isEscm2) {
        Optimizer opt = adam(model);
        float first = Float.NaN, last = Float.NaN;
        Random rng = new Random(21);
        for (int step = 0; step < STEPS; step++) {
            Map<String, Tensor> x = randomFeatMap(feats, BATCH, 200, rng);
            Tensor domain = torch.randint(3, new long[]{BATCH},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
            Tensor click = torch.randint(2, new long[]{BATCH},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            Tensor conv = click.mul(torch.randint(2, new long[]{BATCH},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))));

            opt.zero_grad();
            Tensor loss;
            if (isEscm2) {
                ESCM2 m = (ESCM2) model;
                Tensor preds = m.forward(x, domain);
                Tensor h = m.backboneFeatures(x, domain);
                Tensor elapsed = torch.rand(new long[]{BATCH}).mul(new Scalar(48f)).add(new Scalar(0.1f));
                loss = m.computeLoss(preds, click, conv, h, elapsed, 0.1f);
            } else {
                DBMTL m = (DBMTL) model;
                Tensor preds = m.forward(x, domain);
                Tensor aux = torch.randint(2, new long[]{BATCH},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
                loss = m.computeLoss(preds, click, conv, aux, null, null);
            }
            loss.backward();
            opt.step();
            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            System.out.printf("  %s step %d loss=%.6f%n", name, step, v);
        }
        assertFinite(first, last);
    }

    // ---- fintech: TabTransformer ----
    private static void benchFintech() {
        int[] vocabs = new int[]{50, 40, 30, 20};
        TabTransformer model = new TabTransformer(vocabs, 5, 16, 4, 2,
                new long[]{32L, 16L}, 0.0f, DEVICE);
        Optimizer opt = adam(model);
        float first = Float.NaN, last = Float.NaN;
        for (int step = 0; step < STEPS; step++) {
            Tensor cat = torch.randint(20, new long[]{BATCH, vocabs.length},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
            Tensor cont = torch.randn(new long[]{BATCH, 5},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            Tensor y = torch.randint(2, new long[]{BATCH},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            opt.zero_grad();
            Tensor p = model.forward(cat, cont);
            Tensor loss = bce(p, y);
            loss.backward();
            opt.step();
            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            System.out.printf("  step %d bce=%.6f%n", step, v);
        }
        assertFinite(first, last);
    }

    // ---- short video: WLR ----
    private static void benchShortVideo() {
        List<Feature> feats = sparseFeats(5, 150, 8);
        WLR model = new WLR(feats, new long[]{64L, 32L}, 8, true, DEVICE);
        Optimizer opt = adam(model);
        float first = Float.NaN, last = Float.NaN;
        Random rng = new Random(33);
        for (int step = 0; step < STEPS; step++) {
            Map<String, Tensor> x = randomFeatMap(feats, BATCH, 150, rng);
            Tensor y = torch.randint(2, new long[]{BATCH},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            Tensor w = torch.rand(new long[]{BATCH}).mul(new Scalar(10f)).add(new Scalar(0.5f));
            opt.zero_grad();
            Tensor p = model.forward(x);
            Tensor loss = WLR.weightedBceLoss(p, y, w);
            loss.backward();
            opt.step();
            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            System.out.printf("  step %d wlr=%.6f%n", step, v);
        }
        assertFinite(first, last);
    }

    // ---- avazu via existing downloader + MultiDomainCTR ----
    private static void benchAvazu() {
        AvazuDataset.Split split = AvazuDataset.load(0.8f, 2_000, 42);
        System.out.println("  avazu train=" + split.train.sizeLong()
                + " val=" + split.val.sizeLong());
        List<Feature> feats = new ArrayList<>();
        for (int i = 0; i < AvazuDataset.numFeatures(); i++) {
            feats.add(new SparseFeature("feat_" + i, 100_000, 8));
        }
        MultiDomainCTR model = new MultiDomainCTR(feats, 2, new long[]{64L, 32L}, DEVICE);
        Optimizer opt = adam(model);

        // manual mini-batch from TensorDataset sparse maps
        Map<String, Tensor> sparse = split.train.sparseFeatures();
        Tensor labels = split.train.labels();
        int n = (int) split.train.sizeLong();
        float first = Float.NaN, last = Float.NaN;
        for (int step = 0; step < STEPS; step++) {
            int start = (step * BATCH) % Math.max(1, n - BATCH);
            Map<String, Tensor> batch = new LinkedHashMap<>();
            for (Map.Entry<String, Tensor> e : sparse.entrySet()) {
                batch.put(e.getKey(), e.getValue().narrow(0, start, BATCH));
            }
            Tensor y = labels.narrow(0, start, BATCH);
            Tensor domain = torch.zeros(new long[]{BATCH},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
            opt.zero_grad();
            Tensor p = model.forward(batch, domain);
            Tensor loss = bce(p, y);
            loss.backward();
            opt.step();
            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            System.out.printf("  step %d avazu-bce=%.6f%n", step, v);
        }
        assertFinite(first, last);
    }

    // ---- graph drug encoder smoke-train ----
    private static void benchGraphDrug() {
        int nAtoms = 12;
        int featDim = 8;
        GraphDrugEncoder enc = new GraphDrugEncoder(featDim, 16, 16, 2, 0.0f, DEVICE);
        // random atom features + random adj, normalize
        Tensor x = torch.randn(new long[]{nAtoms, featDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        Tensor adjRaw = torch.randint(2, new long[]{nAtoms, nAtoms},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        // symmetrize
        adjRaw = adjRaw.add(adjRaw.transpose(0, 1)).gt(new Scalar(0)).toType(ScalarType.Float);
        Tensor adj = GraphDrugEncoder.normalizeAdj(adjRaw);
        Optimizer opt = adam(enc);
        float first = Float.NaN, last = Float.NaN;
        for (int step = 0; step < STEPS; step++) {
            opt.zero_grad();
            Tensor z = enc.forward(x, adj); // [1, 16]
            Tensor target = torch.ones_like(z);
            Tensor loss = z.sub(target).pow(new Scalar(2.0f)).mean();
            loss.backward();
            opt.step();
            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            System.out.printf("  step %d graph-mse=%.6f%n", step, v);
        }
        assertFinite(first, last);
    }

    // ---- utils ----

    private static Optimizer adam(Module model) {
        AdamOptions opts = new AdamOptions(LR);
        return new Adam(model.parameters(), opts);
    }

    private static Tensor bce(Tensor p, Tensor y) {
        Tensor pp = p.clamp(new ScalarOptional(new Scalar(1e-6f)),
                new ScalarOptional(new Scalar(1.0f - 1e-6f)));
        Tensor yy = y.toType(ScalarType.Float);
        return yy.neg().mul(pp.log())
                .add(torch.sub(torch.ones_like(yy), yy).neg()
                        .mul(torch.sub(torch.ones_like(pp), pp).log()))
                .mean();
    }

    private static List<Feature> sparseFeats(int n, int vocab, int dim) {
        List<Feature> list = new ArrayList<>();
        for (int i = 0; i < n; i++) list.add(new SparseFeature("f" + i, vocab, dim));
        return list;
    }

    private static Map<String, Tensor> randomFeatMap(List<Feature> feats, int batch,
                                                     int vocab, Random rng) {
        Map<String, Tensor> m = new LinkedHashMap<>();
        for (Feature f : feats) {
            m.put(f.name(), torch.randint(Math.max(vocab, 2), new long[]{batch},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))));
        }
        return m;
    }

    private static void assertFinite(float first, float last) {
        if (Float.isNaN(first) || Float.isInfinite(first)
                || Float.isNaN(last) || Float.isInfinite(last)) {
            throw new IllegalStateException("non-finite loss first=" + first + " last=" + last);
        }
        System.out.printf("  loss first=%.6f last=%.6f%n", first, last);
    }
}
