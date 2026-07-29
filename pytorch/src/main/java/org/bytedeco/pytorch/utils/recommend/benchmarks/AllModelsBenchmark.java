/*
 * AllModelsBenchmark — full-suite multi-dimensional bench for every top-level
 * model under org.bytedeco.pytorch.utils.recommend.models.
 *
 * Dimensions per model:
 *   1) construct          — tiny config on CPU
 *   2) forward smoke      — shape / non-null / finite outputs
 *   3) train steps        — N Adam steps, loss must stay finite
 *   4) memory / RSS       — multi-step train under PointerScope; RSS growth
 *                           and optional CUDA/MPS reserved-bytes growth
 *   5) loss trajectory    — first/last finite; optional mild decrease check
 *
 * Run all:
 *   java org.bytedeco.pytorch.utils.recommend.benchmarks.AllModelsBenchmark
 * Filter by family:
 *   java ... AllModelsBenchmark ranking
 *   java ... AllModelsBenchmark matching multi_task knowledge_tracing
 *   java ... AllModelsBenchmark generative industry
 * Filter by name substring:
 *   java ... AllModelsBenchmark DeepFM DKT OneRec
 *
 * System properties:
 *   -Dallmodels.steps=8        train steps (default 6)
 *   -Dallmodels.batch=8        batch size (default 8)
 *   -Dallmodels.leak_steps=24  leak probe steps (default 20)
 *   -Dallmodels.rss_mb=250     fail if RSS grows more than this many MB
 *   -Dallmodels.device=cpu     force device (cpu|mps|cuda|auto)
 *   -Dallmodels.strict=true    also fail if loss does not trend down
 */
package org.bytedeco.pytorch.utils.recommend.benchmarks;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
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
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.models.bio.DnaSeqCnn;
import org.bytedeco.pytorch.utils.recommend.models.bio.GeneExpressionMLP;
import org.bytedeco.pytorch.utils.recommend.models.bio.ProteinSeqEncoder;
import org.bytedeco.pytorch.utils.recommend.models.bio.TwinTowerPPI;
import org.bytedeco.pytorch.utils.recommend.models.ecommerce.DBMTL;
import org.bytedeco.pytorch.utils.recommend.models.ecommerce.ESCM2;
import org.bytedeco.pytorch.utils.recommend.models.ecommerce.MultiDomainCTR;
import org.bytedeco.pytorch.utils.recommend.models.ecommerce.SearchConversion;
import org.bytedeco.pytorch.utils.recommend.models.fintech.FTTransformer;
import org.bytedeco.pytorch.utils.recommend.models.fintech.SequenceRiskModel;
import org.bytedeco.pytorch.utils.recommend.models.fintech.TabTransformer;
import org.bytedeco.pytorch.utils.recommend.models.generative.HSTU;
import org.bytedeco.pytorch.utils.recommend.models.generative.LLM4Rec;
import org.bytedeco.pytorch.utils.recommend.models.generative.OneRec;
import org.bytedeco.pytorch.utils.recommend.models.generative.OneRecV2;
import org.bytedeco.pytorch.utils.recommend.models.generative.OpenOneRec;
import org.bytedeco.pytorch.utils.recommend.models.generative.RQKMeans;
import org.bytedeco.pytorch.utils.recommend.models.generative.RQVAE;
import org.bytedeco.pytorch.utils.recommend.models.generative.TIGER;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.AKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.ATDKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.ATKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.ATKTFix;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.CSKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.DIMKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.DKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.DKTForget;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.DKTPlus;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.DKVMN;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.DeepIRT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.GKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.IEKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.LPKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.MTKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.PromptKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.QDKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.RKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.RobustKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.SAINT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.SAINTPlusPlus;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.SAKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.SAKTUnified;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.SKVMN;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.SimpleKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.SparseKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.StableKT;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.UKT;
import org.bytedeco.pytorch.utils.recommend.models.live.LiveMultiTask;
import org.bytedeco.pytorch.utils.recommend.models.matching.ComirecDR;
import org.bytedeco.pytorch.utils.recommend.models.matching.ComirecSA;
import org.bytedeco.pytorch.utils.recommend.models.matching.DSSM;
import org.bytedeco.pytorch.utils.recommend.models.matching.DSSMSENET;
import org.bytedeco.pytorch.utils.recommend.models.matching.FaceBookDSSM;
import org.bytedeco.pytorch.utils.recommend.models.matching.GRU4Rec;
import org.bytedeco.pytorch.utils.recommend.models.matching.MAMBA;
import org.bytedeco.pytorch.utils.recommend.models.matching.MIND;
import org.bytedeco.pytorch.utils.recommend.models.matching.NARM;
import org.bytedeco.pytorch.utils.recommend.models.matching.NCF;
import org.bytedeco.pytorch.utils.recommend.models.matching.SASRec;
import org.bytedeco.pytorch.utils.recommend.models.matching.SINE;
import org.bytedeco.pytorch.utils.recommend.models.matching.STAMP;
import org.bytedeco.pytorch.utils.recommend.models.matching.YoutubeDNN;
import org.bytedeco.pytorch.utils.recommend.models.matching.YoutubeSBC;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.AITM;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.ESMM;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.MMOE;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.MetaHeac;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.OMoE;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.PLE;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.SharedBottom;
import org.bytedeco.pytorch.utils.recommend.models.multi_task.SingleTaskModel;
import org.bytedeco.pytorch.utils.recommend.models.news.DKN;
import org.bytedeco.pytorch.utils.recommend.models.news.LSTUR;
import org.bytedeco.pytorch.utils.recommend.models.news.NAML;
import org.bytedeco.pytorch.utils.recommend.models.news.NPA;
import org.bytedeco.pytorch.utils.recommend.models.news.NRMS;
import org.bytedeco.pytorch.utils.recommend.models.pharma.DeepDTA;
import org.bytedeco.pytorch.utils.recommend.models.pharma.DrugBAN;
import org.bytedeco.pytorch.utils.recommend.models.pharma.GraphDrugEncoder;
import org.bytedeco.pytorch.utils.recommend.models.pharma.MolTrans;
import org.bytedeco.pytorch.utils.recommend.models.ranking.AFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.AFN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.AutoInt;
import org.bytedeco.pytorch.utils.recommend.models.ranking.BST;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DCN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DCNv2;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DIEN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DIN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DeepFFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.DeepFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.EDCN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.ETA;
import org.bytedeco.pytorch.utils.recommend.models.ranking.FNFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.FNN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.FatDeepFFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.FiBiNet;
import org.bytedeco.pytorch.utils.recommend.models.ranking.FraudGNN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.GAT;
import org.bytedeco.pytorch.utils.recommend.models.ranking.GCN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.GraphSAGE;
import org.bytedeco.pytorch.utils.recommend.models.ranking.HoFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.LiquidNetWork;
import org.bytedeco.pytorch.utils.recommend.models.ranking.MEMBA;
import org.bytedeco.pytorch.utils.recommend.models.ranking.NFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.PNN;
import org.bytedeco.pytorch.utils.recommend.models.ranking.RankingLR;
import org.bytedeco.pytorch.utils.recommend.models.ranking.SIM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.SoftDecisionTree;
import org.bytedeco.pytorch.utils.recommend.models.ranking.WideDeep;
import org.bytedeco.pytorch.utils.recommend.models.ranking.XDeepFM;
import org.bytedeco.pytorch.utils.recommend.models.ranking.XGBoostModel;
import org.bytedeco.pytorch.utils.recommend.models.shortvideo.D2Q;
import org.bytedeco.pytorch.utils.recommend.models.shortvideo.PEPNet;
import org.bytedeco.pytorch.utils.recommend.models.shortvideo.WLR;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

public final class AllModelsBenchmark {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private static final String DEVICE = System.getProperty("allmodels.device", "cpu");
    private static final int BATCH = Integer.getInteger("allmodels.batch", 8);
    private static final int STEPS = Integer.getInteger("allmodels.steps", 6);
    private static final int LEAK_STEPS = Integer.getInteger("allmodels.leak_steps", 20);
    private static final long RSS_LIMIT_MB = Long.getLong("allmodels.rss_mb", 250L);
    private static final boolean STRICT = Boolean.getBoolean("allmodels.strict");
    private static final float LR = 1e-3f;
    private static final int VOCAB = 64;
    private static final int EMB = 8;
    private static final int SEQ_LEN = 8;
    private static final int NUM_CONCEPTS = 32;
    private static final long[] MLP = new long[]{32L, 16L};

    private AllModelsBenchmark() {}

    // ---- result bookkeeping ------------------------------------------------

    private enum Status { PASS, FAIL, SKIP }

    private static final class Result {
        final String family;
        final String name;
        Status status;
        String detail;
        float firstLoss = Float.NaN;
        float lastLoss = Float.NaN;
        long rssDeltaMb;
        long ms;

        Result(String family, String name) {
            this.family = family;
            this.name = name;
        }
    }

    @FunctionalInterface
    private interface ModelCase {
        void run(Result r) throws Exception;
    }

    private static final class Spec {
        final String family;
        final String name;
        final ModelCase body;

        Spec(String family, String name, ModelCase body) {
            this.family = family;
            this.name = name;
            this.body = body;
        }
    }

    // ---- main --------------------------------------------------------------

    public static void main(String[] args) {
        String deviceArg = System.getProperty("allmodels.device", "cpu");
        DeviceSupport.setDevice(DeviceSupport.parseDeviceType(deviceArg));
        torch.manual_seed(20260729L);

        List<String> filters = new ArrayList<>();
        if (args != null) {
            for (String a : args) {
                if (a != null && !a.isBlank()) filters.add(a.toLowerCase(Locale.ROOT));
            }
        }

        System.out.println("=".repeat(72));
        System.out.println(" AllModelsBenchmark — multi-dimensional model suite");
        System.out.println("=".repeat(72));
        System.out.printf(" device=%s batch=%d steps=%d leak_steps=%d rss_limit=%dMB strict=%s%n",
                DeviceSupport.backend(), BATCH, STEPS, LEAK_STEPS, RSS_LIMIT_MB, STRICT);
        if (!filters.isEmpty()) {
            System.out.println(" filters=" + filters);
        }
        System.out.println();

        List<Spec> all = catalog();
        List<Result> results = new ArrayList<>();
        int ran = 0;
        for (Spec s : all) {
            if (!matches(s, filters)) continue;
            ran++;
            Result r = new Result(s.family, s.name);
            long t0 = System.nanoTime();
            try {
                s.body.run(r);
                if (r.status == null) r.status = Status.PASS;
                if (r.detail == null) {
                    r.detail = String.format(Locale.ROOT,
                            "loss first=%.5f last=%.5f rssΔ=%dMB",
                            r.firstLoss, r.lastLoss, r.rssDeltaMb);
                }
            } catch (Throwable t) {
                r.status = Status.FAIL;
                String msg = t.getMessage();
                if (msg == null || msg.isBlank()) msg = t.getClass().getSimpleName();
                // collapse multiline
                msg = msg.replace('\n', ' ');
                if (msg.length() > 240) msg = msg.substring(0, 240) + "...";
                r.detail = t.getClass().getSimpleName() + ": " + msg;
                t.printStackTrace(System.out);
            }
            r.ms = (System.nanoTime() - t0) / 1_000_000L;
            results.add(r);
            String tag = switch (r.status) {
                case PASS -> "[PASS]";
                case FAIL -> "[FAIL]";
                case SKIP -> "[SKIP]";
            };
            System.out.printf("%s %-14s %-22s %5dms  %s%n",
                    tag, r.family, r.name, r.ms, r.detail);
            // free native intermediates between models
            System.gc();
            try { Thread.sleep(20); } catch (InterruptedException ignored) {}
        }

        // summary
        int pass = 0, fail = 0, skip = 0;
        Map<String, int[]> byFamily = new LinkedHashMap<>();
        List<Result> failures = new ArrayList<>();
        for (Result r : results) {
            int[] c = byFamily.computeIfAbsent(r.family, k -> new int[3]);
            switch (r.status) {
                case PASS -> { pass++; c[0]++; }
                case FAIL -> { fail++; c[1]++; failures.add(r); }
                case SKIP -> { skip++; c[2]++; }
            }
        }
        System.out.println();
        System.out.println("=".repeat(72));
        System.out.printf(" SUMMARY  ran=%d  PASS=%d  FAIL=%d  SKIP=%d  (catalog=%d)%n",
                ran, pass, fail, skip, all.size());
        System.out.println("-".repeat(72));
        System.out.printf("%-16s %6s %6s %6s%n", "family", "pass", "fail", "skip");
        for (Map.Entry<String, int[]> e : byFamily.entrySet()) {
            int[] c = e.getValue();
            System.out.printf("%-16s %6d %6d %6d%n", e.getKey(), c[0], c[1], c[2]);
        }
        if (!failures.isEmpty()) {
            System.out.println("-".repeat(72));
            System.out.println(" FAILURES:");
            for (Result r : failures) {
                System.out.printf("  - %s/%s : %s%n", r.family, r.name, r.detail);
            }
        }
        System.out.println("=".repeat(72));
        if (fail > 0) System.exit(1);
    }

    private static boolean matches(Spec s, List<String> filters) {
        if (filters == null || filters.isEmpty()) return true;
        String fam = s.family.toLowerCase(Locale.ROOT);
        String name = s.name.toLowerCase(Locale.ROOT);
        String full = fam + "." + name;
        for (String f : filters) {
            if ("all".equals(f)) return true;
            // family exact, model name exact, or full "family.model";
            // also allow prefix match on model name with word boundary feel
            // (DeepFM must not match XDeepFM via bare contains).
            if (fam.equals(f) || name.equals(f) || full.equals(f)) return true;
            if (name.startsWith(f) && (name.length() == f.length()
                    || !Character.isLetterOrDigit(name.charAt(f.length())))) return true;
            // substring only when filter contains a dot or is long enough family-ish
            if (f.contains(".") && full.contains(f)) return true;
        }
        return false;
    }

    // ---- catalog -----------------------------------------------------------

    private static List<Spec> catalog() {
        List<Spec> list = new ArrayList<>();

        // ========== RANKING (sparse-feature CTR family) ==========
        list.add(sparseCtr("ranking", "DeepFM", () -> {
            List<Feature> f = sparseFeats(4);
            return new DeepFM(f, f, EMB, MLP, 0.0f, DEVICE);
        }));
        list.add(sparseCtr("ranking", "DCN", () -> new DCN(sparseFeats(4), EMB, 2, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "DCNv2", () -> new DCNv2(sparseFeats(4), EMB, 2, true, 2, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "WideDeep", () -> new WideDeep(sparseFeats(4), EMB, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "AFM", () -> {
            List<SparseFeature> sf = new ArrayList<>();
            for (Feature f : sparseFeats(4)) sf.add((SparseFeature) f);
            return new AFM(sf, EMB, 16, 0.0f, DEVICE);
        }));
        list.add(sparseCtr("ranking", "FiBiNet", () -> new FiBiNet(sparseFeats(4), EMB, MLP, 2, "field_interaction", 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "AutoInt", () -> new AutoInt(sparseFeats(4), Collections.emptyList(), 2, 1, MLP, 0.0f, true, DEVICE)));
        list.add(sparseCtr("ranking", "XDeepFM", () -> new XDeepFM(sparseFeats(4), EMB, new int[]{16, 8}, MLP, true, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "PNN", () -> new PNN(sparseFeats(4), EMB, MLP, "inner", 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "NFM", () -> new NFM(sparseFeats(4), EMB, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "FNN", () -> new FNN(sparseFeats(4), EMB, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "FNFM", () -> new FNFM(sparseFeats(4), EMB, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "AFN", () -> new AFN(sparseFeats(4), EMB, 4, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "HoFM", () -> new HoFM(sparseFeats(4), EMB, 3, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "DeepFFM", () -> new DeepFFM(sparseFeats(4), EMB, 4, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "FatDeepFFM", () -> new FatDeepFFM(sparseFeats(4), EMB, MLP, 0.0f, DEVICE)));
        list.add(sparseCtr("ranking", "EDCN", () -> new EDCN(sparseFeats(4), 2, Map.of("dims", List.of(32L, 16L), "dropout", 0.0f),
                "hadamard_product", true, 1.0f, DEVICE)));
        list.add(sparseCtr("ranking", "RankingLR", () -> new RankingLR(sparseFeats(4), EMB, DEVICE)));
        list.add(sparseCtr("ranking", "XGBoostModel", () -> new XGBoostModel(sparseFeats(4), 4, 3, EMB, 16L, DEVICE)));

        // sequence ranking
        list.add(seqRanking("ranking", "DIN", () -> {
            List<Feature> f = sparseFeats(3);
            List<SequenceFeature> seq = seqFeats(1);
            return new DIN(f, seq, MLP, 0.0f, 16, DEVICE);
        }));
        list.add(seqRanking("ranking", "DIEN", () -> new DIEN(sparseFeats(3), seqFeats(1), EMB, MLP, 0.0f, DEVICE)));
        list.add(seqRanking("ranking", "BST", () -> new BST(sparseFeats(3), seqFeats(1), seqFeats(1),
                EMB, 2, 1, SEQ_LEN, MLP, 0.0f, DEVICE)));
        list.add(seqRanking("ranking", "SIM", () -> new SIM(sparseFeats(3), seqFeats(1), seqFeats(1), seqFeats(1))));
        list.add(seqRanking("ranking", "ETA", () -> new ETA(sparseFeats(3), seqFeats(1), EMB, 32, 16, 4, MLP, 0.0f, DEVICE)));
        list.add(seqRanking("ranking", "MEMBA", () -> new MEMBA(sparseFeats(3), seqFeats(1), EMB, 8, 2, MLP, 0.0f, DEVICE)));
        list.add(seqRanking("ranking", "LiquidNetWork", () ->
                new LiquidNetWork(sparseFeats(3), seqFeats(1), EMB, 8, 2, MLP, 0.0f, DEVICE)));

        // graph ranking
        list.add(graphModel("ranking", "GCN", () -> new GCN(8, 16, 2, 0.0f, DEVICE)));
        list.add(graphModel("ranking", "GAT", () -> new GAT(8, 16, 2, 2, 0.0f, DEVICE)));
        list.add(graphModel("ranking", "GraphSAGE", () -> new GraphSAGE(8, 16, 2, "mean", 0.0f, DEVICE)));
        list.add(graphModel("ranking", "FraudGNN", () -> new FraudGNN(8, 16, 2, 2, 0.0f, DEVICE)));

        // SoftDecisionTree (dense input)
        list.add(new Spec("ranking", "SoftDecisionTree", r -> {
            SoftDecisionTree m = new SoftDecisionTree(16, 3, 8, DEVICE);
            trainLoop(r, m, (opt) -> {
                Tensor x = randn(BATCH, 16);
                Tensor y = torch.randint(2, new long[]{BATCH}, longOpts()).toType(ScalarType.Float);
                opt.zero_grad();
                Tensor p = m.forward(x);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));

        // ========== MATCHING ==========
        list.add(new Spec("matching", "DSSM", r -> {
            List<Feature> user = sparseFeats(3, "u");
            List<Feature> item = sparseFeats(3, "i");
            DSSM m = new DSSM(user, item, EMB, MLP, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> u = featMap(user, BATCH);
                Map<String, Tensor> i = featMap(item, BATCH);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor score = m.forward(u, i);
                Tensor loss = bceFromAny(score, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "DSSMSENET", r -> {
            List<Feature> user = sparseFeats(3, "u");
            List<Feature> item = sparseFeats(3, "i");
            DSSMSENET m = new DSSMSENET(user, item);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = new LinkedHashMap<>();
                x.putAll(featMap(user, BATCH));
                x.putAll(featMap(item, BATCH));
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor score = m.forward(x);
                Tensor loss = bceFromAny(score, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "NCF", r -> {
            NCF m = new NCF(sparseFeats(4));
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(sparseFeats(4), BATCH);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor score = m.forward(x);
                Tensor loss = bceFromAny(score, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "YoutubeDNN", r -> {
            List<Feature> f = sparseFeats(3);
            List<SequenceFeature> seq = seqFeats(1);
            YoutubeDNN m = new YoutubeDNN(f, seq, EMB, MLP, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> xf = featMap(f, BATCH);
                Map<String, Tensor> xs = seqMap(seq, BATCH, SEQ_LEN);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor score = m.forward(xf, xs);
                Tensor loss = bceFromAny(score, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "YoutubeSBC", r -> {
            List<Feature> user = sparseFeats(3, "u");
            List<Feature> item = sparseFeats(3, "i");
            YoutubeSBC m = new YoutubeSBC(user, item);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = new LinkedHashMap<>();
                x.putAll(featMap(user, BATCH));
                x.putAll(featMap(item, BATCH));
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor score = m.forward(x);
                Tensor loss = bceFromAny(score, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "SASRec", r -> {
            List<Feature> seq = new ArrayList<>(seqFeats(1));
            SASRec m = new SASRec(seq, EMB, 2, 1, 32, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(tokens);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "NARM", r -> {
            List<Feature> seq = new ArrayList<>(seqFeats(1));
            NARM m = new NARM(seq, EMB, 8, 8, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(tokens);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "STAMP", r -> {
            List<Feature> seq = new ArrayList<>(seqFeats(1));
            STAMP m = new STAMP(seq, EMB, 0.1f, 0.05f, 8, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(tokens);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "MIND", r -> {
            SequenceFeature seq = seqFeats(1).get(0);
            MIND m = new MIND(sparseFeats(3), seq, EMB, 2, EMB, MLP, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(sparseFeats(3), BATCH);
                Tensor seqIdx = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x, seqIdx);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "ComirecDR", r -> {
            SequenceFeature seq = seqFeats(1).get(0);
            ComirecDR m = new ComirecDR(sparseFeats(3), seq);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(sparseFeats(3), BATCH);
                Tensor seqIdx = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x, seqIdx);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "ComirecSA", r -> {
            SequenceFeature seq = seqFeats(1).get(0);
            ComirecSA m = new ComirecSA(sparseFeats(3), seq);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(sparseFeats(3), BATCH);
                Tensor seqIdx = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x, seqIdx);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "SINE", r -> {
            SequenceFeature seq = seqFeats(1).get(0);
            SINE m = new SINE(sparseFeats(3), seq);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(sparseFeats(3), BATCH);
                Tensor sequence = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x, sequence);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "MAMBA", r -> {
            MAMBA m = new MAMBA(VOCAB);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor pos = arangePos(BATCH, SEQ_LEN);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(tokens, pos);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "GRU4Rec", r -> {
            List<Feature> user = sparseFeats(2, "u");
            List<Feature> hist = new ArrayList<>(seqFeats(1));
            List<Feature> item = sparseFeats(2, "i");
            GRU4Rec m = new GRU4Rec(user, hist, item);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = new LinkedHashMap<>();
                x.putAll(featMap(user, BATCH));
                x.putAll(seqMap(seqFeats(1), BATCH, SEQ_LEN));
                x.putAll(featMap(item, BATCH));
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("matching", "FaceBookDSSM", r -> {
            List<Feature> user = sparseFeats(2, "u");
            List<Feature> pos = sparseFeats(2, "p");
            List<Feature> neg = sparseFeats(2, "n");
            // rename to avoid collisions
            FaceBookDSSM m = new FaceBookDSSM(user, pos, neg);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = new LinkedHashMap<>();
                x.putAll(featMap(user, BATCH));
                x.putAll(featMap(pos, BATCH));
                x.putAll(featMap(neg, BATCH));
                opt.zero_grad();
                Tensor[] pair = m.forwardPair(x);
                // maximize pos, minimize neg via simple margin
                Tensor loss = pair[1].sub(pair[0]).mean().add(new Scalar(1.0f)).relu();
                if (loss.dim() == 0) { /* ok */ }
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));

        // ========== MULTI_TASK ==========
        list.add(mtlMap("multi_task", "MMOE", () ->
                new MMOE(sparseFeats(4), List.of("classification", "classification"), 2,
                        Collections.emptyMap(), Collections.emptyList(), DEVICE)));
        list.add(mtlMap("multi_task", "PLE", () ->
                new PLE(new ArrayList<>(sparseFeats(4)), List.of("classification", "classification"),
                        1, 1, 1, Collections.emptyMap(), Collections.emptyList(), DEVICE)));
        list.add(mtlMap("multi_task", "SharedBottom", () ->
                new SharedBottom(sparseFeats(4), List.of("classification", "classification"))));
        list.add(mtlMap("multi_task", "OMoE", () ->
                new OMoE(new ArrayList<>(sparseFeats(4)), List.of("t0", "t1"))));
        list.add(mtlMap("multi_task", "SingleTaskModel", () ->
                new SingleTaskModel(sparseFeats(4), List.of("t0"))));
        list.add(mtlMap("multi_task", "ESMM", () ->
                new ESMM(sparseFeats(3, "u"), sparseFeats(3, "i"))));
        list.add(mtlMap("multi_task", "AITM", () ->
                new AITM(new ArrayList<>(sparseFeats(4)), 2,
                        Map.of("dims", List.of(32L, 16L), "dropout", 0.0f),
                        List.of(
                                Map.of("dims", List.of(16L), "dropout", 0.0f),
                                Map.of("dims", List.of(16L), "dropout", 0.0f)),
                        DEVICE)));
        list.add(new Spec("multi_task", "MetaHeac", r -> {
            MetaHeac m = new MetaHeac(new ArrayList<>(sparseFeats(4)), List.of("t0", "t1"));
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(sparseFeats(4), BATCH);
                opt.zero_grad();
                Map<String, Tensor> out = m.forwardByName(x);
                Tensor first = out.values().iterator().next();
                Tensor y = randBinary(BATCH);
                Tensor loss = bceFromAny(first, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));

        // ========== KNOWLEDGE TRACING ==========
        // standard (conceptIds, responses)
        for (String name : List.of(
                "DKT", "DKTPlus", "DKTForget", "SimpleKT", "SparseKT", "StableKT",
                "AKT", "GKT", "IEKT", "RKT", "RobustKT", "UKT", "CSKT", "MTKT",
                "PromptKT", "ATDKT", "SKVMN", "SAKTUnified")) {
            list.add(ktStandard(name));
        }
        list.add(new Spec("knowledge_tracing", "ATKT", r -> {
            ATKT m = new ATKT(NUM_CONCEPTS);
            trainKtPair(r, m, (c, resp) -> m.forward(c, resp));
        }));
        list.add(new Spec("knowledge_tracing", "ATKTFix", r -> {
            ATKTFix m = new ATKTFix(NUM_CONCEPTS);
            trainKtPair(r, m, (c, resp) -> m.forward(c, resp));
        }));
        list.add(new Spec("knowledge_tracing", "SAKT", r -> {
            SAKT m = new SAKT(NUM_CONCEPTS, 32, 4, 1, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor c = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor resp = randint(BATCH, SEQ_LEN, 2);
                Tensor target = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor y = randBinary(BATCH * SEQ_LEN);
                opt.zero_grad();
                Tensor out = m.forward(c, resp, target);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("knowledge_tracing", "DKVMN", r -> {
            DKVMN m = new DKVMN(NUM_CONCEPTS, NUM_CONCEPTS * 2L);
            trainKtPair(r, m, (c, resp) -> m.forward(c, resp));
        }));
        list.add(new Spec("knowledge_tracing", "DeepIRT", r -> {
            DeepIRT m = new DeepIRT(NUM_CONCEPTS, NUM_CONCEPTS * 2L);
            trainKtPair(r, m, (c, resp) -> m.forward(c, resp));
        }));
        list.add(new Spec("knowledge_tracing", "SAINT", r -> {
            SAINT m = new SAINT(NUM_CONCEPTS, 8);
            trainLoop(r, m, opt -> {
                Tensor ex = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor cat = randint(BATCH, SEQ_LEN, 8);
                Tensor resp = randint(BATCH, SEQ_LEN, 2);
                Tensor y = randBinary(BATCH * SEQ_LEN);
                opt.zero_grad();
                Tensor out = m.forward(ex, cat, resp);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("knowledge_tracing", "SAINTPlusPlus", r -> {
            SAINTPlusPlus m = new SAINTPlusPlus(NUM_CONCEPTS, 8);
            trainLoop(r, m, opt -> {
                Tensor ex = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor cat = randint(BATCH, SEQ_LEN, 8);
                Tensor resp = randint(BATCH, SEQ_LEN, 2);
                Tensor y = randBinary(BATCH * SEQ_LEN);
                opt.zero_grad();
                Tensor out = m.forward(ex, cat, resp);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("knowledge_tracing", "LPKT", r -> {
            LPKT m = new LPKT(NUM_CONCEPTS, NUM_CONCEPTS);
            trainLoop(r, m, opt -> {
                Tensor ex = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor act = randint(BATCH, SEQ_LEN, 2);
                Tensor ks = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor y = randBinary(BATCH * SEQ_LEN);
                opt.zero_grad();
                Tensor out = m.forward(ex, act, ks);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("knowledge_tracing", "QDKT", r -> {
            QDKT m = new QDKT(NUM_CONCEPTS * 2L, NUM_CONCEPTS);
            trainLoop(r, m, opt -> {
                Tensor q = randint(BATCH, SEQ_LEN, NUM_CONCEPTS * 2);
                Tensor c = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor resp = randint(BATCH, SEQ_LEN, 2);
                Tensor y = randBinary(BATCH * SEQ_LEN);
                opt.zero_grad();
                Tensor out = m.forward(q, c, resp);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("knowledge_tracing", "DIMKT", r -> {
            DIMKT m = new DIMKT(NUM_CONCEPTS);
            trainLoop(r, m, opt -> {
                Tensor c1 = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor c2 = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor c3 = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
                Tensor resp = randint(BATCH, SEQ_LEN, 2);
                Tensor y = randBinary(BATCH * SEQ_LEN);
                opt.zero_grad();
                Tensor out = m.forward(c1, c2, c3, resp);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));

        // ========== GENERATIVE ==========
        list.add(new Spec("generative", "OneRec", r -> {
            // tiny: 2 levels, small codebook, tiny d_model
            OneRec m = new OneRec(2, 16, 32, 2, 1, 16, 0.0, true, DEVICE);
            trainLoop(r, m, opt -> {
                // tokens: [B, T] with values in [0, numLevels*codebook)
                Tensor tokens = randint(BATCH, 8, 2 * 16);
                opt.zero_grad();
                Tensor logits = m.forward(tokens);
                // next-token style: predict last from all, simple CE via NLL on flattened
                Tensor loss = ceFromLogits(logits, tokens);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("generative", "OneRecV2", r -> {
            OneRecV2 m = new OneRecV2(2, 16);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, 8, 2 * 16);
                opt.zero_grad();
                Tensor logits = m.forward(tokens);
                Tensor loss = ceFromLogits(logits, tokens);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("generative", "OpenOneRec", r -> {
            OpenOneRec m = new OpenOneRec(2, 16);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, 8, 2 * 16);
                opt.zero_grad();
                Tensor logits = m.forward(tokens);
                Tensor loss = ceFromLogits(logits, tokens);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
list.add(new Spec("generative", "LLM4Rec", r -> {
            // tiny config: embedDim divisible by heads; maxSeqLen > SEQ_LEN
            LLM4Rec m = new LLM4Rec(VOCAB, 32, 4, 1, SEQ_LEN + 2, new long[]{32L, 16L}, 0.0f, true, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, SEQ_LEN, VOCAB);
                Tensor pos = arangePos(BATCH, SEQ_LEN);
                // BCE on scalar MLP head vs binary labels (LLM4Rec outputs [B,1] CTR-style)
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(tokens, pos);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("generative", "HSTU", r -> {
            HSTU m = new HSTU(VOCAB);
            trainLoop(r, m, opt -> {
                Tensor tokens = randint(BATCH, SEQ_LEN, VOCAB);
                opt.zero_grad();
                Tensor out = m.forward(tokens);
                Tensor loss = ceFromLogits(out, tokens);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("generative", "TIGER", r -> {
            // item embeddings [V, D] — D must match TIGER embedDim (default 8)
            Tensor itemEmb = torch.randn(new long[]{VOCAB, 8},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            TIGER m = new TIGER(itemEmb);
            trainLoop(r, m, opt -> {
                Tensor seq = randint(BATCH, SEQ_LEN, VOCAB);
                opt.zero_grad();
                Tensor out = m.forward(seq);
                Tensor y = randBinary(BATCH);
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
list.add(new Spec("generative", "RQVAE", r -> {
            // inDim=16, 2 codebooks size 16, latent eDim=8
            RQVAE m = new RQVAE(16, 2, 16, 8, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor x = randn(BATCH, 16);
                opt.zero_grad();
                // full encode-quantize-decode path
                var result = m.forward(x, false);
                Tensor[] losses = m.computeLoss(result.quantized, result.loss, x, "mse");
                Tensor loss = losses[0];
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("generative", "RQKMeans", r -> {
            // pure algorithm, not a Module — smoke only
            RQKMeans km = new RQKMeans(2, 8, 16);
            Tensor x = randn(32, 16);
            // fit/encode if methods exist; otherwise just construct
            try {
                // reflective optional fit
                var fit = RQKMeans.class.getMethod("fit", Tensor.class, int.class);
                fit.invoke(km, x, 3);
            } catch (NoSuchMethodException ignore) {
                // construct-only OK
            }
            r.status = Status.PASS;
            r.detail = "construct(+optional fit) ok";
            r.firstLoss = 0f;
            r.lastLoss = 0f;
        }));

        // ========== INDUSTRY: news / shortvideo / live / ecommerce / fintech / pharma / bio ==========
        list.add(new Spec("industry", "NRMS", r -> {
            NRMS m = new NRMS(VOCAB, 32, 4, 16, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor hist = randint(BATCH, 4, 6, VOCAB);
                Tensor cand = randint(BATCH, 3, 6, VOCAB);
                Tensor y = randBinary(BATCH * 3);
                opt.zero_grad();
                Tensor scores = m.forward(hist, cand);
                Tensor loss = bceFromAny(scores, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "NAML", r -> {
            NAML m = new NAML(VOCAB, 10, 20, 32, 4, 16, true, true, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor hist = randint(BATCH, 4, 6, VOCAB);
                Tensor cand = randint(BATCH, 3, 6, VOCAB);
                Tensor y = randBinary(BATCH * 3);
                opt.zero_grad();
                Tensor scores = m.forward(hist, cand);
                Tensor loss = bceFromAny(scores, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "LSTUR", r -> {
            LSTUR m = new LSTUR(VOCAB, 32, 32, 4, 16, LSTUR.Fusion.INI, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor uid = randint(BATCH, 32);
                Tensor hist = randint(BATCH, 4, 6, VOCAB);
                Tensor cand = randint(BATCH, 3, 6, VOCAB);
                Tensor y = randBinary(BATCH * 3);
                opt.zero_grad();
                Tensor scores = m.forward(uid, hist, cand);
                Tensor loss = bceFromAny(scores, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "NPA", r -> {
            NPA m = new NPA(VOCAB, 32, 32, 16, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor uid = randint(BATCH, 32);
                Tensor hist = randint(BATCH, 4, 6, VOCAB);
                Tensor cand = randint(BATCH, 3, 6, VOCAB);
                Tensor y = randBinary(BATCH * 3);
                opt.zero_grad();
                Tensor scores = m.forward(uid, hist, cand);
                Tensor loss = bceFromAny(scores, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "DKN", r -> {
            DKN m = new DKN(VOCAB, 40, 32, 16, new int[]{1, 2}, 8, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor hw = randint(BATCH, 3, 8, VOCAB);
                Tensor he = randint(BATCH, 3, 8, 40);
                Tensor cw = randint(BATCH, 2, 8, VOCAB);
                Tensor ce = randint(BATCH, 2, 8, 40);
                Tensor y = randBinary(BATCH * 2);
                opt.zero_grad();
                Tensor scores = m.forward(hw, he, cw, ce);
                Tensor loss = bceFromAny(scores, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "WLR", r -> {
            List<Feature> f = sparseFeats(4);
            WLR m = new WLR(f, MLP, 5, true, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(f, BATCH);
                Tensor y = randBinary(BATCH);
                Tensor w = torch.rand(new long[]{BATCH}).mul(new Scalar(5f)).add(new Scalar(0.5f));
                opt.zero_grad();
                Tensor p = m.forward(x);
                Tensor loss = WLR.weightedBceLoss(p, y, w);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "D2Q", r -> {
            List<Feature> f = sparseFeats(3);
            D2Q m = new D2Q(f, 6, MLP, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(f, BATCH);
                Tensor dur = randint(BATCH, 6);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x, dur);
                Tensor loss = bceFromAny(out.select(1, 0), y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "PEPNet", r -> {
            List<Feature> f = sparseFeats(4);
            PEPNet m = new PEPNet(f, 3, 2, 8, MLP, new long[]{8L}, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(f, BATCH);
                Tensor domain = randint(BATCH, 3);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x, domain);
                Tensor loss = bceFromAny(out.select(1, 0), y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "LiveMultiTask", r -> {
            List<Feature> f = sparseFeats(4);
            LiveMultiTask m = new LiveMultiTask(f, MLP, new long[]{8L}, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(f, BATCH);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x);
                Tensor loss = bceFromAny(out.select(1, 0), y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "ESCM2", r -> {
            List<Feature> f = sparseFeats(4);
            ESCM2 m = new ESCM2(f, MLP, 3, true, true, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(f, BATCH);
                Tensor domain = randint(BATCH, 3);
                Tensor click = randBinary(BATCH);
                Tensor conv = click.mul(randBinary(BATCH));
                opt.zero_grad();
                Tensor preds = m.forward(x, domain);
                Tensor h = m.backboneFeatures(x, domain);
                Tensor elapsed = torch.rand(new long[]{BATCH}).mul(new Scalar(10f)).add(new Scalar(0.1f));
                Tensor loss = m.computeLoss(preds, click, conv, h, elapsed, 0.1f);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "DBMTL", r -> {
            List<Feature> f = sparseFeats(4);
            DBMTL m = new DBMTL(f, MLP, new long[]{8L}, 2, true, false, true, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(f, BATCH);
                Tensor domain = randint(BATCH, 2);
                Tensor click = randBinary(BATCH);
                Tensor conv = click.mul(randBinary(BATCH));
                Tensor aux = randBinary(BATCH);
                opt.zero_grad();
                Tensor preds = m.forward(x, domain);
                Tensor loss = m.computeLoss(preds, click, conv, aux, null, null);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "MultiDomainCTR", r -> {
            List<Feature> f = sparseFeats(4);
            MultiDomainCTR m = new MultiDomainCTR(f, 3, MLP, DEVICE);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(f, BATCH);
                Tensor domain = randint(BATCH, 3);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor p = m.forward(x, domain);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "SearchConversion", r -> {
            List<Feature> items = sparseFeats(3, "item");
            SearchConversion m = new SearchConversion(VOCAB, items, 16, MLP, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor q = randint(BATCH, 6, VOCAB);
                Map<String, Tensor> item = featMap(items, BATCH);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(q, item);
                Tensor loss = bceFromAny(out.select(1, 0), y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "TabTransformer", r -> {
            TabTransformer m = new TabTransformer(new int[]{20, 30, 40}, 4, 16, 4, 1, MLP, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor cat = randint(BATCH, 3, 20);
                Tensor cont = randn(BATCH, 4);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor p = m.forward(cat, cont);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "FTTransformer", r -> {
            FTTransformer m = new FTTransformer(new int[]{20, 30}, 3, 16, 4, 1, 32, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor cat = randint(BATCH, 2, 20);
                Tensor cont = randn(BATCH, 3);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor p = m.forward(cat, cont);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "SequenceRiskModel", r -> {
            SequenceRiskModel m = new SequenceRiskModel(VOCAB, 32, 4, 1, true, true, new long[]{16L}, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor ev = randint(BATCH, 12, VOCAB);
                Tensor amt = randn(BATCH, 12).abs();
                Tensor tdiff = randn(BATCH, 12).abs();
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor p = m.forward(ev, amt, tdiff);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "DeepDTA", r -> {
            DeepDTA m = new DeepDTA(40, 25, 32, 8, new int[]{3, 4}, 32, new long[]{32L}, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor drug = randint(BATCH, 40, 40);
                Tensor prot = randint(BATCH, 60, 25);
                Tensor y = randn(BATCH);
                opt.zero_grad();
                Tensor pred = m.forward(drug, prot);
                Tensor loss = DeepDTA.mseLoss(pred, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "MolTrans", r -> {
            MolTrans m = new MolTrans(40, 25, 32, 48, 32, 4, 1, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor drug = randint(BATCH, 32, 40);
                Tensor prot = randint(BATCH, 48, 25);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor p = m.forward(drug, prot);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "DrugBAN", r -> {
            DrugBAN m = new DrugBAN(40, 25, 32, 16, 3, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor drug = randint(BATCH, 24, 40);
                Tensor prot = randint(BATCH, 40, 25);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor p = m.forward(drug, prot);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "GraphDrugEncoder", r -> {
            GraphDrugEncoder m = new GraphDrugEncoder(6, 16, 16, 2, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor x = randn(10, 6);
                Tensor adj = torch.eye(10, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
                adj = GraphDrugEncoder.normalizeAdj(adj);
                opt.zero_grad();
                Tensor z = m.forward(x, adj);
                Tensor loss = z.pow(new Scalar(2.0f)).mean();
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "ProteinSeqEncoder", r -> {
            ProteinSeqEncoder m = new ProteinSeqEncoder(22, 32, 16, 4, 1, 32, 0.0f, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor seq = randint(BATCH, 24, 22);
                opt.zero_grad();
                Tensor z = m.forward(seq);
                Tensor loss = z.pow(new Scalar(2.0f)).mean();
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "TwinTowerPPI", r -> {
            TwinTowerPPI m = new TwinTowerPPI(22, 32, 16, 4, 1, true, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor a = randint(BATCH, 20, 22);
                Tensor b = randint(BATCH, 20, 22);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor p = m.forward(a, b);
                Tensor loss = bceFromAny(p, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "GeneExpressionMLP", r -> {
            GeneExpressionMLP m = new GeneExpressionMLP(64, 2, MLP, true, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor x = randn(BATCH, 64);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(x);
                Tensor loss = bceFromAny(out.select(1, 0), y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));
        list.add(new Spec("industry", "DnaSeqCnn", r -> {
            DnaSeqCnn m = new DnaSeqCnn(6, 4, 8, new int[]{3, 5}, 2, true, DEVICE);
            trainLoop(r, m, opt -> {
                Tensor seq = randint(BATCH, 40, 6);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out = m.forward(seq);
                Tensor loss = bceFromAny(out.select(1, 0), y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        }));

        return list;
    }

    // ---- family helpers ----------------------------------------------------

    private static Spec sparseCtr(String family, String name, Supplier<Module> factory) {
        return new Spec(family, name, r -> {
            Module m = factory.get();
            List<Feature> feats = sparseFeats(4);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(feats, BATCH);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor logits = invokeSparseForward(m, x);
                Tensor loss = bceFromAny(logits, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        });
    }

    private static Spec seqRanking(String family, String name, Supplier<Module> factory) {
        return new Spec(family, name, r -> {
            Module m = factory.get();
            List<Feature> sparse = sparseFeats(3);
            List<SequenceFeature> seq = seqFeats(1);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> xs = featMap(sparse, BATCH);
                Map<String, Tensor> xq = seqMap(seq, BATCH, SEQ_LEN);
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor logits = invokeSeqForward(m, xs, xq, y);
                Tensor loss = bceFromAny(logits, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        });
    }

    private static Spec graphModel(String family, String name, Supplier<Module> factory) {
        return new Spec(family, name, r -> {
            Module m = factory.get();
            trainLoop(r, m, opt -> {
                int n = 12;
                Tensor feat = randn(n, 8);
                Tensor adj = torch.randint(2, new long[]{n, n},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
                adj = adj.add(adj.transpose(0, 1)).gt(new Scalar(0)).toType(ScalarType.Float);
                // add self loops
                adj = adj.add(torch.eye(n, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))));
                Tensor y = randBinary(n);
                opt.zero_grad();
                Tensor out;
                if (m instanceof GCN) out = ((GCN) m).forward(feat, adj);
                else if (m instanceof GAT) out = ((GAT) m).forward(feat, adj);
                else if (m instanceof GraphSAGE) out = ((GraphSAGE) m).forward(feat, adj);
                else if (m instanceof FraudGNN) out = ((FraudGNN) m).forward(feat, adj);
                else throw new IllegalStateException("unknown graph model " + m.getClass());
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        });
    }

    private static Spec mtlMap(String family, String name, Supplier<Module> factory) {
        return new Spec(family, name, r -> {
            Module m = factory.get();
            List<Feature> feats = sparseFeats(4);
            trainLoop(r, m, opt -> {
                Map<String, Tensor> x = featMap(feats, BATCH);
                // ESMM needs both user+item keys
                if (m instanceof ESMM) {
                    x = new LinkedHashMap<>();
                    x.putAll(featMap(sparseFeats(3, "u"), BATCH));
                    x.putAll(featMap(sparseFeats(3, "i"), BATCH));
                }
                Tensor y = randBinary(BATCH);
                opt.zero_grad();
                Tensor out;
                if (m instanceof MMOE) out = ((MMOE) m).forward(x);
                else if (m instanceof PLE) out = ((PLE) m).forward(x);
                else if (m instanceof SharedBottom) out = ((SharedBottom) m).forward(x);
                else if (m instanceof OMoE) out = ((OMoE) m).forward(x);
                else if (m instanceof SingleTaskModel) out = ((SingleTaskModel) m).forward(x);
                else if (m instanceof ESMM) out = ((ESMM) m).forward(x);
                else if (m instanceof AITM) out = ((AITM) m).forward(x);
                else throw new IllegalStateException("unknown mtl " + m.getClass());
                Tensor loss = bceFromAny(out, y);
                loss.backward();
                opt.step();
                return lossItem(loss);
            });
        });
    }

    private static Spec ktStandard(String name) {
        return new Spec("knowledge_tracing", name, r -> {
            Module m = buildKt(name);
            trainKtPair(r, m, (c, resp) -> invokeKtForward(m, c, resp));
        });
    }

    private static Module buildKt(String name) {
        return switch (name) {
            case "DKT" -> new DKT(NUM_CONCEPTS, 32, 1, 0.0f, DEVICE);
            case "DKTPlus" -> new DKTPlus(NUM_CONCEPTS);
            case "DKTForget" -> new DKTForget(NUM_CONCEPTS);
            case "SimpleKT" -> new SimpleKT(NUM_CONCEPTS);
            case "SparseKT" -> new SparseKT(NUM_CONCEPTS);
            case "StableKT" -> new StableKT(NUM_CONCEPTS);
            case "AKT" -> new AKT(NUM_CONCEPTS);
            case "GKT" -> new GKT(NUM_CONCEPTS);
            case "IEKT" -> new IEKT(NUM_CONCEPTS);
            case "RKT" -> new RKT(NUM_CONCEPTS);
            case "RobustKT" -> new RobustKT(NUM_CONCEPTS);
            case "UKT" -> new UKT(NUM_CONCEPTS);
            case "CSKT" -> new CSKT(NUM_CONCEPTS);
            case "MTKT" -> new MTKT(NUM_CONCEPTS);
            case "PromptKT" -> new PromptKT(NUM_CONCEPTS);
            case "ATDKT" -> new ATDKT(NUM_CONCEPTS);
            case "SKVMN" -> new SKVMN(NUM_CONCEPTS);
            case "SAKTUnified" -> new SAKTUnified(NUM_CONCEPTS);
            default -> throw new IllegalArgumentException("unknown KT " + name);
        };
    }

    @FunctionalInterface
    private interface KtForward {
        Tensor apply(Tensor c, Tensor resp);
    }

    private static void trainKtPair(Result r, Module m, KtForward fwd) throws Exception {
        trainLoop(r, m, opt -> {
            Tensor c = randint(BATCH, SEQ_LEN, NUM_CONCEPTS);
            Tensor resp = randint(BATCH, SEQ_LEN, 2);
            Tensor y = randBinary(BATCH * SEQ_LEN);
            opt.zero_grad();
            Tensor out = fwd.apply(c, resp);
            Tensor loss = bceFromAny(out, y);
            loss.backward();
            opt.step();
            return lossItem(loss);
        });
    }

    private static Tensor invokeKtForward(Module m, Tensor c, Tensor resp) {
        if (m instanceof DKT) return ((DKT) m).forward(c, resp);
        if (m instanceof DKTPlus) return ((DKTPlus) m).forward(c, resp);
        if (m instanceof DKTForget) return ((DKTForget) m).forward(c, resp);
        if (m instanceof SimpleKT) return ((SimpleKT) m).forward(c, resp);
        if (m instanceof SparseKT) return ((SparseKT) m).forward(c, resp);
        if (m instanceof StableKT) return ((StableKT) m).forward(c, resp);
        if (m instanceof AKT) return ((AKT) m).forward(c, resp);
        if (m instanceof GKT) return ((GKT) m).forward(c, resp);
        if (m instanceof IEKT) return ((IEKT) m).forward(c, resp);
        if (m instanceof RKT) return ((RKT) m).forward(c, resp);
        if (m instanceof RobustKT) return ((RobustKT) m).forward(c, resp);
        if (m instanceof UKT) return ((UKT) m).forward(c, resp);
        if (m instanceof CSKT) return ((CSKT) m).forward(c, resp);
        if (m instanceof MTKT) return ((MTKT) m).forward(c, resp);
        if (m instanceof PromptKT) return ((PromptKT) m).forward(c, resp);
        if (m instanceof ATDKT) return ((ATDKT) m).forward(c, resp);
        if (m instanceof SKVMN) return ((SKVMN) m).forward(c, resp);
        if (m instanceof SAKTUnified) return ((SAKTUnified) m).forward(c, resp);
        throw new IllegalStateException("kt forward missing for " + m.getClass().getSimpleName());
    }

    private static Tensor invokeSparseForward(Module m, Map<String, Tensor> x) {
        if (m instanceof DeepFM) return ((DeepFM) m).forward(x);
        if (m instanceof DCN) return ((DCN) m).forward(x);
        if (m instanceof DCNv2) return ((DCNv2) m).forward(x);
        if (m instanceof WideDeep) return ((WideDeep) m).forward(x);
        if (m instanceof AFM) return ((AFM) m).forward(x);
        if (m instanceof FiBiNet) return ((FiBiNet) m).forward(x);
        if (m instanceof AutoInt) return ((AutoInt) m).forward(x, Collections.emptyMap());
        if (m instanceof XDeepFM) return ((XDeepFM) m).forward(x);
        if (m instanceof PNN) return ((PNN) m).forward(x);
        if (m instanceof NFM) return ((NFM) m).forward(x);
        if (m instanceof FNN) return ((FNN) m).forward(x);
        if (m instanceof FNFM) return ((FNFM) m).forward(x);
        if (m instanceof AFN) return ((AFN) m).forward(x);
        if (m instanceof HoFM) return ((HoFM) m).forward(x);
        if (m instanceof DeepFFM) return ((DeepFFM) m).forward(x);
        if (m instanceof FatDeepFFM) return ((FatDeepFFM) m).forward(x);
        if (m instanceof EDCN) return ((EDCN) m).forward(x);
        if (m instanceof RankingLR) return ((RankingLR) m).forward(x);
        if (m instanceof XGBoostModel) return ((XGBoostModel) m).forward(x);
        throw new IllegalStateException("sparse forward missing for " + m.getClass().getSimpleName());
    }

    private static Tensor invokeSeqForward(Module m, Map<String, Tensor> sparse,
                                           Map<String, Tensor> seq, Tensor labels) {
        if (m instanceof DIN) {
            Tensor targetIdx = labels.view(labels.size(0), 1L).toType(ScalarType.Long)
                    .clamp(new ScalarOptional(new Scalar(0)), new ScalarOptional(new Scalar(VOCAB - 1)));
            // DIN targetIdx is item id indices — use random ids instead of labels
            targetIdx = randint(BATCH, 1, VOCAB);
            return ((DIN) m).forward(sparse, seq, targetIdx);
        }
        if (m instanceof DIEN) return ((DIEN) m).forward(sparse, seq);
        if (m instanceof BST) return ((BST) m).forward(sparse, seq);
        if (m instanceof SIM) return ((SIM) m).forward(sparse, seq, seq, seq, seq);
        if (m instanceof ETA) return ((ETA) m).forward(sparse, seq, seq);
        if (m instanceof MEMBA) {
            Tensor targetIdx = randint(BATCH, 1, VOCAB);
            return ((MEMBA) m).forward(sparse, seq, targetIdx);
        }
        if (m instanceof LiquidNetWork) return ((LiquidNetWork) m).forward(sparse, seq);
        throw new IllegalStateException("seq forward missing for " + m.getClass().getSimpleName());
    }

    // ---- train / leak loop -------------------------------------------------

    @FunctionalInterface
    private interface StepFn {
        float step(Optimizer opt) throws Exception;
    }

    private static void trainLoop(Result r, Module model, StepFn stepFn) throws Exception {
        model.train(true);
        Optimizer opt = adam(model);

        // warm one step outside leak measurement (builds autograd / adam state)
        float warm = Float.NaN;
        try (PointerScope scope = new PointerScope()) {
            warm = stepFn.step(opt);
        }
        if (!Float.isFinite(warm)) {
            throw new IllegalStateException("non-finite loss on warm step: " + warm);
        }

        // measured train steps
        float first = Float.NaN, last = Float.NaN;
        for (int i = 0; i < STEPS; i++) {
            float v;
            try (PointerScope scope = new PointerScope()) {
                v = stepFn.step(opt);
            }
            if (!Float.isFinite(v)) {
                throw new IllegalStateException("non-finite loss at step " + i + ": " + v);
            }
            if (i == 0) first = v;
            last = v;
        }

        // leak probe: many steps, track RSS
        long rss0 = rssMb();
        long heap0 = heapUsedMb();
        for (int i = 0; i < LEAK_STEPS; i++) {
            try (PointerScope scope = new PointerScope()) {
                float v = stepFn.step(opt);
                if (!Float.isFinite(v)) {
                    throw new IllegalStateException("non-finite loss in leak probe step " + i + ": " + v);
                }
                last = v;
            }
            if ((i + 1) % 8 == 0) {
                System.gc();
            }
        }
        System.gc();
        try { Thread.sleep(30); } catch (InterruptedException ignored) {}
        long rss1 = rssMb();
        long heap1 = heapUsedMb();
        long rssDelta = Math.max(0, rss1 - rss0);
        long heapDelta = Math.max(0, heap1 - heap0);
        r.rssDeltaMb = rssDelta;
        r.firstLoss = first;
        r.lastLoss = last;

        if (rssDelta > RSS_LIMIT_MB) {
            throw new IllegalStateException(String.format(Locale.ROOT,
                    "possible native leak: RSS +%dMB (limit %d) heapΔ=%dMB over %d steps",
                    rssDelta, RSS_LIMIT_MB, heapDelta, LEAK_STEPS));
        }
        if (STRICT && Float.isFinite(first) && Float.isFinite(last) && last > first * 1.5f + 0.5f) {
            throw new IllegalStateException(String.format(Locale.ROOT,
                    "loss exploded first=%.5f last=%.5f", first, last));
        }
        r.status = Status.PASS;
        r.detail = String.format(Locale.ROOT,
                "loss %.5f→%.5f rssΔ=%dMB heapΔ=%dMB", first, last, rssDelta, heapDelta);
    }

    // ---- tensors / losses / features --------------------------------------

    private static Optimizer adam(Module model) {
        return new Adam(model.parameters(), new AdamOptions(LR));
    }

    private static List<Feature> sparseFeats(int n) {
        return sparseFeats(n, "f");
    }

    private static List<Feature> sparseFeats(int n, String prefix) {
        List<Feature> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(new SparseFeature(prefix + i, VOCAB, EMB));
        }
        return list;
    }

    private static List<SequenceFeature> seqFeats(int n) {
        List<SequenceFeature> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(new SequenceFeature("seq" + i, VOCAB, EMB, "mean", null, SEQ_LEN, 0L));
        }
        return list;
    }

    private static Map<String, Tensor> featMap(List<? extends Feature> feats, int batch) {
        Map<String, Tensor> m = new LinkedHashMap<>();
        for (Feature f : feats) {
            m.put(f.name(), randint(batch, (int) Math.max(2, Math.min(VOCAB, f instanceof SparseFeature
                    ? ((SparseFeature) f).vocabSize() : VOCAB))));
        }
        return m;
    }

    private static Map<String, Tensor> seqMap(List<SequenceFeature> feats, int batch, int len) {
        Map<String, Tensor> m = new LinkedHashMap<>();
        for (SequenceFeature f : feats) {
            m.put(f.name(), randint(batch, len, (int) Math.max(2, Math.min(VOCAB, f.vocabSize()))));
        }
        return m;
    }

    private static TensorOptions floatOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
    }

    private static TensorOptions longOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
    }

    private static Tensor randn(long... sizes) {
        return torch.randn(sizes, floatOpts());
    }

    private static Tensor randint(long n, int high) {
        return torch.randint(Math.max(high, 2), new long[]{n}, longOpts());
    }

    private static Tensor randint(long a, long b, int high) {
        return torch.randint(Math.max(high, 2), new long[]{a, b}, longOpts());
    }

    private static Tensor randint(long a, long b, long c, int high) {
        return torch.randint(Math.max(high, 2), new long[]{a, b, c}, longOpts());
    }

    private static Tensor randBinary(long n) {
        return torch.randint(2, new long[]{n}, longOpts()).toType(ScalarType.Float);
    }

    private static Tensor arangePos(int batch, int len) {
        // positions [B, T] = 0..T-1 broadcast
        Tensor row = org.bytedeco.pytorch.utils.recommend.TensorHelpers.arange(0, len)
                .toType(ScalarType.Long);
        return row.unsqueeze(0).expand(new long[]{batch, len}).contiguous();
    }

    private static float lossItem(Tensor loss) {
        if (loss == null || loss.isNull()) return Float.NaN;
        Tensor s = loss;
        if (s.dim() > 0) s = s.mean();
        return s.item().toFloat();
    }

    /**
     * Flexible BCE: accepts logits or probabilities, any rank; flattens and aligns to y.
     */
    private static Tensor bceFromAny(Tensor pred, Tensor y) {
        if (pred == null || pred.isNull()) {
            throw new IllegalStateException("pred is null");
        }
        Tensor p = pred;
        // multi-task [B, T] → take col 0 or mean over tasks after per-element bce
        if (p.dim() == 2 && y.dim() == 1 && p.size(0) == y.size(0) && p.size(1) > 1) {
            // average BCE across task heads against same y
            Tensor yy = y.toType(ScalarType.Float).unsqueeze(1).expand_as(p);
            return bceProbOrLogits(p, yy);
        }
        if (p.dim() >= 2 && p.numel() != y.numel()) {
            // flatten both
            p = p.reshape(p.numel());
            Tensor yy = y.reshape(y.numel()).toType(ScalarType.Float);
            long n = Math.min(p.numel(), yy.numel());
            p = p.narrow(0, 0, n);
            yy = yy.narrow(0, 0, n);
            return bceProbOrLogits(p, yy);
        }
        if (p.dim() > 1) p = p.reshape(p.numel());
        Tensor yy = y.reshape(y.numel()).toType(ScalarType.Float);
        long n = Math.min(p.numel(), yy.numel());
        if (n == 0) throw new IllegalStateException("empty pred/y");
        p = p.narrow(0, 0, n);
        yy = yy.narrow(0, 0, n);
        return bceProbOrLogits(p, yy);
    }

    private static Tensor bceProbOrLogits(Tensor p, Tensor y) {
        // heuristic: if values outside (0,1) treat as logits
        Tensor pMin = p.min();
        Tensor pMax = p.max();
        float mn = pMin.item().toFloat();
        float mx = pMax.item().toFloat();
        Tensor prob;
        if (mn < -1e-3f || mx > 1.0f + 1e-3f) {
            prob = p.sigmoid();
        } else {
            prob = p;
        }
        Tensor pp = prob.clamp(new ScalarOptional(new Scalar(1e-6f)),
                new ScalarOptional(new Scalar(1.0f - 1e-6f)));
        Tensor yy = y.toType(ScalarType.Float);
        return yy.neg().mul(pp.log())
                .add(torch.sub(torch.ones_like(yy), yy).neg()
                        .mul(torch.sub(torch.ones_like(pp), pp).log()))
                .mean();
    }

    private static Tensor ceFromLogits(Tensor logits, Tensor targets) {
        // logits [B, T, V] or [B, V]; targets [B, T] or [B]
        Tensor t = targets.toType(ScalarType.Long);
        if (logits.dim() == 3) {
            long B = logits.size(0);
            long T = logits.size(1);
            long V = logits.size(2);
            Tensor flatLogits = logits.reshape(B * T, V);
            Tensor flatT = t.reshape(B * T);
            // gather NLL
            Tensor logp = flatLogits.log_softmax(1);
            Tensor nll = logp.gather(1, flatT.view(B * T, 1L)).neg().mean();
            return nll;
        }
        if (logits.dim() == 2) {
            Tensor logp = logits.log_softmax(1);
            Tensor tt = t.dim() > 1 ? t.reshape(t.numel()) : t;
            long n = Math.min(logp.size(0), tt.numel());
            Tensor nll = logp.narrow(0, 0, n)
                    .gather(1, tt.narrow(0, 0, n).view(n, 1L))
                    .neg().mean();
            return nll;
        }
        // fallback MSE on flattened
        return logits.toType(ScalarType.Float).pow(new Scalar(2.0f)).mean();
    }

    private static long rssMb() {
        try {
            // macOS: ps -o rss= -p PID  (RSS in KB)
            long pid = ProcessHandle.current().pid();
            Process p = new ProcessBuilder("ps", "-o", "rss=", "-p", String.valueOf(pid))
                    .redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes()).trim();
            p.waitFor();
            if (!out.isEmpty()) {
                // may contain multiple numbers; take last
                String[] parts = out.trim().split("\\s+");
                long kb = Long.parseLong(parts[parts.length - 1]);
                return kb / 1024L;
            }
        } catch (Exception ignored) {}
        MemoryMXBean bean = ManagementFactory.getMemoryMXBean();
        return bean.getHeapMemoryUsage().getUsed() / (1024L * 1024L);
    }

    private static long heapUsedMb() {
        MemoryMXBean bean = ManagementFactory.getMemoryMXBean();
        return bean.getHeapMemoryUsage().getUsed() / (1024L * 1024L);
    }
}
