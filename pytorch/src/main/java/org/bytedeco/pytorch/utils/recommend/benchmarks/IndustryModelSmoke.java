/*
 * IndustryModelSmoke — minimal [B, ...] forward shape checks for every industry model.
 *
 * Run:
 *   java org.bytedeco.pytorch.utils.recommend.benchmarks.IndustryModelSmoke
 *
 * Does NOT require real datasets; uses tiny random / synthetic tensors on CPU.
 * Exits non-zero if any model throws or returns an unexpected rank.
 */
package org.bytedeco.pytorch.utils.recommend.benchmarks;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.AdditiveAttention;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.DelayedFeedbackHead;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.DomainAdapter;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.DurationDeconfoundHead;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.GateFusion;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.MultiHeadSelfAttention;
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
import org.bytedeco.pytorch.utils.recommend.models.live.LiveMultiTask;
import org.bytedeco.pytorch.utils.recommend.models.news.DKN;
import org.bytedeco.pytorch.utils.recommend.models.news.LSTUR;
import org.bytedeco.pytorch.utils.recommend.models.news.NAML;
import org.bytedeco.pytorch.utils.recommend.models.news.NPA;
import org.bytedeco.pytorch.utils.recommend.models.news.NRMS;
import org.bytedeco.pytorch.utils.recommend.models.pharma.DeepDTA;
import org.bytedeco.pytorch.utils.recommend.models.pharma.GraphDrugEncoder;
import org.bytedeco.pytorch.utils.recommend.models.pharma.DrugBAN;
import org.bytedeco.pytorch.utils.recommend.models.pharma.MolTrans;
import org.bytedeco.pytorch.utils.recommend.models.shortvideo.D2Q;
import org.bytedeco.pytorch.utils.recommend.models.shortvideo.PEPNet;
import org.bytedeco.pytorch.utils.recommend.models.shortvideo.WLR;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class IndustryModelSmoke {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private static final String DEVICE = "cpu";
    private static final int B = 4;

    private IndustryModelSmoke() {}

    public static void main(String[] args) {
        DeviceSupport.setDevice(DeviceSupport.DeviceType.CPU);
        int failed = 0;
        int passed = 0;
        List<String> errors = new ArrayList<>();

        for (Case c : cases()) {
            try {
                c.run();
                System.out.println("[PASS] " + c.name);
                passed++;
            } catch (Throwable t) {
                failed++;
                String msg = c.name + ": " + t.getClass().getSimpleName() + ": " + t.getMessage();
                errors.add(msg);
                System.out.println("[FAIL] " + msg);
                t.printStackTrace(System.out);
            }
        }

        System.out.println("============================================================");
        System.out.printf("Industry smoke: passed=%d failed=%d total=%d%n",
                passed, failed, passed + failed);
        if (failed > 0) {
            System.out.println("Failures:");
            for (String e : errors) System.out.println("  - " + e);
            System.exit(1);
        }
    }

    private static abstract class Case {
        final String name;
        Case(String name) { this.name = name; }
        public abstract void run();
    }

    private static List<Case> cases() {
        List<Case> list = new ArrayList<>();

        // ---- shared layers ----
        list.add(new Case("AdditiveAttention") {
            @Override public void run() {
                AdditiveAttention m = new AdditiveAttention(32, 16, DEVICE);
                Tensor x = randn(B, 10, 32);
                Tensor y = m.forward(x);
                expectRank(y, 2, "out");
                expectSize(y, 0, B);
                expectSize(y, 1, 32);
            }
        });
        list.add(new Case("MultiHeadSelfAttention") {
            @Override public void run() {
                MultiHeadSelfAttention m = new MultiHeadSelfAttention(32, 4, 0.0f, DEVICE);
                Tensor y = m.forward(randn(B, 8, 32));
                expectRank(y, 3, "out");
                expectSize(y, 2, 32);
            }
        });
        list.add(new Case("GateFusion") {
            @Override public void run() {
                GateFusion m = new GateFusion(16, 8, 0, GateFusion.Mode.MULTIPLICATIVE, DEVICE);
                Tensor y = m.forward(randn(B, 16), randn(B, 8));
                expectSize(y, 1, 16);
            }
        });
        list.add(new Case("DurationDeconfoundHead") {
            @Override public void run() {
                DurationDeconfoundHead m = new DurationDeconfoundHead(32, 10, 8, new long[]{16L}, DEVICE);
                Tensor y = m.forward(randn(B, 32), randint(B, 10));
                expectSize(y, 1, 2);
            }
        });
        list.add(new Case("DelayedFeedbackHead") {
            @Override public void run() {
                DelayedFeedbackHead m = new DelayedFeedbackHead(32, new long[]{16L}, DEVICE);
                Tensor y = m.forward(randn(B, 32));
                expectSize(y, 1, 2);
                Tensor nll = m.delayedFeedbackNll(randn(B, 32), randint(B, 2).toType(ScalarType.Float),
                        randn(B).abs().add(new org.bytedeco.pytorch.Scalar(0.1f)));
                expectRank(nll, 0, "nll");
            }
        });
        list.add(new Case("DomainAdapter") {
            @Override public void run() {
                DomainAdapter m = new DomainAdapter(16, 3, 8, true, DEVICE);
                Tensor y = m.forward(randn(B, 16), randint(B, 3));
                expectSize(y, 1, 16);
            }
        });

        // ---- news ----
        list.add(new Case("NRMS") {
            @Override public void run() {
                NRMS m = new NRMS(200, 32, 4, 16, 0.0f, DEVICE);
                Tensor hist = randint(B, 5, 8, 200);
                Tensor cand = randint(B, 3, 8, 200);
                Tensor y = m.forward(hist, cand);
                expectSize(y, 0, B);
                expectSize(y, 1, 3);
            }
        });
        list.add(new Case("NAML") {
            @Override public void run() {
                NAML m = new NAML(200, 20, 40, 32, 4, 16, true, true, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 5, 8, 200), randint(B, 3, 8, 200));
                expectSize(y, 1, 3);
            }
        });
        list.add(new Case("LSTUR") {
            @Override public void run() {
                LSTUR m = new LSTUR(200, 50, 32, 4, 16, LSTUR.Fusion.INI, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 50), randint(B, 5, 8, 200), randint(B, 3, 8, 200));
                expectSize(y, 1, 3);
            }
        });
        list.add(new Case("NPA") {
            @Override public void run() {
                NPA m = new NPA(200, 50, 32, 16, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 50), randint(B, 5, 8, 200), randint(B, 3, 8, 200));
                expectSize(y, 1, 3);
            }
        });
        list.add(new Case("DKN") {
            @Override public void run() {
                DKN m = new DKN(200, 100, 32, 16, new int[]{1, 2}, 8, 0.0f, DEVICE);
                Tensor y = m.forward(
                        randint(B, 4, 10, 200), randint(B, 4, 10, 100),
                        randint(B, 2, 10, 200), randint(B, 2, 10, 100));
                expectSize(y, 1, 2);
            }
        });

        // ---- short video / live ----
        list.add(new Case("WLR") {
            @Override public void run() {
                List<Feature> feats = sparseFeats(4, 100, 8);
                WLR m = new WLR(feats, new long[]{32L, 16L}, 5, true, DEVICE);
                Tensor y = m.forward(featMap(feats, B, 100));
                expectRank(y, 1, "p");
                Tensor y2 = m.forwardWithDuration(featMap(feats, B, 100), randint(B, 5));
                expectSize(y2, 1, 3);
            }
        });
        list.add(new Case("D2Q") {
            @Override public void run() {
                List<Feature> feats = sparseFeats(3, 50, 8);
                D2Q m = new D2Q(feats, 6, new long[]{32L, 16L}, DEVICE);
                Tensor y = m.forward(featMap(feats, B, 50), randint(B, 6));
                expectSize(y, 1, 2);
            }
        });
        list.add(new Case("PEPNet") {
            @Override public void run() {
                List<Feature> feats = sparseFeats(4, 80, 8);
                PEPNet m = new PEPNet(feats, 3, 2, 8, new long[]{32L, 16L}, new long[]{8L}, DEVICE);
                Tensor y = m.forward(featMap(feats, B, 80), randint(B, 3));
                expectSize(y, 1, 2);
            }
        });
        list.add(new Case("LiveMultiTask") {
            @Override public void run() {
                List<Feature> feats = sparseFeats(4, 60, 8);
                LiveMultiTask m = new LiveMultiTask(feats, new long[]{32L, 16L}, new long[]{8L}, DEVICE);
                Tensor y = m.forward(featMap(feats, B, 60));
                expectSize(y, 1, LiveMultiTask.NUM_OUTPUTS);
            }
        });

        // ---- ecommerce ----
        list.add(new Case("ESCM2") {
            @Override public void run() {
                List<Feature> feats = sparseFeats(5, 100, 8);
                ESCM2 m = new ESCM2(feats, new long[]{32L, 16L}, 3, true, true, DEVICE);
                Tensor y = m.forward(featMap(feats, B, 100), randint(B, 3));
                expectSize(y, 1, ESCM2.NUM_OUTPUTS);
            }
        });
        list.add(new Case("MultiDomainCTR") {
            @Override public void run() {
                List<Feature> feats = sparseFeats(4, 100, 8);
                MultiDomainCTR m = new MultiDomainCTR(feats, 3, new long[]{32L, 16L}, DEVICE);
                Tensor y = m.forward(featMap(feats, B, 100), randint(B, 3));
                expectRank(y, 1, "ctr");
            }
        });
        list.add(new Case("SearchConversion") {
            @Override public void run() {
                List<Feature> items = sparseFeats(3, 50, 8);
                SearchConversion m = new SearchConversion(100, items, 16, new long[]{32L, 16L}, DEVICE);
                Tensor y = m.forward(randint(B, 6, 100), featMap(items, B, 50));
                expectSize(y, 1, 3);
            }
        });

        // ---- fintech ----
        list.add(new Case("TabTransformer") {
            @Override public void run() {
                TabTransformer m = new TabTransformer(new int[]{20, 30, 40}, 4, 16, 4, 1,
                        new long[]{32L}, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 3, 20), randn(B, 4));
                expectRank(y, 1, "p");
            }
        });
        list.add(new Case("FTTransformer") {
            @Override public void run() {
                FTTransformer m = new FTTransformer(new int[]{20, 30}, 3, 16, 4, 1, 32, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 2, 20), randn(B, 3));
                expectRank(y, 1, "p");
            }
        });
        list.add(new Case("SequenceRiskModel") {
            @Override public void run() {
                SequenceRiskModel m = new SequenceRiskModel(100, 32, 4, 1, true, true,
                        new long[]{16L}, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 12, 100), randn(B, 12).abs(), randn(B, 12).abs());
                expectRank(y, 1, "risk");
            }
        });

        // ---- pharma / bio ----
        list.add(new Case("DeepDTA") {
            @Override public void run() {
                DeepDTA m = new DeepDTA(40, 25, 32, 8, new int[]{3, 4}, 32,
                        new long[]{32L}, DEVICE);
                Tensor y = m.forward(randint(B, 40, 40), randint(B, 60, 25));
                expectRank(y, 1, "aff");
            }
        });
        list.add(new Case("MolTrans") {
            @Override public void run() {
                MolTrans m = new MolTrans(40, 25, 32, 48, 32, 4, 1, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 32, 40), randint(B, 48, 25));
                expectRank(y, 1, "p");
            }
        });
        list.add(new Case("DrugBAN") {
            @Override public void run() {
                DrugBAN m = new DrugBAN(40, 25, 32, 16, 3, DEVICE);
                Tensor y = m.forward(randint(B, 24, 40), randint(B, 40, 25));
                expectRank(y, 1, "p");
            }
        });
        list.add(new Case("ProteinSeqEncoder") {
            @Override public void run() {
                ProteinSeqEncoder m = new ProteinSeqEncoder(22, 64, 32, 4, 1, 64, 0.0f, DEVICE);
                Tensor y = m.forward(randint(B, 32, 22));
                expectSize(y, 1, 32);
            }
        });
        list.add(new Case("TwinTowerPPI") {
            @Override public void run() {
                TwinTowerPPI m = new TwinTowerPPI(22, 48, 32, 4, 1, true, DEVICE);
                Tensor y = m.forward(randint(B, 24, 22), randint(B, 24, 22));
                expectRank(y, 1, "p");
            }
        });
        list.add(new Case("GeneExpressionMLP") {
            @Override public void run() {
                GeneExpressionMLP m = new GeneExpressionMLP(100, 2, new long[]{32L, 16L}, true, DEVICE);
                Tensor y = m.forward(randn(B, 100));
                expectSize(y, 1, 2);
            }
        });
        list.add(new Case("DnaSeqCnn") {
            @Override public void run() {
                DnaSeqCnn m = new DnaSeqCnn(6, 4, 8, new int[]{3, 5}, 2, true, DEVICE);
                Tensor y = m.forward(randint(B, 50, 6));
                expectSize(y, 1, 2);
            }
        });
        list.add(new Case("DBMTL") {
            @Override public void run() {
                List<Feature> feats = sparseFeats(4, 80, 8);
                DBMTL m = new DBMTL(feats, new long[]{32L, 16L}, new long[]{8L},
                        2, true, false, true, DEVICE);
                Tensor y = m.forward(featMap(feats, B, 80), randint(B, 2));
                expectSize(y, 1, DBMTL.NUM_OUTPUTS);
            }
        });
        list.add(new Case("GraphDrugEncoder") {
            @Override public void run() {
                GraphDrugEncoder m = new GraphDrugEncoder(6, 16, 16, 2, 0.0f, DEVICE);
                Tensor x = randn(10, 6);
                Tensor adj = torch.eye(10, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
                adj = GraphDrugEncoder.normalizeAdj(adj);
                Tensor y = m.forward(x, adj);
                expectSize(y, 0, 1);
                expectSize(y, 1, 16);
            }
        });

        return list;
    }

    // ---- helpers ----

    private static List<Feature> sparseFeats(int n, int vocab, int dim) {
        List<Feature> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            list.add(new SparseFeature("f" + i, vocab, dim));
        }
        return list;
    }

    private static Map<String, Tensor> featMap(List<Feature> feats, int batch, int vocab) {
        Map<String, Tensor> m = new LinkedHashMap<>();
        for (Feature f : feats) {
            m.put(f.name(), randint(batch, Math.max(vocab, 2)));
        }
        return m;
    }

    private static Tensor randn(long... sizes) {
        return torch.randn(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
    }

    private static Tensor randint(long n, int high) {
        // uniform long ids in [0, high)
        Tensor t = torch.randint(high, new long[]{n},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        return t;
    }

    private static Tensor randint(long a, long b, int high) {
        return torch.randint(high, new long[]{a, b},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
    }

    private static Tensor randint(long a, long b, long c, int high) {
        return torch.randint(high, new long[]{a, b, c},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
    }

    private static void expectRank(Tensor t, int rank, String name) {
        if (t == null || t.isNull()) throw new AssertionError(name + " is null");
        if ((int) t.dim() != rank) {
            throw new AssertionError(name + " rank=" + t.dim() + " expected " + rank
                    + " shape=" + t.sizes());
        }
    }

    private static void expectSize(Tensor t, int dim, long size) {
        if (t.size(dim) != size) {
            throw new AssertionError("size[" + dim + "]=" + t.size(dim) + " expected " + size
                    + " shape=" + t.sizes());
        }
    }
}
