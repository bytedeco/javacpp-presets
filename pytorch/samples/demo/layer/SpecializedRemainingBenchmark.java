package samples.demo.layer;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.nn.conv.*;
import org.bytedeco.pytorch.geometric.nn.kge.ComplEx;
import org.bytedeco.pytorch.geometric.nn.kge.DistMult;
import org.bytedeco.pytorch.geometric.nn.kge.RotatE;
import org.bytedeco.pytorch.geometric.nn.kge.TransE;
import org.bytedeco.pytorch.geometric.nn.model.EdgeCNN;
import org.bytedeco.pytorch.geometric.nn.model.GAE;
import org.bytedeco.pytorch.geometric.nn.model.GAT;
import org.bytedeco.pytorch.geometric.nn.model.GCNEncoder;
import org.bytedeco.pytorch.geometric.nn.model.GIN;
import org.bytedeco.pytorch.geometric.nn.model.GraphUNet;
import org.bytedeco.pytorch.geometric.nn.model.InnerProductDecoder;
import org.bytedeco.pytorch.geometric.nn.model.JumpingKnowledge;
import org.bytedeco.pytorch.geometric.nn.model.LightGCN;
import org.bytedeco.pytorch.geometric.nn.model.NeuralFingerprint;
import org.bytedeco.pytorch.geometric.nn.model.PNA;
import org.bytedeco.pytorch.geometric.transforms.HighOrderTransforms;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Smoke + micro-benchmark for remaining specialized geometric components:
 * CuGraph*, FusedGAT, HEAT/HAN/HGT/Hetero, XConv/GravNet/Spline/GMM,
 * point-cloud convs, other specialized convs, KGE, nn.model, HighOrderTransforms.
 */
public class SpecializedRemainingBenchmark {

    private static final int WARMUP = 1;
    private static final int ITERS = 2;
    private static final long N = 16;
    private static final long IN = 8;
    private static final long OUT = 8;
    private static final long E = 32;

    private static int passed = 0;
    private static int failed = 0;
    private static final List<String> failures = new ArrayList<>();
    private static final List<String> rows = new ArrayList<>();

    public static void main(String[] args) {
        torch.manual_seed(21);
        System.out.println("=== Specialized Remaining Benchmark ===");
        System.out.printf(Locale.ROOT, "N=%d E=%d in=%d out=%d%n%n", N, E, IN, OUT);

        try (PointerScope scope = new PointerScope()) {
            Tensor x = torch.randn(N, IN);
            Tensor edgeIndex = randomEdges(N, E);
            Tensor batch = torch.zeros(new long[]{N}, longOpts());
            Tensor pos = torch.randn(N, 3);
            Tensor normalRaw = torch.randn(N, 3);
            // unit-ish normals
            final Tensor normal = normalRaw.div(normalRaw.norm(
                    new org.bytedeco.pytorch.ScalarOptional(new Scalar(2)),
                    new long[]{1}, true).add(new Scalar(1e-6)));
            Tensor edgeAttr2 = torch.rand(E, 2); // pseudo in [0,1]
            Tensor edgeAttr3 = torch.rand(E, 3);
            final Tensor edgeType = torch.arange(new Scalar(E)).remainder(new Scalar(2))
                    .to(torch.ScalarType.Long);

            // ========== CSC / CSR for CuGraph + FusedGAT ==========
            Object[] fmt = FusedGATConv.toGraphFormat(edgeIndex, N);
            Tensor[] csr = (Tensor[]) fmt[0];
            Tensor[] csc = (Tensor[]) fmt[1];
            Tensor perm = (Tensor) fmt[2];
            Tensor row = csc[0];
            Tensor colptr = csc[1];

            bench("CuGraphSAGEConv", () -> new CuGraphSAGEConv(IN, OUT, "mean", false, true, true),
                    m -> ((CuGraphSAGEConv) m).forward(x, row, colptr), OUT);
            bench("CuGraphGATConv", () -> new CuGraphGATConv(IN, OUT, 2, true, 0.2, true),
                    m -> ((CuGraphGATConv) m).forward(x, row, colptr), 2 * OUT);
            bench("CuGraphRGCNConv", () -> new CuGraphRGCNConv(IN, OUT, 2, true, true, "mean"),
                    m -> ((CuGraphRGCNConv) m).forward(x, row, colptr, edgeType), OUT);
            bench("FusedGATConv", () -> new FusedGATConv(IN, OUT, 2, true, 0.2),
                    m -> ((FusedGATConv) m).forward(x, csr, csc, perm), 2 * OUT);

            // ========== Pseudo-coord / geometric convs ==========
            bench("GMMConv", () -> new GMMConv(IN, OUT, 2, 4),
                    m -> ((GMMConv) m).forward(x, edgeIndex, edgeAttr2), OUT);
            bench("SplineConv", () -> new SplineConv(IN, OUT, 2, 4),
                    m -> ((SplineConv) m).forward(x, edgeIndex, edgeAttr2), OUT);
            bench("GravNetConv", () -> new GravNetConv(IN, OUT, 4, 8, 4),
                    m -> ((GravNetConv) m).forward(x, batch), OUT);
            bench("XConv", () -> new XConv(IN, OUT, 3, 4, 8, 1, true),
                    m -> ((XConv) m).forward(x, pos, batch), OUT);

            // PointNet / PPF / PointGNN / PointTransformer / DynamicEdge
            SequentialImpl localPn = mlp(IN + 3, OUT); // x||relPos often
            // PointNet local typically takes [x_j || pos_j - pos_i] = IN+3
            SequentialImpl globalPn = mlp(OUT, OUT);
            bench("PointNetConv", () -> new PointNetConv(mlp(IN + 3, OUT), mlp(OUT, OUT), true),
                    m -> ((PointNetConv) m).forward(x, pos, edgeIndex), OUT);

            SequentialImpl ppfLocal = mlp(IN + 4, OUT); // x_j || PPF(4)
            SequentialImpl ppfGlobal = mlp(OUT, OUT);
            bench("PPFConv", () -> new PPFConv(ppfLocal, ppfGlobal, true),
                    m -> ((PPFConv) m).forward(x, pos, normal, edgeIndex), OUT);

            SequentialImpl mlpH = mlp(IN, 3);   // delta pos
            SequentialImpl mlpF = mlp(IN + 3, OUT);
            SequentialImpl mlpG = mlp(OUT, OUT);
            bench("PointGNNConv", () -> new PointGNNConv(mlpH, mlpF, mlpG),
                    m -> ((PointGNNConv) m).forward(x, pos, edgeIndex), OUT);

            SequentialImpl posNN = mlp(3, OUT);
            SequentialImpl attnNN = mlp(OUT, 1);
            bench("PointTransformerConv",
                    () -> new PointTransformerConv(IN, OUT, 3, posNN, attnNN),
                    m -> ((PointTransformerConv) m).forward(x, pos, edgeIndex), OUT);

            SequentialImpl dynNn = mlp(2 * IN, OUT);
            bench("DynamicEdgeConv", () -> new DynamicEdgeConv(dynNn, 4, "max"),
                    m -> ((DynamicEdgeConv) m).forward(x, batch), OUT);

            // ========== Specialized graph convs ==========
            bench("MixHopConv", () -> new MixHopConv(IN, OUT, Arrays.asList(0, 1, 2), true),
                    m -> ((MixHopConv) m).forward(x, edgeIndex), 3 * OUT); // cat powers
            bench("DNAConv", () -> new DNAConv(IN, 2, 1, true),
                    m -> ((DNAConv) m).forward(x, edgeIndex), IN);
            bench("PDNConv", () -> new PDNConv(IN, OUT, 2, 8, true, true),
                    m -> ((PDNConv) m).forward(x, edgeIndex, edgeAttr2), OUT);
            SequentialImpl nnEdge = mlp(2, IN * OUT); // edge_attr → weight matrix flattened
            bench("NNConv", () -> new NNConv(IN, OUT, nnEdge),
                    m -> ((NNConv) m).forward(x, edgeIndex, edgeAttr2), OUT);
            bench("EGConv", () -> new EGConv(IN, OUT, Arrays.asList("sum", "mean", "max"), 2, 4, true),
                    m -> ((EGConv) m).forward(x, edgeIndex), OUT);
            bench("RGATConv", () -> new RGATConv(IN, OUT, 2, 2, true),
                    m -> ((RGATConv) m).forward(x, edgeIndex, edgeType), 2 * OUT);
            bench("GeneralConv", () -> new GeneralConv(IN, OUT, null, 2, true, "additive", true, false, true),
                    m -> ((GeneralConv) m).forward(x, edgeIndex), 2 * OUT);
            bench("DirGNNConv", () -> new DirGNNConv(new SAGEConvV2(IN, OUT), 0.5f, true, IN, OUT),
                    m -> ((DirGNNConv) m).forward(x, edgeIndex), OUT);
            bench("GPSConv", () -> new GPSConv((int) IN, new GCNConv(IN, IN), 2, 0.0f),
                    m -> ((GPSConv) m).forward(x, edgeIndex, batch), IN);
            bench("AntiSymmetricConv",
                    () -> new AntiSymmetricConv((int) IN, new GCNConv(IN, IN), 2, 0.1f, 0.1f, true),
                    m -> ((AntiSymmetricConv) m).forward(x, edgeIndex), IN);

            // WLConv needs Long labels
            Tensor labels = torch.arange(new Scalar(N)).remainder(new Scalar(3)).to(torch.ScalarType.Long);
            bench("WLConv", WLConv::new,
                    m -> ((WLConv) m).forward(labels, edgeIndex), /*feat*/ -1); // labels 1-D

            // HEATConv needs node/edge types + edge_attr
            Tensor nodeType = torch.arange(new Scalar(N)).remainder(new Scalar(2)).to(torch.ScalarType.Long);
            Tensor heatEdgeAttr = torch.randn(E, 4);
            bench("HEATConv", () -> new HEATConv(IN, OUT, 2, 2, 4, 4, 4, 2, true),
                    m -> ((HEATConv) m).forward(x, edgeIndex, nodeType, edgeType, heatEdgeAttr),
                    2 * OUT);

            // Hetero: HAN / HGT / HeteroConv
            run("HANConv", () -> {
                Map<String, Integer> inDict = new LinkedHashMap<>();
                inDict.put("user", (int) IN);
                inDict.put("item", (int) IN);
                List<String> nTypes = Arrays.asList("user", "item");
                List<String[]> eTypes = new ArrayList<>();
                eTypes.add(new String[]{"user", "buys", "item"});
                eTypes.add(new String[]{"item", "bought_by", "user"});
                HANConv han = new HANConv(inDict, (int) OUT, nTypes, eTypes, 2);
                Map<String, Tensor> xDict = new HashMap<>();
                xDict.put("user", torch.randn(8, IN));
                xDict.put("item", torch.randn(8, IN));
                Map<String[], Tensor> eiDict = new HashMap<>();
                eiDict.put(eTypes.get(0), randomEdges(8, 12));
                eiDict.put(eTypes.get(1), randomEdges(8, 12));
                Map<String, Tensor> out = han.forward(xDict, eiDict);
                if (out == null || out.isEmpty()) {
                    throw new AssertionError("HANConv empty output");
                }
                for (Tensor t : out.values()) {
                    checkFinite(t);
                }
            });

            run("HGTConv", () -> {
                Map<String, Integer> inDict = new LinkedHashMap<>();
                inDict.put("user", (int) IN);
                inDict.put("item", (int) IN);
                List<String> nTypes = Arrays.asList("user", "item");
                List<String[]> eTypes = new ArrayList<>();
                eTypes.add(new String[]{"user", "buys", "item"});
                eTypes.add(new String[]{"item", "bought_by", "user"});
                HGTConv hgt = new HGTConv(inDict, (int) OUT, nTypes, eTypes, 2);
                Map<String, Tensor> xDict = new HashMap<>();
                xDict.put("user", torch.randn(8, IN));
                xDict.put("item", torch.randn(8, IN));
                Map<String[], Tensor> eiDict = new HashMap<>();
                eiDict.put(eTypes.get(0), randomEdges(8, 12));
                eiDict.put(eTypes.get(1), randomEdges(8, 12));
                Map<String, Tensor> out = hgt.forward(xDict, eiDict);
                if (out == null) {
                    throw new AssertionError("HGTConv null");
                }
                for (Tensor t : out.values()) {
                    if (t != null) checkFinite(t);
                }
            });

            run("HeteroConv", () -> {
                Map<String, MessagePassing> convs = new LinkedHashMap<>();
                convs.put("user,to,user", new SAGEConv(IN, OUT));
                convs.put("item,to,item", new SAGEConv(IN, OUT));
                HeteroConv hc = new HeteroConv(convs, "sum");
                Map<String, Tensor> xDict = new HashMap<>();
                xDict.put("user", torch.randn(8, IN));
                xDict.put("item", torch.randn(8, IN));
                Map<String, Tensor> eiDict = new HashMap<>();
                eiDict.put("user,to,user", randomEdges(8, 10));
                eiDict.put("item,to,item", randomEdges(8, 10));
                Map<String, Tensor> out = hc.forward(xDict, eiDict);
                if (out == null || out.isEmpty()) {
                    throw new AssertionError("HeteroConv empty");
                }
                for (Tensor t : out.values()) {
                    checkFinite(t);
                    if (t.size(1) != OUT) {
                        throw new AssertionError("HeteroConv feat " + t.size(1));
                    }
                }
            });

            // ========== KGE ==========
            Tensor headIdx = torch.tensor(new long[]{0, 1, 2});
            Tensor relIdx = torch.tensor(new long[]{0, 1, 0});
            Tensor tailIdx = torch.tensor(new long[]{3, 4, 5});
            bench("TransE", () -> new TransE(10, 4, 16, 1L),
                    m -> ((TransE) m).forward(headIdx, relIdx, tailIdx), -1);
            bench("DistMult", () -> new DistMult(10, 4, 16),
                    m -> ((DistMult) m).forward(headIdx, relIdx, tailIdx), -1);
            bench("ComplEx", () -> new ComplEx(10, 4, 16),
                    m -> ((ComplEx) m).forward(headIdx, relIdx, tailIdx), -1);
            bench("RotatE", () -> new RotatE(10, 4, 16, 0.001),
                    m -> ((RotatE) m).forward(headIdx, relIdx, tailIdx), -1);

            // ========== nn.model ==========
            bench("GIN.model", () -> new GIN(IN, 16, 4, 2, 0.0),
                    m -> ((GIN) m).forward(x, edgeIndex), 4);
            bench("GAT.model", () -> new GAT(IN, 8, 4, 2, 1, 0.0),
                    m -> ((GAT) m).forward(x, edgeIndex), 4);
            bench("PNA.model", () -> new PNA(IN, 8, 4, 2,
                            new String[]{"mean", "max"}, new String[]{"identity"}, 4.0),
                    m -> ((PNA) m).forward(x, edgeIndex), 4);
            bench("LightGCN", () -> new LightGCN(N, 8, 2),
                    m -> ((LightGCN) m).forward(edgeIndex), 8);
            bench("GCNEncoder", () -> new GCNEncoder(IN, OUT),
                    m -> ((GCNEncoder) m).forward(x, edgeIndex), OUT);
            bench("NeuralFingerprint", () -> new NeuralFingerprint(IN, 16, 32, 2),
                    m -> ((NeuralFingerprint) m).forward(x, edgeIndex, batch), 32);
            bench("GraphUNet", () -> new GraphUNet(IN, 8, 4, 0.5),
                    m -> ((GraphUNet) m).forward(x, edgeIndex), 4);
            bench("EdgeCNN", () -> new EdgeCNN(IN, OUT, 4),
                    m -> ((EdgeCNN) m).forward(x, batch), OUT);
            bench("JumpingKnowledge.cat", () -> new JumpingKnowledge("cat", IN, 3),
                    m -> {
                        List<Tensor> xs = Arrays.asList(x, x, x);
                        return ((JumpingKnowledge) m).forward(xs);
                    }, 3 * IN);
            bench("GAE", () -> new GAE(new GCNEncoder(IN, OUT)),
                    m -> {
                        GAE gae = (GAE) m;
                        Tensor z = gae.encode(x, edgeIndex);
                        Tensor dec = gae.decode(z, edgeIndex, true);
                        checkFinite(z);
                        checkFinite(dec);
                        return z;
                    }, OUT);
            run("InnerProductDecoder", () -> {
                InnerProductDecoder dec = new InnerProductDecoder();
                Tensor z = torch.randn(N, OUT);
                Tensor out = dec.forward(z, edgeIndex, true);
                checkFinite(out);
                if (out.size(0) != edgeIndex.size(1)) {
                    throw new AssertionError("decoder scores E mismatch");
                }
            });

            // ========== HighOrderTransforms ==========
            run("AddMetaPaths", () -> {
                GraphData gMeta = new GraphData(torch.randn(N, IN), edgeIndex.contiguous());
                HighOrderTransforms.AddMetaPaths meta =
                        new HighOrderTransforms.AddMetaPaths(new String[]{"drug", "binds", "protein", "binds", "drug"});
                GraphData outMeta = meta.apply(gMeta);
                if (outMeta.get("_metapath_len") == null
                        || outMeta.get("_metapath_len").item().toLong() != 5) {
                    throw new AssertionError("metapath_len expected 5");
                }
            });
            run("LargestConnectedComponents", () -> {
                // Two components: 0-1-2 and 3-4; plus isolated 5.. should keep 0-1-2
                Tensor eiLcc = torch.tensor(new long[]{
                        0, 1, 1, 2, 3, 4,
                        1, 0, 2, 1, 4, 3
                }).reshape(2, 6);
                GraphData gLcc = new GraphData(torch.randn(6, IN), eiLcc);
                gLcc.pos = torch.randn(6, 3);
                gLcc.y = torch.arange(new Scalar(6));
                HighOrderTransforms.LargestConnectedComponents lcc =
                        new HighOrderTransforms.LargestConnectedComponents(1);
                GraphData outLcc = lcc.apply(gLcc);
                if (outLcc.x == null || outLcc.x.size(0) != 3) {
                    throw new AssertionError("LCC expected 3 nodes, got "
                            + (outLcc.x == null ? null : outLcc.x.size(0)));
                }
                if (outLcc.edge_index == null || outLcc.edge_index.size(1) < 2) {
                    throw new AssertionError("LCC edges missing");
                }
                // remapped indices must be in [0,3)
                long maxIdx = outLcc.edge_index.max().item().toLong();
                if (maxIdx >= 3) {
                    throw new AssertionError("LCC remapping failed maxIdx=" + maxIdx);
                }
            });
            run("LargestConnectedComponents.top2", () -> {
                Tensor eiTop2 = torch.tensor(new long[]{
                        0, 1, 2, 3,
                        1, 0, 3, 2
                }).reshape(2, 4); // two edges components size 2 each + isolates
                GraphData gTop2 = new GraphData(torch.randn(5, IN), eiTop2);
                GraphData outTop2 = new HighOrderTransforms.LargestConnectedComponents(2).apply(gTop2);
                if (outTop2.x.size(0) != 4) {
                    throw new AssertionError("top2 expected 4 nodes got " + outTop2.x.size(0));
                }
            });
        }

        System.out.println();
        System.out.println("----- Results -----");
        System.out.printf(Locale.ROOT, "%-32s %8s %10s %s%n", "name", "status", "ms/iter", "notes");
        for (String r : rows) System.out.println(r);
        System.out.println();
        System.out.printf(Locale.ROOT, "PASSED=%d FAILED=%d TOTAL=%d%n",
                passed, failed, passed + failed);
        if (!failures.isEmpty()) {
            System.out.println("\n----- Failures -----");
            for (String f : failures) System.out.println(f);
            System.exit(1);
        }
    }

    private static SequentialImpl mlp(long in, long out) {
        SequentialImpl s = new SequentialImpl();
        s.push_back(new LinearImpl(in, out));
        s.push_back(new ReLUImpl());
        s.push_back(new LinearImpl(out, out));
        return s;
    }

    private static TensorOptions longOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long));
    }

    private static Tensor randomEdges(long n, long e) {
        java.util.Random rnd = new java.util.Random(21);
        long[] flat = new long[(int) (2 * e)];
        for (int i = 0; i < e; i++) {
            flat[i] = rnd.nextInt((int) n);
            flat[(int) e + i] = rnd.nextInt((int) n);
        }
        return torch.tensor(flat).reshape(2, e);
    }

    @FunctionalInterface
    interface Forward {
        Tensor apply(Object module);
    }

    private static void bench(String name, Supplier<Object> factory, Forward fwd, long outFeat) {
        run(name, () -> {
            Object m = factory.get();
            Tensor last = null;
            for (int i = 0; i < WARMUP; i++) {
                last = fwd.apply(m);
                checkOut(last, outFeat);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                last = fwd.apply(m);
            }
            long t1 = System.nanoTime();
            checkOut(last, outFeat);
            double ms = (t1 - t0) / 1e6 / ITERS;
            rows.add(String.format(Locale.ROOT, "%-32s %8s %8.3f  shape=%s",
                    name, "PASS", ms, Arrays.toString(last.shape())));
        });
    }

    private static void checkOut(Tensor out, long outFeat) {
        checkFinite(out);
        if (outFeat < 0) {
            // 1-D scores / labels — just finite
            return;
        }
        if (out.dim() == 1) {
            // e.g. WL labels
            return;
        }
        long last = out.size(out.dim() - 1);
        if (last != outFeat) {
            if (out.dim() == 3 && out.size(1) * out.size(2) == outFeat) {
                return;
            }
            throw new AssertionError("feat expected " + outFeat + " got " + last
                    + " shape=" + Arrays.toString(out.shape()));
        }
    }

    private static void checkFinite(Tensor out) {
        if (out.isnan().any().item().toBool()) throw new AssertionError("NaN");
        if (out.isinf().any().item().toBool()) throw new AssertionError("Inf");
    }

    private static void run(String name, Runnable body) {
        try {
            body.run();
            if (rows.stream().noneMatch(r -> r.startsWith(name + " "))) {
                rows.add(String.format(Locale.ROOT, "%-32s %8s %8s  ok", name, "PASS", "-"));
            }
            passed++;
            System.out.println("PASS  " + name);
        } catch (Throwable t) {
            failed++;
            String msg = t.getClass().getSimpleName() + ": " + t.getMessage();
            failures.add(name + " -> " + msg);
            rows.add(String.format(Locale.ROOT, "%-32s %8s %8s  %s",
                    name, "FAIL", "-", abbreviate(msg, 90)));
            System.err.println("FAIL  " + name + " -> " + msg);
            t.printStackTrace(System.err);
        }
    }

    private static String abbreviate(String s, int n) {
        if (s == null) return "";
        String one = s.replace('\n', ' ');
        return one.length() <= n ? one : one.substring(0, n - 3) + "...";
    }
}
