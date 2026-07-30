package samples.demo.layer;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.*;
import org.bytedeco.pytorch.geometric.nn.norm.*;
import org.bytedeco.pytorch.geometric.nn.pooling.GlobalPooling;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

/**
 * Smoke + micro-benchmark for geometric/nn conv + norm (+ a few pooling helpers).
 * Only uses documented public constructors / forward overloads — no invented APIs.
 */
public class NNConvBenchmark {

    private static final int WARMUP = 1;
    private static final int ITERS = 3;
    private static final long IN = 8;
    private static final long OUT = 8;
    private static final long N = 32;
    private static final long E = 96;

    private static int passed = 0;
    private static int failed = 0;
    private static final List<String> failures = new ArrayList<>();
    private static final List<String> rows = new ArrayList<>();

    public static void main(String[] args) {
        torch.manual_seed(7);
        System.out.println("=== NN Conv / Norm Benchmark ===");
        System.out.printf(Locale.ROOT, "N=%d E=%d in=%d out=%d warmup=%d iters=%d%n%n",
                N, E, IN, OUT, WARMUP, ITERS);

        try (PointerScope scope = new PointerScope()) {
            Tensor x = torch.randn(N, IN);
            Tensor edgeIndex = randomEdges(N, E);
            Tensor batch = torch.zeros(new long[]{N}, edgeIndex.options()); // all one graph
            // dense adj for Dense* layers: [B=1, N, N] identity-ish undirected
            Tensor adj = torch.eye(N).unsqueeze(0); // [1, N, N]
            Tensor xDense = x.unsqueeze(0);         // [1, N, IN]

            // ---- standard (x, edge_index) layers ----
            bench("GCNConv", () -> new GCNConv(IN, OUT),
                    m -> ((GCNConv) m).forward(x, edgeIndex), OUT);
            bench("GCNConvV2", () -> new GCNConvV2(IN, OUT),
                    m -> ((GCNConvV2) m).forward(x, edgeIndex), OUT);
            bench("GCNConvV3", () -> new GCNConvV3(IN, OUT),
                    m -> ((GCNConvV3) m).forward(x, edgeIndex), OUT);
            bench("SAGEConv", () -> new SAGEConv(IN, OUT),
                    m -> ((SAGEConv) m).forward(x, edgeIndex), OUT);
            bench("SAGEConvV2", () -> new SAGEConvV2(IN, OUT), // inDimSrc, outDim
                    m -> ((SAGEConvV2) m).forward(x, edgeIndex), OUT);
            bench("SAGEConvV3", () -> new SAGEConvV3(IN, OUT),
                    m -> ((SAGEConvV3) m).forward(x, edgeIndex), OUT);
            bench("GraphConv", () -> new GraphConv(IN, OUT),
                    m -> ((GraphConv) m).forward(x, edgeIndex), OUT);
            bench("GATConv", () -> new GATConv(IN, OUT, 2, 0.2),
                    m -> ((GATConv) m).forward(x, edgeIndex), 2 * OUT); // concat heads
            bench("GATConvV2", () -> new GATConvV2(IN, OUT, 2, 0.2),
                    m -> ((GATConvV2) m).forward(x, edgeIndex), 2 * OUT);
            bench("GATConvFinal", () -> new GATConvFinal(IN, OUT, 2, 0.2),
                    m -> ((GATConvFinal) m).forward(x, edgeIndex), 2 * OUT);
            bench("GATv2Conv", () -> new GATv2Conv(IN, OUT, 2),
                    m -> ((GATv2Conv) m).forward(x, edgeIndex), 2 * OUT);
            bench("TransformerConv", () -> new TransformerConv(IN, OUT, 2),
                    m -> ((TransformerConv) m).forward(x, edgeIndex), 2 * OUT);
            bench("TransformerConvV2", () -> new TransformerConvV2(IN, OUT, 2),
                    m -> ((TransformerConvV2) m).forward(x, edgeIndex), 2 * OUT);
            bench("ChebConv", () -> new ChebConv(IN, OUT, 2, "sym", true),
                    m -> ((ChebConv) m).forward(x, edgeIndex), OUT);
            bench("ARMAConv", () -> new ARMAConv(IN, OUT, 1, 1),
                    m -> ((ARMAConv) m).forward(x, edgeIndex), OUT);
            bench("SGConv", () -> new SGConv(IN, OUT, 2),
                    m -> ((SGConv) m).forward(x, edgeIndex), OUT);
            bench("SSGConv", () -> new SSGConv(IN, OUT, 0.1, 2),
                    m -> ((SSGConv) m).forward(x, edgeIndex), OUT);
            bench("TAGConv", () -> new TAGConv(IN, OUT, 2),
                    m -> ((TAGConv) m).forward(x, edgeIndex), OUT);
            bench("APPNP", () -> new APPNP(3, 0.1),
                    m -> ((APPNP) m).forward(x, edgeIndex), IN); // preserves channels
            bench("AGNNConv", () -> new AGNNConv(),
                    m -> ((AGNNConv) m).forward(x, edgeIndex), IN);
            bench("LEConv", () -> new LEConv(IN, OUT),
                    m -> ((LEConv) m).forward(x, edgeIndex), OUT);
            bench("LGConv", () -> new LGConv(),
                    m -> ((LGConv) m).forward(x, edgeIndex), IN);
            bench("SimpleConv", () -> new SimpleConv(),
                    m -> ((SimpleConv) m).forward(x, edgeIndex), IN);
            // GatedGraphConv requires x.size(1) == outChannels (no internal projection).
            bench("GatedGraphConv", () -> new GatedGraphConv(IN, 2),
                    m -> ((GatedGraphConv) m).forward(x, edgeIndex), IN);
            bench("GENConv", () -> new GENConv(IN, OUT),
                    m -> ((GENConv) m).forward(x, edgeIndex), OUT);
            bench("ResGatedGraphConv", () -> new ResGatedGraphConv(IN, OUT),
                    m -> ((ResGatedGraphConv) m).forward(x, edgeIndex), OUT);
            bench("FeaStConv", () -> new FeaStConv(IN, OUT, 2),
                    m -> ((FeaStConv) m).forward(x, edgeIndex), OUT);
            bench("ClusterGCNConv", () -> new ClusterGCNConv(IN, OUT),
                    m -> ((ClusterGCNConv) m).forward(x, edgeIndex), OUT);
            bench("EdgeConv", () -> new EdgeConv(IN, OUT),
                    m -> ((EdgeConv) m).forward(x, edgeIndex), OUT);
            bench("CGConv", () -> new CGConv(IN),
                    m -> ((CGConv) m).forward(x, edgeIndex), IN);
            bench("FAConv", () -> new FAConv(IN, 0.1f, 0.0f, true),
                    m -> ((FAConv) m).forward(x, edgeIndex), IN);
            bench("PANConv", () -> new PANConv(IN, OUT, 2),
                    m -> ((PANConv) m).forward(x, edgeIndex), OUT);
            bench("MFConv", () -> new MFConv(IN, OUT, 10, true),
                    m -> ((MFConv) m).forward(x, edgeIndex), OUT);
            bench("SuperGATConv", () -> new SuperGATConv(IN, OUT, 2, true, "MX"),
                    m -> ((SuperGATConv) m).forward(x, edgeIndex), 2 * OUT);
            bench("WLConvContinuous", () -> new WLConvContinuous(),
                    m -> ((WLConvContinuous) m).forward(x, edgeIndex), IN);
            bench("PNAConv", () -> new PNAConv(IN, OUT,
                            new String[]{"mean", "max", "min"},
                            new String[]{"identity", "amplification"},
                            4.0),
                    m -> ((PNAConv) m).forward(x, edgeIndex), OUT);

            // GIN / GINE need MLP
            bench("GINConv", () -> {
                SequentialImpl mlp = new SequentialImpl();
                mlp.push_back(new LinearImpl(IN, OUT));
                mlp.push_back(new ReLUImpl());
                mlp.push_back(new LinearImpl(OUT, OUT));
                return new GINConv(mlp, true);
            }, m -> ((GINConv) m).forward(x, edgeIndex), OUT);

            bench("GINEConv", () -> {
                SequentialImpl mlp = new SequentialImpl();
                mlp.push_back(new LinearImpl(IN, OUT));
                mlp.push_back(new ReLUImpl());
                mlp.push_back(new LinearImpl(OUT, OUT));
                // nodeDim must be > 0 (GINE uses it for edge-feature alignment)
                return new GINEConv(mlp, 0.0, true, null, (int) IN);
            }, m -> ((GINEConv) m).forward(x, edgeIndex), OUT);

            // GCN2 needs (x, x0, edge_index) ideally
            bench("GCN2Conv", () -> new GCN2Conv(IN, 0.1f, 0.5f, 1, true, true),
                    m -> ((GCN2Conv) m).forward(x, edgeIndex), IN);

            // SignedConv first layer
            bench("SignedConv", () -> new SignedConv(IN, OUT, true, true),
                    m -> {
                        Tensor pos = edgeIndex;
                        Tensor neg = edgeIndex; // reuse for smoke
                        return ((SignedConv) m).forward(x, pos, neg);
                    }, 2 * OUT); // firstAggr typically cats pos/neg

            // RGCN / FiLM need edge_type
            Tensor edgeType = torch.zeros(new long[]{edgeIndex.size(1)}, edgeIndex.options());
            bench("RGCNConv", () -> new RGCNConv(IN, OUT, 2, true, true),
                    m -> ((RGCNConv) m).forward(x, edgeIndex, edgeType), OUT);
            bench("FastRGCNConv", () -> new FastRGCNConv(IN, OUT, 2, true, true),
                    m -> ((FastRGCNConv) m).forward(x, edgeIndex, edgeType), OUT);
            bench("FiLMConv", () -> new FiLMConv(IN, OUT, 2),
                    m -> ((FiLMConv) m).forward(x, edgeIndex, edgeType), OUT);

            // Hypergraph: edge_index is [2, E] node-hyperedge incidence
            // For smoke, reuse graph edges as incidence
            bench("HypergraphConv", () -> new HypergraphConv(IN, OUT, false, 1, true),
                    m -> ((HypergraphConv) m).forward(x, edgeIndex), OUT);

            // Dense variants
            bench("DenseGCNConv", () -> new DenseGCNConv(IN, OUT),
                    m -> ((DenseGCNConv) m).forward(xDense, adj), OUT, /*dense*/ true);
            bench("DenseSAGEConv", () -> new DenseSAGEConv(IN, OUT),
                    m -> ((DenseSAGEConv) m).forward(xDense, adj), OUT, true);
            bench("DenseGraphConv", () -> new DenseGraphConv(IN, OUT),
                    m -> ((DenseGraphConv) m).forward(xDense, adj), OUT, true);
            bench("DenseGATConv", () -> new DenseGATConv(IN, OUT, 2),
                    m -> ((DenseGATConv) m).forward(xDense, adj), 2 * OUT, true);

            // ---- norms ----
            bench("BatchNorm", () -> new BatchNorm(IN),
                    m -> ((BatchNorm) m).forward(x), IN);
            bench("LayerNorm", () -> new LayerNorm(IN),
                    m -> ((LayerNorm) m).forward(x), IN);
            bench("PairNorm", () -> new PairNorm(),
                    m -> ((PairNorm) m).forward(x), IN);
            bench("GraphSizeNorm", () -> new GraphSizeNorm(),
                    m -> ((GraphSizeNorm) m).forward(x, batch), IN);
            bench("GraphNorm", () -> new GraphNorm(IN),
                    m -> ((GraphNorm) m).forward(x, batch), IN);
            bench("MeanSubtractionNorm", () -> new MeanSubtractionNorm(),
                    m -> ((MeanSubtractionNorm) m).forward(x, batch), IN);
            bench("MessageNorm", () -> new MessageNorm(),
                    m -> ((MessageNorm) m).forward(x, x), IN);
            bench("DiffGroupNorm", () -> new DiffGroupNorm(IN, 2, 1e-5),
                    m -> ((DiffGroupNorm) m).forward(x), IN);
            bench("InstanceNorm", () -> new InstanceNorm(IN, 1e-5, 0.1, true, true),
                    m -> ((InstanceNorm) m).forward(x, batch), IN);

            // ---- global pooling smoke ----
            run("GlobalPooling.mean", () -> {
                Tensor out = GlobalPooling.pool(x, batch, "mean");
                if (out.size(0) != 1) throw new AssertionError("batch graphs expected 1, got " + out.size(0));
                if (out.size(1) != IN) throw new AssertionError("feat dim " + out.size(1));
                checkFinite(out);
            });
            run("GlobalPooling.max", () -> {
                Tensor out = GlobalPooling.pool(x, batch, "max");
                checkFinite(out);
            });
            run("GlobalPooling.sum", () -> {
                Tensor out = GlobalPooling.pool(x, batch, "sum");
                checkFinite(out);
            });
        }

        System.out.println();
        System.out.println("----- Results -----");
        System.out.printf(Locale.ROOT, "%-28s %8s %10s %s%n", "name", "status", "ms/iter", "notes");
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

    private static Tensor randomEdges(long n, long e) {
        java.util.Random rnd = new java.util.Random(7);
        long[] src = new long[(int) e];
        long[] dst = new long[(int) e];
        for (int i = 0; i < e; i++) {
            src[i] = rnd.nextInt((int) n);
            dst[i] = rnd.nextInt((int) n);
        }
        // build [2, E]
        long[] flat = new long[(int) (2 * e)];
        for (int i = 0; i < e; i++) {
            flat[i] = src[i];
            flat[(int) e + i] = dst[i];
        }
        return torch.tensor(flat).reshape(2, e);
    }

    @FunctionalInterface
    interface Forward {
        Tensor apply(Object module);
    }

    private static void bench(String name, Supplier<Object> factory, Forward fwd, long outFeat) {
        bench(name, factory, fwd, outFeat, false);
    }

    private static void bench(String name, Supplier<Object> factory, Forward fwd,
                              long outFeat, boolean dense) {
        run(name, () -> {
            Object m = factory.get();
            Tensor last = null;
            for (int i = 0; i < WARMUP; i++) {
                last = fwd.apply(m);
                checkOut(last, outFeat, dense);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                last = fwd.apply(m);
            }
            long t1 = System.nanoTime();
            checkOut(last, outFeat, dense);
            double ms = (t1 - t0) / 1e6 / ITERS;
            rows.add(String.format(Locale.ROOT, "%-28s %8s %8.3f  shape=%s",
                    name, "PASS", ms, Arrays.toString(last.shape())));
        });
    }

    private static void checkOut(Tensor out, long outFeat, boolean dense) {
        checkFinite(out);
        if (dense) {
            // [B, N, F]
            if (out.dim() != 3) throw new AssertionError("dense expected 3D, got " + out.dim());
            if (out.size(2) != outFeat) {
                throw new AssertionError("dense feat expected " + outFeat + " got " + out.size(2));
            }
        } else {
            if (out.dim() < 2) throw new AssertionError("expected >=2D, got " + out.dim());
            if (out.size(out.dim() - 1) != outFeat) {
                // some layers may return [N, heads, C] — accept if product matches or last==outFeat after view
                long last = out.size(out.dim() - 1);
                if (out.dim() == 3 && out.size(1) * out.size(2) == outFeat) {
                    return; // multi-head unconcatenated
                }
                throw new AssertionError("feat expected " + outFeat + " got last=" + last
                        + " shape=" + Arrays.toString(out.shape()));
            }
            if (out.size(0) != N && out.dim() == 2) {
                // allow different row count only if explicitly expected (none here)
                // Signed/others should still be N
                throw new AssertionError("rows expected " + N + " got " + out.size(0)
                        + " shape=" + Arrays.toString(out.shape()));
            }
        }
    }

    private static void checkFinite(Tensor out) {
        if (out.isnan().any().item().toBool()) throw new AssertionError("NaN in output");
        if (out.isinf().any().item().toBool()) throw new AssertionError("Inf in output");
    }

    private static void run(String name, Runnable body) {
        try {
            body.run();
            if (rows.stream().noneMatch(r -> r.startsWith(name + " "))) {
                rows.add(String.format(Locale.ROOT, "%-28s %8s %8s  ok", name, "PASS", "-"));
            }
            passed++;
            System.out.println("PASS  " + name);
        } catch (Throwable t) {
            failed++;
            String msg = t.getClass().getSimpleName() + ": " + t.getMessage();
            failures.add(name + " -> " + msg);
            rows.add(String.format(Locale.ROOT, "%-28s %8s %8s  %s",
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
