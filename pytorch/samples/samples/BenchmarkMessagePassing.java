package samples;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.aggr.MaxAggregation;
import org.bytedeco.pytorch.geometric.aggr.MeanAggregation;
import org.bytedeco.pytorch.geometric.aggr.MinAggregation;
import org.bytedeco.pytorch.geometric.aggr.MultiAggregation;
import org.bytedeco.pytorch.geometric.aggr.SoftmaxAggregation;
import org.bytedeco.pytorch.geometric.aggr.StdAggregation;
import org.bytedeco.pytorch.geometric.aggr.SumAggregation;
import org.bytedeco.pytorch.geometric.aggr.VarAggregation;
import org.bytedeco.pytorch.geometric.attention.PerformerAttention;
import org.bytedeco.pytorch.geometric.attention.PolynormerAttention;
import org.bytedeco.pytorch.geometric.attention.SGFormerAttention;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGCNConv;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGATConv;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGINConv;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGraphConv;
import org.bytedeco.pytorch.geometric.nn.conv.DenseSAGEConv;
import org.bytedeco.pytorch.geometric.nn.conv.FeaStConv;
import org.bytedeco.pytorch.geometric.nn.conv.GMMConv;
import org.bytedeco.pytorch.geometric.nn.conv.PointTransformerConv;
import org.bytedeco.pytorch.geometric.nn.conv.SplineConv;
import org.bytedeco.pytorch.geometric.nn.kge.ComplEx;
import org.bytedeco.pytorch.geometric.nn.kge.DistMult;
import org.bytedeco.pytorch.geometric.nn.kge.TransE;
import org.bytedeco.pytorch.geometric.nn.norm.BatchNorm;
import org.bytedeco.pytorch.geometric.nn.norm.GraphNorm;
import org.bytedeco.pytorch.geometric.nn.norm.GraphSizeNorm;
import org.bytedeco.pytorch.geometric.nn.norm.LayerNorm;
import org.bytedeco.pytorch.geometric.nn.norm.MessageNorm;
import org.bytedeco.pytorch.geometric.nn.norm.PairNorm;
import org.bytedeco.pytorch.geometric.nn.pooling.GlobalPooling;
import org.bytedeco.pytorch.geometric.nn.conv.APPNP;
import org.bytedeco.pytorch.geometric.nn.conv.ARMAConv;
import org.bytedeco.pytorch.geometric.nn.conv.CGConv;
import org.bytedeco.pytorch.geometric.nn.conv.ClusterGCNConv;
import org.bytedeco.pytorch.geometric.nn.conv.EdgeConv;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
import org.bytedeco.pytorch.geometric.nn.conv.GATv2Conv;
import org.bytedeco.pytorch.geometric.nn.conv.GCN2Conv;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
import org.bytedeco.pytorch.geometric.nn.conv.GatedGraphConv;
import org.bytedeco.pytorch.geometric.nn.conv.GENConv;
import org.bytedeco.pytorch.geometric.nn.conv.GINConv;
import org.bytedeco.pytorch.geometric.nn.conv.GraphConv;
import org.bytedeco.pytorch.geometric.nn.conv.LEConv;
import org.bytedeco.pytorch.geometric.nn.conv.LGConv;
import org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;
import org.bytedeco.pytorch.geometric.nn.conv.SGConv;
import org.bytedeco.pytorch.geometric.nn.conv.SSGConv;
import org.bytedeco.pytorch.geometric.nn.conv.SimpleConv;
import org.bytedeco.pytorch.geometric.nn.conv.TAGConv;
import org.bytedeco.pytorch.geometric.nn.conv.TransformerConv;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import java.util.Locale;

/**
 * Multi-dimensional industrial benchmark for MessagePassing + Aggregation + GraphUtils
 * and the core GNN conv layers under {@code geometric.nn.conv}.
 */
public class BenchmarkMessagePassing {

    static int passed = 0;
    static int failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable {
        void run() throws Exception;
    }

    public static void main(String[] args) throws Exception {
        Loader.load(org.bytedeco.pytorch.global.torch.class);
        System.out.println("=== MessagePassing Industrial Benchmark ===\n");

        try (PointerScope scope = new PointerScope()) {
            benchAggrCorrectness();
            benchFlow();
            benchBipartite();
            benchEdgeAttr();
            benchLayerSmoke();
            benchExtendedLayers();
            benchSpecializedAndDense();
            benchGradient();
            benchAggregationModule();
            benchExtendedAggr();
            benchGraphUtils();
            benchAttentionSanity();
            benchLinearAttention();
            benchNorms();
            benchGlobalPooling();
            benchKGE();
            benchPerf();
            benchMultiLayer();
        }

        System.out.println("\n=== Summary ===");
        System.out.println(report);
        System.out.printf(Locale.ROOT, "passed=%d failed=%d%n", passed, failed);
        if (failed > 0) {
            System.exit(1);
        }
    }

    // ------------------------------------------------------------------ helpers

    static void benchmark(String name, CheckedRunnable body) {
        System.out.println("-- " + name);
        try {
            body.run();
            System.out.println("   OK");
        } catch (Throwable t) {
            failed++;
            String msg = "FAIL [" + name + "]: " + t.getClass().getSimpleName() + ": " + t.getMessage();
            System.out.println("   " + msg);
            report.append(msg).append('\n');
            t.printStackTrace(System.out);
        }
    }

    static void check(String label, boolean cond) {
        if (cond) {
            passed++;
        } else {
            failed++;
            String msg = "  check failed: " + label;
            System.out.println(msg);
            report.append(msg).append('\n');
        }
    }

    static void checkClose(String label, double a, double b, double tol) {
        check(label + " (" + a + " vs " + b + ")", Math.abs(a - b) <= tol);
    }

    static boolean isFinite(Tensor t) {
        return !t.isnan().any().item().toBool() && !t.isinf().any().item().toBool();
    }

    static TensorOptions longOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long));
    }

    static TensorOptions floatOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float));
    }

    /** 1-D long tensor (edge indices). Requires TensorOptions. */
    static Tensor longs(long... v) {
        return torch.tensor(v, longOpts());
    }

    /** 1-D float tensor. Requires TensorOptions. */
    static Tensor floats(float... v) {
        return torch.tensor(v, floatOpts());
    }

    /**
     * Read first feature of row {@code row} (or scalar at {@code row} for 1-D).
     * Prefer {@link #at(Tensor, long, long)} for multi-feature rows.
     */
    static float at(Tensor t, long row) {
        Tensor c = t.contiguous().to(torch.ScalarType.Float).cpu();
        if (c.dim() == 0) {
            return c.item_float();
        }
        if (c.dim() == 1) {
            return c.select(0, row).item_float();
        }
        // [N, F, ...] → first feature of row
        return c.select(0, row).reshape(-1).select(0, 0).item_float();
    }

    /** Read t[row, col] for rank ≥ 2. */
    static float at(Tensor t, long row, long col) {
        return t.contiguous().to(torch.ScalarType.Float).cpu()
                .select(0, row).select(0, col).item_float();
    }

    /** Build edge_index [2,E] from parallel src/dst long arrays. */
    static Tensor edgeIndex(long[] src, long[] dst) {
        if (src.length != dst.length) {
            throw new IllegalArgumentException("src/dst length mismatch");
        }
        return torch.stack(new TensorVector(longs(src), longs(dst)), 0);
    }

    // Identity MessagePassing for raw pipeline tests
    static final class IdentityMP extends MessagePassing {
        IdentityMP(String aggr) {
            super(aggr);
        }

        IdentityMP(String aggr, String flow) {
            super(aggr, flow);
        }

        IdentityMP(org.bytedeco.pytorch.geometric.aggr.Aggregation a) {
            super(a);
        }

        @Override
        public Tensor forward(Tensor x, Tensor edge_index) {
            return propagate(edge_index, x);
        }
    }

    static final class NeedsXiMP extends MessagePassing {
        NeedsXiMP() {
            super("sum");
        }

        @Override
        protected boolean needsX_i() {
            return true;
        }

        @Override
        public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
            return x_j.add(x_i);
        }

        @Override
        public Tensor forward(Tensor x, Tensor edge_index) {
            return propagate(edge_index, x);
        }
    }

    // ================================================================== 1. Aggr

    static void benchAggrCorrectness() {
        benchmark("Aggr sum/mean/max/min + isolated", () -> {
            // edges → targets 1,2,2 with values 1,2,3; node 3 isolated
            Tensor index = longs(1, 2, 2);
            Tensor src = floats(1f, 2f, 3f).view(3, 1);
            long N = 4;

            Tensor sum = AggrUtils.scatter(src, index, N, "sum");
            check("sum shape [4,1]", sum.size(0) == 4 && sum.size(1) == 1);
            checkClose("sum[0]=0 isolated", at(sum, 0), 0f, 1e-5);
            checkClose("sum[1]=1", at(sum, 1), 1f, 1e-5);
            checkClose("sum[2]=2+3=5", at(sum, 2), 5f, 1e-5);
            checkClose("sum[3]=0 isolated", at(sum, 3), 0f, 1e-5);

            Tensor mean = AggrUtils.scatter(src, index, N, "mean");
            checkClose("mean[1]=1", at(mean, 1), 1f, 1e-5);
            checkClose("mean[2]=2.5", at(mean, 2), 2.5f, 1e-5);
            checkClose("mean[3]=0 isolated", at(mean, 3), 0f, 1e-5);

            Tensor max = AggrUtils.scatter(src, index, N, "max");
            checkClose("max[2]=3", at(max, 2), 3f, 1e-5);
            checkClose("max[3]=0 isolated", at(max, 3), 0f, 1e-5);

            Tensor min = AggrUtils.scatter(src, index, N, "min");
            checkClose("min[2]=2", at(min, 2), 2f, 1e-5);
            checkClose("min[0]=0 isolated", at(min, 0), 0f, 1e-5);

            Tensor viaFacade = Scatter.scatter(src, index, N, "add");
            checkClose("Scatter facade sum[2]", at(viaFacade, 2), 5f, 1e-5);

            Tensor viaAdd = AggrUtils.scatter(src, index, N, "add");
            checkClose("alias add≡sum", at(viaAdd, 2), 5f, 1e-5);
        });
    }

    // ================================================================== 2. Flow

    static void benchFlow() {
        benchmark("Flow source_to_target vs target_to_source", () -> {
            Tensor x = floats(10f, 20f, 0f).view(3, 1);
            Tensor ei = edgeIndex(new long[]{0}, new long[]{1});

            IdentityMP stt = new IdentityMP("sum", "source_to_target");
            Tensor outStt = stt.propagate(ei, x);
            checkClose("stt out[1]=10", at(outStt, 1), 10f, 1e-5);
            checkClose("stt out[0]=0", at(outStt, 0), 0f, 1e-5);

            IdentityMP tts = new IdentityMP("sum", "target_to_source");
            Tensor outTts = tts.propagate(ei, x);
            checkClose("tts out[0]=20", at(outTts, 0), 20f, 1e-5);
            checkClose("tts out[1]=0", at(outTts, 1), 0f, 1e-5);
        });
    }

    // ================================================================== 3. Bipartite

    static void benchBipartite() {
        benchmark("Bipartite size N_src != N_dst", () -> {
            Tensor xSrc = floats(1f, 2f, 3f).view(3, 1);
            Tensor xDst = torch.zeros(new long[]{5, 1}, floatOpts());
            Tensor ei = edgeIndex(new long[]{0, 1, 2}, new long[]{0, 4, 4});

            IdentityMP mp = new IdentityMP("sum");
            Tensor out = mp.propagate(ei, xSrc, xDst, (Tensor) null, new long[]{3, 5});
            check("bipartite out rows=5", out.size(0) == 5);
            checkClose("dst0=1", at(out, 0), 1f, 1e-5);
            checkClose("dst4=2+3=5", at(out, 4), 5f, 1e-5);
            checkClose("dst1 isolated 0", at(out, 1), 0f, 1e-5);

            Tensor x = torch.arange(new Scalar(0), new Scalar(3), floatOpts())
                    .to(torch.ScalarType.Float).view(3, 1);
            Tensor ei2 = edgeIndex(new long[]{0, 1}, new long[]{0, 3});
            Tensor out2 = mp.propagate(ei2, x, new long[]{3, 4});
            check("size override rows=4", out2.size(0) == 4);
            checkClose("size override dst3", at(out2, 3), 1f, 1e-5);
        });
    }

    // ================================================================== 4. Edge attr

    static void benchEdgeAttr() {
        benchmark("Edge attr / weight multiply", () -> {
            Tensor x = torch.ones(new long[]{3, 2}, floatOpts());
            Tensor ei = edgeIndex(new long[]{0, 1}, new long[]{1, 2});
            Tensor w = floats(2f, 3f);

            IdentityMP mp = new IdentityMP("sum");
            Tensor out = mp.propagate(ei, x, w);
            checkClose("weighted n1 f0", at(out, 1, 0), 2f, 1e-4);
            checkClose("weighted n2 f1", at(out, 2, 1), 3f, 1e-4);
        });
    }

    // ================================================================== 5. Layer smoke

    static void benchLayerSmoke() {
        benchmark("Layer smoke GCN/SAGE/GAT/GIN/GraphConv/SimpleConv", () -> {
            long N = 8, F = 4, E = 16;
            Tensor x = torch.randn(new long[]{N, F}, floatOpts());
            Tensor ei = torch.randint(N, new long[]{2, E}, longOpts());

            GCNConv gcn = new GCNConv(F, 6);
            Tensor yGcn = gcn.forward(x, ei);
            check("GCN out [8,6]", yGcn.size(0) == N && yGcn.size(1) == 6);
            check("GCN finite", isFinite(yGcn));

            SAGEConv sage = new SAGEConv(F, 6, false, true);
            Tensor ySage = sage.forward(x, ei);
            check("SAGE out [8,6]", ySage.size(0) == N && ySage.size(1) == 6);
            check("SAGE finite", isFinite(ySage));

            Tensor xSrc = torch.randn(new long[]{5, F}, floatOpts());
            Tensor xDst = torch.randn(new long[]{7, F}, floatOpts());
            Tensor eiB = edgeIndex(new long[]{0, 1, 2, 3}, new long[]{0, 1, 6, 6});
            Tensor ySageB = sage.forward(xSrc, xDst, eiB);
            check("SAGE bipartite [7,6]", ySageB.size(0) == 7 && ySageB.size(1) == 6);

            GATConv gat = new GATConv(F, 3, 2, true, 0.2);
            Tensor yGat = gat.forward(x, ei);
            check("GAT concat out [8,6]", yGat.size(0) == N && yGat.size(1) == 6);
            check("GAT finite", isFinite(yGat));

            SequentialImpl mlp = new SequentialImpl();
            mlp.push_back(new LinearImpl(F, 6));
            GINConv gin = new GINConv(mlp, false);
            Tensor yGin = gin.forward(x, ei);
            check("GIN out [8,6]", yGin.size(0) == N && yGin.size(1) == 6);
            check("GIN finite", isFinite(yGin));

            GraphConv gc = new GraphConv(F, 6);
            Tensor yGc = gc.forward(x, ei);
            check("GraphConv out [8,6]", yGc.size(0) == N && yGc.size(1) == 6);
            check("GraphConv finite", isFinite(yGc));

            Tensor yGcB = gc.forwardBipartite(xSrc, xDst, eiB);
            check("GraphConv bipartite [7,6]", yGcB.size(0) == 7 && yGcB.size(1) == 6);

            // Hetero-style 3-arg dispatch (xSrc, xDst, edge_index)
            GraphConv gcH = new GraphConv(F, F, 6);
            Tensor yGcH = gcH.forward(xSrc, xDst, eiB);
            check("GraphConv hetero dispatch [7,6]", yGcH.size(0) == 7 && yGcH.size(1) == 6);

            SimpleConv simple = new SimpleConv("mean", "sum");
            Tensor ySimple = simple.forward(x, ei);
            check("SimpleConv out [8,4]", ySimple.size(0) == N && ySimple.size(1) == F);
            check("SimpleConv finite", isFinite(ySimple));

            NeedsXiMP nxi = new NeedsXiMP();
            Tensor yXi = nxi.forward(x, ei);
            check("needsX_i finite", isFinite(yXi));
        });
    }

    // ================================================================== 5b. Extended layers

    static void benchExtendedLayers() {
        benchmark("Extended layers APPNP/SG/SSG/GCN2/LE/LG/Edge/Gated/Cluster/GATv2/Transformer", () -> {
            long N = 12, F = 8, E = 30;
            Tensor x = torch.randn(new long[]{N, F}, floatOpts());
            Tensor ei = torch.randint(N, new long[]{2, E}, longOpts());

            APPNP appnp = new APPNP(5, 0.1, 0.0, true, true);
            Tensor y1 = appnp.forward(x, ei);
            check("APPNP shape", y1.size(0) == N && y1.size(1) == F);
            check("APPNP finite", isFinite(y1));

            SGConv sg = new SGConv(F, 16, 2);
            Tensor y2 = sg.forward(x, ei);
            check("SGConv shape", y2.size(0) == N && y2.size(1) == 16);
            check("SGConv finite", isFinite(y2));

            SSGConv ssg = new SSGConv(F, 16, 0.05, 2);
            Tensor y3 = ssg.forward(x, ei);
            check("SSGConv shape", y3.size(0) == N && y3.size(1) == 16);
            check("SSGConv finite", isFinite(y3));

            GCN2Conv gcn2 = new GCN2Conv(F, 0.1f, 0.5f, 1, true, true);
            Tensor y4 = gcn2.forward(x, x, ei, null);
            check("GCN2 shape", y4.size(0) == N && y4.size(1) == F);
            check("GCN2 finite", isFinite(y4));
            // 2-arg convenience
            Tensor y4b = gcn2.forward(x, ei);
            check("GCN2 2-arg finite", isFinite(y4b));

            LEConv le = new LEConv(F, 16);
            Tensor y5 = le.forward(x, ei);
            check("LEConv shape", y5.size(0) == N && y5.size(1) == 16);
            check("LEConv finite", isFinite(y5));

            LGConv lg = new LGConv(true);
            Tensor y6 = lg.forward(x, ei);
            check("LGConv shape", y6.size(0) == N && y6.size(1) == F);
            check("LGConv finite", isFinite(y6));

            EdgeConv edge = new EdgeConv(F, 16);
            Tensor y7 = edge.forward(x, ei);
            check("EdgeConv shape", y7.size(0) == N && y7.size(1) == 16);
            check("EdgeConv finite", isFinite(y7));

            GatedGraphConv gated = new GatedGraphConv(F, 2);
            Tensor y8 = gated.forward(x, ei);
            check("GatedGraph shape", y8.size(0) == N && y8.size(1) == F);
            check("GatedGraph finite", isFinite(y8));

            ClusterGCNConv cluster = new ClusterGCNConv(F, 16, 0.5f, true, true);
            Tensor y9 = cluster.forward(x, ei);
            check("ClusterGCN shape", y9.size(0) == N && y9.size(1) == 16);
            check("ClusterGCN finite", isFinite(y9));

            GATv2Conv gatv2 = new GATv2Conv(F, 4, 2, true, 0.2, null, true);
            Tensor y10 = gatv2.forward(x, ei);
            check("GATv2 shape", y10.size(0) == N && y10.size(1) == 8);
            check("GATv2 finite", isFinite(y10));

            TransformerConv tr = new TransformerConv(F, 4, 2);
            Tensor y11 = tr.forward(x, ei);
            check("TransformerConv shape", y11.size(0) == N && y11.size(1) == 8);
            check("TransformerConv finite", isFinite(y11));

            TAGConv tag = new TAGConv(F, 16, 2);
            Tensor y12 = tag.forward(x, ei);
            check("TAGConv shape", y12.size(0) == N && y12.size(1) == 16);
            check("TAGConv finite", isFinite(y12));

            ARMAConv arma = new ARMAConv(F, 16, 2, 1);
            Tensor y13 = arma.forward(x, ei);
            check("ARMAConv shape", y13.size(0) == N && y13.size(1) == 16);
            check("ARMAConv finite", isFinite(y13));

            // CGConv with edge features
            int edgeDim = 4;
            CGConv cg = new CGConv(F, edgeDim, "mean", true, true);
            Tensor eattr = torch.randn(new long[]{E, edgeDim}, floatOpts());
            Tensor y14 = cg.forward(x, ei, eattr);
            check("CGConv shape", y14.size(0) == N && y14.size(1) == F);
            check("CGConv finite", isFinite(y14));

            GENConv gen = new GENConv(F, 16);
            Tensor y15 = gen.forward(x, ei);
            check("GENConv shape", y15.size(0) == N && y15.size(1) == 16);
            check("GENConv finite", isFinite(y15));
        });
    }

    // ================================================================== 6. Gradient

    static void benchGradient() {
        benchmark("Gradient step GCN + SAGE", () -> {
            long N = 10, F = 4, E = 20;
            Tensor x = torch.randn(new long[]{N, F}, floatOpts());
            Tensor ei = torch.randint(N, new long[]{2, E}, longOpts());
            Tensor target = torch.randn(new long[]{N, 3}, floatOpts());

            GCNConv gcn = new GCNConv(F, 3);
            Adam opt = new Adam(gcn.parameters(), new AdamOptions(1e-2));
            opt.zero_grad();
            Tensor out = gcn.forward(x, ei);
            Tensor loss = out.sub(target).pow(new Scalar(2)).mean();
            loss.backward();
            boolean anyGrad = false;
            TensorVector params = gcn.parameters();
            for (long i = 0; i < params.size(); i++) {
                Tensor p = params.get(i);
                if (p.grad() != null && p.grad().defined() && p.grad().numel() > 0) {
                    if (p.grad().abs().sum().item().toFloat() > 0) {
                        anyGrad = true;
                        break;
                    }
                }
            }
            check("GCN has nonzero grad", anyGrad);
            opt.step();

            SAGEConv sage = new SAGEConv(F, 3, false, true, true, false);
            Adam opt2 = new Adam(sage.parameters(), new AdamOptions(1e-2));
            opt2.zero_grad();
            Tensor out2 = sage.forward(x, ei);
            Tensor loss2 = out2.sub(target).pow(new Scalar(2)).mean();
            loss2.backward();
            boolean anyGrad2 = false;
            TensorVector params2 = sage.parameters();
            for (long i = 0; i < params2.size(); i++) {
                Tensor p = params2.get(i);
                if (p.grad() != null && p.grad().defined() && p.grad().numel() > 0) {
                    if (p.grad().abs().sum().item().toFloat() > 0) {
                        anyGrad2 = true;
                        break;
                    }
                }
            }
            check("SAGE has nonzero grad", anyGrad2);
            opt2.step();
        });
    }

    // ================================================================== 7. Aggr module

    static void benchAggregationModule() {
        benchmark("Aggregation module vs string reduce", () -> {
            Tensor x = floats(1f, 2f, 3f, 4f).view(4, 1);
            Tensor ei = edgeIndex(new long[]{0, 1, 2, 3}, new long[]{0, 0, 1, 1});

            IdentityMP strSum = new IdentityMP("sum");
            Tensor a = strSum.propagate(ei, x);

            IdentityMP modSum = new IdentityMP(new SumAggregation());
            Tensor b = modSum.propagate(ei, x);
            checkClose("string sum == SumAggregation [0]", at(a, 0), at(b, 0), 1e-5);
            checkClose("string sum == SumAggregation [1]", at(a, 1), at(b, 1), 1e-5);

            IdentityMP meanMp = new IdentityMP(new MeanAggregation());
            Tensor m = meanMp.propagate(ei, x);
            checkClose("MeanAggregation node0", at(m, 0), 1.5f, 1e-5);
            checkClose("MeanAggregation node1", at(m, 1), 3.5f, 1e-5);

            // MultiAggregation concatenates [sum | max] on feature dim
            // node0: msgs {1,2} → sum=3, max=2; node1: msgs {3,4} → sum=7, max=4
            MultiAggregation multi = new MultiAggregation(new SumAggregation(), new MaxAggregation());
            IdentityMP multiMp = new IdentityMP(multi);
            Tensor mu = multiMp.propagate(ei, x);
            check("MultiAggregation feat dim=2", mu.size(1) == 2);
            checkClose("Multi n0 sum", at(mu, 0, 0), 3f, 1e-5);
            checkClose("Multi n0 max", at(mu, 0, 1), 2f, 1e-5);
            checkClose("Multi n1 sum", at(mu, 1, 0), 7f, 1e-5);
            checkClose("Multi n1 max", at(mu, 1, 1), 4f, 1e-5);
        });
    }

    // ================================================================== 8. GraphUtils

    static void benchGraphUtils() {
        benchmark("GraphUtils self-loops / degree / gcn_norm", () -> {
            Tensor ei = edgeIndex(new long[]{0, 1}, new long[]{1, 0});
            long N = 3;
            Tensor withLoops = GraphUtils.add_self_loops(ei, N);
            check("self-loops E=2+3=5", withLoops.size(1) == 5);

            Tensor[] pair = GraphUtils.add_self_loops(ei, torch.ones(new long[]{2}, floatOpts()), N, 1.0);
            check("weighted loops E", pair[0].size(1) == 5);
            check("weighted loops W", pair[1].size(0) == 5);

            Tensor deg = GraphUtils.degree(ei.select(0, 1), N);
            checkClose("degree[0]", at(deg, 0), 1f, 1e-5);
            checkClose("degree[2] isolated", at(deg, 2), 0f, 1e-5);

            Tensor[] normed = GraphUtils.gcn_norm(ei, null, N, true, torch.ScalarType.Float);
            check("gcn_norm edge rows=2", normed[0].size(0) == 2);
            check("gcn_norm weight len", normed[1].size(0) == normed[0].size(1));
            check("gcn_norm finite", isFinite(normed[1]));

            Tensor removed = GraphUtils.remove_self_loops(withLoops);
            check("remove_self_loops back to 2 edges", removed.size(1) == 2);

            long[] bip = GraphUtils.bipartite_size(ei, null, null);
            check("bipartite_size infers", bip[0] >= 2 && bip[1] >= 2);
        });
    }

    // ================================================================== 11. Attention

    static void benchAttentionSanity() {
        benchmark("GAT attention softmax groups sum ≈ 1", () -> {
            AttentionProbe probe = new AttentionProbe(4, 2, 1);
            Tensor x = torch.randn(new long[]{5, 4}, floatOpts());
            Tensor ei = edgeIndex(new long[]{1, 2, 3, 4}, new long[]{0, 0, 0, 0});
            Tensor out = probe.forward(x, ei);
            check("probe out finite", isFinite(out));
            check("probe recorded alpha", probe.lastAlphaSum > 0);
            checkClose("alpha sum per target ≈ 1",
                    probe.lastAlphaSum / Math.max(1, probe.lastNumTargets), 1.0, 0.15);
        });
    }

    static final class AttentionProbe extends MessagePassing {
        final LinearImpl lin;
        final Tensor att;
        final long heads;
        final long outChannels;
        double lastAlphaSum = 0;
        long lastNumTargets = 0;

        AttentionProbe(long in, long out, long heads) {
            super("sum");
            this.heads = heads;
            this.outChannels = out;
            this.lin = register_module("lin", new LinearImpl(in, heads * out));
            Tensor a = torch.randn(new long[]{1, heads, 2 * out});
            torch.xavier_uniform_(a);
            this.att = a.clone();
            register_parameter("att", this.att);
        }

        @Override
        protected boolean needsX_i() {
            return true;
        }

        @Override
        public Tensor forward(Tensor x, Tensor edge_index) {
            long N = x.size(0);
            Tensor ei = GraphUtils.add_self_loops(edge_index, N);
            Tensor xLin = lin.forward(x).view(N, heads, outChannels);
            Tensor out = propagate(ei, xLin);
            return out.view(N, heads * outChannels);
        }

        @Override
        public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
            Tensor targetIdx = _index_i != null ? _index_i : edge_index.select(0, 1);
            targetIdx = AggrUtils.asLongIndex(targetIdx);
            Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1);
            Tensor alpha = catFeat.mul(this.att).sum(-1);
            alpha = torch.leaky_relu(alpha, new Scalar(0.2));
            alpha = AggrUtils.scatter_softmax(alpha, targetIdx, numNodes);
            lastAlphaSum = alpha.sum().item().toDouble();
            Tensor ones = torch.ones(new long[]{targetIdx.size(0)}, floatOpts());
            Tensor presence = torch.zeros(new long[]{numNodes}, floatOpts());
            presence.index_add_(0, targetIdx, ones);
            lastNumTargets = presence.gt(new Scalar(0)).sum().item().toLong() * heads;
            return x_j.mul(alpha.unsqueeze(-1));
        }
    }

    // ================================================================== 9. Perf

    static void benchPerf() {
        benchmark("Perf grid propagate + GCN forward", () -> {
            long[][] grid = {
                    {500, 2000, 32},
                    {2000, 10000, 64},
            };
            for (long[] g : grid) {
                long N = g[0], E = g[1], F = g[2];
                Tensor x = torch.randn(new long[]{N, F}, floatOpts());
                Tensor ei = torch.randint(N, new long[]{2, E}, longOpts());

                IdentityMP mp = new IdentityMP("sum");
                mp.propagate(ei, x); // warmup

                long t0 = System.nanoTime();
                int reps = 10;
                Tensor last = null;
                for (int i = 0; i < reps; i++) {
                    last = mp.propagate(ei, x);
                }
                double msProp = (System.nanoTime() - t0) / 1e6 / reps;
                check("perf propagate finite N=" + N, last != null && isFinite(last));

                GCNConv gcn = new GCNConv(F, F);
                gcn.forward(x, ei);
                t0 = System.nanoTime();
                for (int i = 0; i < reps; i++) {
                    last = gcn.forward(x, ei);
                }
                double msGcn = (System.nanoTime() - t0) / 1e6 / reps;
                check("perf GCN finite N=" + N, last != null && isFinite(last));

                String line = String.format(Locale.ROOT,
                        "  N=%d E=%d F=%d  propagate=%.3f ms  GCN=%.3f ms",
                        N, E, F, msProp, msGcn);
                System.out.println(line);
                report.append(line).append('\n');
            }
        });
    }

    // ================================================================== 10. Multi-layer

    static void benchMultiLayer() {
        benchmark("Multi-layer GCN/SAGE train step timing", () -> {
            long N = 512, F = 32, E = 2048, H = 64, C = 10;
            Tensor x = torch.randn(new long[]{N, F}, floatOpts());
            Tensor ei = torch.randint(N, new long[]{2, E}, longOpts());

            GCNConv gcn1 = new GCNConv(F, H);
            GCNConv gcn2 = new GCNConv(H, C);
            long t0 = System.nanoTime();
            int reps = 5;
            for (int i = 0; i < reps; i++) {
                Tensor h = torch.relu(gcn1.forward(x, ei));
                Tensor logits = gcn2.forward(h, ei);
                Tensor target = torch.randn(new long[]{N, C}, floatOpts());
                Tensor loss = logits.sub(target).pow(new Scalar(2)).mean();
                check("multilayer finite", isFinite(loss));
            }
            double ms = (System.nanoTime() - t0) / 1e6 / reps;
            String line = String.format(Locale.ROOT, "  2-layer GCN forward avg=%.3f ms", ms);
            System.out.println(line);
            report.append(line).append('\n');

            SAGEConv s1 = new SAGEConv(F, H, false, true, true, true);
            Adam opt = new Adam(s1.parameters(), new AdamOptions(1e-3));
            t0 = System.nanoTime();
            for (int i = 0; i < reps; i++) {
                opt.zero_grad();
                Tensor out = s1.forward(x, ei);
                Tensor loss = out.mean();
                loss.backward();
                opt.step();
            }
            ms = (System.nanoTime() - t0) / 1e6 / reps;
            line = String.format(Locale.ROOT, "  SAGE train-step avg=%.3f ms", ms);
            System.out.println(line);
            report.append(line).append('\n');
            check("multilayer train ran", true);
        });
    }

    // ================================================================== Specialized + Dense

    static void benchSpecializedAndDense() {
        benchmark("Specialized GMM/FeaSt/Spline/PointTransformer + Dense*", () -> {
            long N = 10, F = 8, E = 24, dim = 2, K = 4;
            Tensor x = torch.randn(new long[]{N, F}, floatOpts());
            Tensor ei = torch.randint(N, new long[]{2, E}, longOpts());
            Tensor pseudo = torch.rand(new long[]{E, dim}, floatOpts());

            GMMConv gmm = new GMMConv(F, 16, (int) dim, (int) K, true, true);
            Tensor yGmm = gmm.forward(x, ei, pseudo);
            check("GMMConv shape", yGmm.size(0) == N && yGmm.size(1) == 16);
            check("GMMConv finite", isFinite(yGmm));

            FeaStConv feast = new FeaStConv(F, 16, 4, true);
            Tensor yFeast = feast.forward(x, ei);
            check("FeaStConv shape", yFeast.size(0) == N && yFeast.size(1) == 16);
            check("FeaStConv finite", isFinite(yFeast));

            SplineConv spline = new SplineConv(F, 16, (int) dim, 4, 1, true, true);
            Tensor ySpline = spline.forward(x, ei, pseudo);
            check("SplineConv shape", ySpline.size(0) == N && ySpline.size(1) == 16);
            check("SplineConv finite", isFinite(ySpline));

            SequentialImpl posNN = new SequentialImpl();
            posNN.push_back(new LinearImpl(3, 16));
            SequentialImpl attnNN = new SequentialImpl();
            attnNN.push_back(new LinearImpl(16, 16));
            PointTransformerConv ptc = new PointTransformerConv(F, 16, 3, posNN, attnNN);
            Tensor pos = torch.randn(new long[]{N, 3}, floatOpts());
            Tensor yPt = ptc.forward(x, pos, ei);
            check("PointTransformer shape", yPt.size(0) == N && yPt.size(1) == 16);
            check("PointTransformer finite", isFinite(yPt));

            // Dense batch
            long B = 2, Nd = 6, Fd = 8;
            Tensor xd = torch.randn(new long[]{B, Nd, Fd}, floatOpts());
            Tensor adj = torch.rand(new long[]{B, Nd, Nd}, floatOpts());
            // Symmetrize lightly for stability
            adj = adj.add(adj.transpose(1, 2)).mul(new Scalar(0.5));

            DenseGCNConv dgcn = new DenseGCNConv(Fd, 12, true);
            Tensor yd1 = dgcn.forward(xd, adj);
            check("DenseGCN shape", yd1.size(0) == B && yd1.size(1) == Nd && yd1.size(2) == 12);
            check("DenseGCN finite", isFinite(yd1));

            DenseSAGEConv dsage = new DenseSAGEConv(Fd, 12, true);
            Tensor yd2 = dsage.forward(xd, adj);
            check("DenseSAGE shape", yd2.size(0) == B && yd2.size(2) == 12);
            check("DenseSAGE finite", isFinite(yd2));

            DenseGraphConv dgc = new DenseGraphConv(Fd, 12);
            Tensor yd3 = dgc.forward(xd, adj);
            check("DenseGraph shape", yd3.size(0) == B && yd3.size(2) == 12);
            check("DenseGraph finite", isFinite(yd3));

            SequentialImpl mlp = new SequentialImpl();
            mlp.push_back(new LinearImpl(Fd, 12));
            DenseGINConv dgin = new DenseGINConv(mlp, 0.1, false);
            Tensor yd4 = dgin.forward(xd, adj);
            check("DenseGIN shape", yd4.size(0) == B && yd4.size(2) == 12);
            check("DenseGIN finite", isFinite(yd4));

            DenseGATConv dgat = new DenseGATConv(Fd, 4, 2, true, 0.2);
            Tensor yd5 = dgat.forward(xd, adj);
            check("DenseGAT shape", yd5.size(0) == B && yd5.size(2) == 8);
            check("DenseGAT finite", isFinite(yd5));
        });
    }

    // ================================================================== Extended Aggr

    static void benchExtendedAggr() {
        benchmark("Extended Aggregation Softmax/Var/Std/Min", () -> {
            Tensor x = floats(1f, 2f, 3f, 4f, 5f, 6f).view(6, 1);
            Tensor index = longs(0, 0, 0, 1, 1, 1);

            SoftmaxAggregation soft = new SoftmaxAggregation(1, false);
            Tensor ys = soft.forward(x, index, 2);
            check("SoftmaxAggr shape", ys.size(0) == 2 && ys.size(1) == 1);
            check("SoftmaxAggr finite", isFinite(ys));
            // Softmax weighted sum should be between min and max of group
            check("SoftmaxAggr in range n0", at(ys, 0) >= 1f - 1e-3 && at(ys, 0) <= 3f + 1e-3);

            VarAggregation varA = new VarAggregation();
            Tensor yv = varA.forward(x, index, 2);
            check("VarAggr nonneg", at(yv, 0) >= -1e-5 && at(yv, 1) >= -1e-5);

            StdAggregation stdA = new StdAggregation();
            Tensor ystd = stdA.forward(x, index, 2);
            check("StdAggr nonneg", at(ystd, 0) >= -1e-5);
            checkClose("Std ~ sqrt(Var)", at(ystd, 0), Math.sqrt(Math.max(0, at(yv, 0))), 1e-3);

            MinAggregation minA = new MinAggregation();
            Tensor ymin = minA.forward(x, index, 2);
            checkClose("MinAggr n0", at(ymin, 0), 1f, 1e-5);
            checkClose("MinAggr n1", at(ymin, 1), 4f, 1e-5);
        });
    }

    // ================================================================== Linear Attention

    static void benchLinearAttention() {
        benchmark("Linear Attention Performer/Polynormer/SGFormer", () -> {
            long N = 32, C = 16, H = 4;
            Tensor x = torch.randn(new long[]{N, C}, floatOpts());

            PerformerAttention perf = new PerformerAttention(C, H, 32);
            Tensor yp = perf.forward(x);
            check("Performer shape", yp.size(0) == N && yp.size(1) == C);
            check("Performer finite", isFinite(yp));

            PolynormerAttention poly = new PolynormerAttention(C, H);
            Tensor yo = poly.forward(x);
            check("Polynormer shape", yo.size(0) == N && yo.size(1) == C);
            check("Polynormer finite", isFinite(yo));

            SGFormerAttention sg = new SGFormerAttention(C, H);
            Tensor ys = sg.forward(x);
            check("SGFormer shape", ys.size(0) == N && ys.size(1) == C);
            check("SGFormer finite", isFinite(ys));
        });
    }

    // ================================================================== Norms

    static void benchNorms() {
        benchmark("Norms Layer/Graph/Pair/Message/Batch/GraphSize", () -> {
            int N = 20;
            long C = 8;
            Tensor x = torch.randn(new long[]{N, C}, floatOpts());
            // Two graphs of 10 nodes
            long[] b = new long[N];
            for (int i = 0; i < N; i++) b[i] = i < 10 ? 0 : 1;
            Tensor batch = longs(b);

            LayerNorm ln = new LayerNorm(C);
            Tensor y1 = ln.forward(x);
            check("LayerNorm shape", y1.size(0) == N && y1.size(1) == C);
            check("LayerNorm finite", isFinite(y1));
            // Feature mean ≈ 0
            Tensor featMean = y1.mean(new long[]{1}, false, new ScalarTypeOptional());
            checkClose("LayerNorm row-mean~0", Math.abs(featMean.mean().item_float()), 0.0, 0.15);

            GraphNorm gn = new GraphNorm(C);
            Tensor y2 = gn.forward(x, batch);
            check("GraphNorm shape", y2.size(0) == N);
            check("GraphNorm finite", isFinite(y2));

            PairNorm pn = new PairNorm(1.0, false);
            Tensor y3 = pn.forward(x, batch);
            check("PairNorm finite", isFinite(y3));

            MessageNorm mn = new MessageNorm(1.0);
            Tensor msg = torch.randn(new long[]{N, C}, floatOpts());
            Tensor y4 = mn.forward(x, msg);
            check("MessageNorm finite", isFinite(y4));

            BatchNorm bn = new BatchNorm(C, true);
            bn.eval();
            Tensor y5 = bn.forward(x);
            check("BatchNorm shape", y5.size(0) == N && y5.size(1) == C);
            check("BatchNorm finite", isFinite(y5));

            GraphSizeNorm gsn = new GraphSizeNorm();
            Tensor y6 = gsn.forward(x, batch);
            check("GraphSizeNorm finite", isFinite(y6));
        });
    }

    // ================================================================== Global Pooling

    static void benchGlobalPooling() {
        benchmark("GlobalPooling sum/mean/max", () -> {
            // Graph0: nodes 0,1  Graph1: nodes 2,3,4
            Tensor x = floats(1f, 2f, 3f, 4f, 5f).view(5, 1);
            Tensor batch = longs(0, 0, 1, 1, 1);

            Tensor sum = GlobalPooling.global_add_pool(x, batch);
            check("pool sum shape", sum.size(0) == 2);
            checkClose("pool sum g0", at(sum, 0), 3f, 1e-5); // 1+2
            checkClose("pool sum g1", at(sum, 1), 12f, 1e-5); // 3+4+5

            Tensor mean = GlobalPooling.global_mean_pool(x, batch);
            checkClose("pool mean g0", at(mean, 0), 1.5f, 1e-5);
            checkClose("pool mean g1", at(mean, 1), 4f, 1e-5);

            Tensor max = GlobalPooling.global_max_pool(x, batch);
            checkClose("pool max g0", at(max, 0), 2f, 1e-5);
            checkClose("pool max g1", at(max, 1), 5f, 1e-5);

            Tensor single = GlobalPooling.pool(x, null, "sum");
            check("pool single-graph shape", single.size(0) == 1);
            checkClose("pool single sum", at(single, 0), 15f, 1e-5);
        });
    }

    // ================================================================== KGE

    static void benchKGE() {
        benchmark("KGE TransE/DistMult/ComplEx scoring", () -> {
            long numE = 50, numR = 10, d = 16, B = 8;
            Tensor head = torch.randint(numE, new long[]{B}, longOpts());
            Tensor rel = torch.randint(numR, new long[]{B}, longOpts());
            Tensor tail = torch.randint(numE, new long[]{B}, longOpts());

            TransE transe = new TransE(numE, numR, d, 1);
            Tensor s1 = transe.forward(head, rel, tail);
            check("TransE score shape", s1.size(0) == B);
            check("TransE finite", isFinite(s1));

            DistMult dm = new DistMult(numE, numR, d);
            Tensor s2 = dm.forward(head, rel, tail);
            check("DistMult score shape", s2.size(0) == B);
            check("DistMult finite", isFinite(s2));

            ComplEx cx = new ComplEx(numE, numR, d);
            Tensor s3 = cx.forward(head, rel, tail);
            check("ComplEx score shape", s3.size(0) == B);
            check("ComplEx finite", isFinite(s3));

            // Margin loss sanity
            Tensor negTail = torch.randint(numE, new long[]{B}, longOpts());
            Tensor pos = dm.forward(head, rel, tail);
            Tensor neg = dm.forward(head, rel, negTail);
            Tensor loss = dm.loss(pos, neg, 1.0);
            check("KGE margin loss finite", isFinite(loss));
            check("KGE margin loss scalar", loss.dim() == 0 || loss.numel() == 1);
        });
    }
}
