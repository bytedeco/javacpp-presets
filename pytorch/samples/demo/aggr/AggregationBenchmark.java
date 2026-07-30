package samples.demo.aggr;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.aggr.*;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

/**
 * Smoke + micro-benchmark for every concrete Aggregation under
 * {@code org.bytedeco.pytorch.geometric.aggr}.
 *
 * <p>Does not invent APIs: only uses public constructors and
 * {@link Aggregation#forward(Tensor, Tensor, long)}.</p>
 */
public class AggregationBenchmark {

    private static final int WARMUP = 2;
    private static final int ITERS = 8;
    private static final long FEATURES = 16;
    private static final long NUM_NODES = 64;
    private static final long NUM_EDGES = 256;

    private static int passed = 0;
    private static int failed = 0;
    private static final List<String> failures = new ArrayList<>();
    private static final List<String> rows = new ArrayList<>();

    public static void main(String[] args) {
        torch.manual_seed(42);
        System.out.println("=== Aggregation Benchmark ===");
        System.out.printf(Locale.ROOT,
                "nodes=%d edges=%d features=%d warmup=%d iters=%d%n%n",
                NUM_NODES, NUM_EDGES, FEATURES, WARMUP, ITERS);

        try (PointerScope scope = new PointerScope()) {
            Tensor x = torch.randn(NUM_EDGES, FEATURES);
            // index must be long; torch.tensor(long[]) already yields Long dtype
            long[] idxData = new long[(int) NUM_EDGES];
            java.util.Random rnd = new java.util.Random(42);
            for (int i = 0; i < idxData.length; i++) {
                idxData[i] = rnd.nextInt((int) NUM_NODES);
            }
            Tensor index = torch.tensor(idxData);
            // Ensure isolated nodes exist for zero-fill policy checks.
            Tensor smallX = torch.tensor(new float[]{
                    1, 1, 1, 1,
                    2, 2, 2, 2,
                    3, 3, 3, 3,
                    4, 4, 4, 4,
                    5, 5, 5, 5
            }).reshape(5, 4);
            Tensor smallIndex = torch.tensor(new long[]{0, 0, 1, 1, 1});
            long smallDim = 3; // node 2 isolated

            // --- basic reduce family ---
            bench("SumAggregation", () -> new SumAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("MeanAggregation", () -> new MeanAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("MaxAggregation", () -> new MaxAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("MinAggregation", () -> new MinAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("MulAggregation", () -> new MulAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("StdAggregation", () -> new StdAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("VarAggregation", () -> new VarAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("MedianAggregation", () -> new MedianAggregation(), x, index, NUM_NODES,
                    out -> expectShape(out, NUM_NODES, FEATURES));
            bench("VariancePreservingAggregation", () -> new VariancePreservingAggregation(),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));

            // correctness: Sum on small graph
            run("SumAggregation.correctness", () -> {
                Tensor out = new SumAggregation().forward(smallX, smallIndex, smallDim);
                float r0 = out.select(0, 0).select(0, 0).item().toFloat();
                float r1 = out.select(0, 1).select(0, 0).item().toFloat();
                float r2 = out.select(0, 2).abs().sum().item().toFloat();
                if (Math.abs(r0 - 3f) > 1e-4f) throw new AssertionError("sum g0 expected 3 got " + r0);
                if (Math.abs(r1 - 12f) > 1e-4f) throw new AssertionError("sum g1 expected 12 got " + r1);
                if (r2 > 1e-5f) throw new AssertionError("isolated should be 0 got " + r2);
            });

            run("MeanAggregation.correctness", () -> {
                Tensor out = new MeanAggregation().forward(smallX, smallIndex, smallDim);
                float r0 = out.select(0, 0).select(0, 0).item().toFloat();
                float r1 = out.select(0, 1).select(0, 0).item().toFloat();
                if (Math.abs(r0 - 1.5f) > 1e-4f) throw new AssertionError("mean g0 expected 1.5 got " + r0);
                if (Math.abs(r1 - 4f) > 1e-4f) throw new AssertionError("mean g1 expected 4 got " + r1);
            });

            // Softmax / PowerMean
            bench("SoftmaxAggregation", () -> new SoftmaxAggregation(FEATURES, true),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("PowerMeanAggregation", () -> new PowerMeanAggregation(FEATURES, true),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));

            run("PowerMeanAggregation.p1_equals_mean", () -> {
                Tensor mean = new MeanAggregation().forward(smallX, smallIndex, smallDim);
                Tensor power = new PowerMeanAggregation(4, true).forward(smallX, smallIndex, smallDim);
                float diff = power.sub(mean).abs().sum().item().toFloat();
                if (diff > 1e-4f) throw new AssertionError("PowerMean(p=1) != Mean, diff=" + diff);
            });

            bench("QuantileAggregation", () -> new QuantileAggregation(0.5),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("SortAggregation", () -> new SortAggregation(4),
                    x, index, NUM_NODES, out -> {
                        // SortAggregation concatenates top-k features → [N, k*F]
                        if (out.size(0) != NUM_NODES) {
                            throw new AssertionError("rows=" + out.size(0));
                        }
                        if (out.size(1) != 4 * FEATURES) {
                            throw new AssertionError("cols expected " + (4 * FEATURES) + " got " + out.size(1));
                        }
                    });

            // learnable / sequential aggregators
            bench("MLPAggregation", () -> new MLPAggregation(FEATURES, FEATURES, "mean"),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("AttentionalAggregation", () -> new AttentionalAggregation(FEATURES),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("LSTMAggregation", () -> new LSTMAggregation(FEATURES, FEATURES),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("GRUAggregation", () -> new GRUAggregation(FEATURES, FEATURES),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("Set2Set", () -> new Set2Set(FEATURES, 2, 1),
                    x, index, NUM_NODES, out -> {
                        // Set2Set returns 2 * inChannels
                        if (out.size(0) != NUM_NODES || out.size(1) != 2 * FEATURES) {
                            throw new AssertionError("shape=" + Arrays.toString(out.shape()));
                        }
                    });
            bench("GraphMultisetTransformer",
                    () -> new GraphMultisetTransformer(FEATURES, FEATURES, 2, 4),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("SetTransformerAggregation",
                    () -> new SetTransformerAggregation(FEATURES, FEATURES, 2, 4),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("PatchTransformerAggregation",
                    () -> new PatchTransformerAggregation(FEATURES, 2, 1),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("LCMAggregation", () -> new LCMAggregation(FEATURES, 4),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));
            bench("EquilibriumAggregation", () -> new EquilibriumAggregation((int) FEATURES, 5, 1e-4),
                    x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));

            // DeepSets with explicit MLPs
            bench("DeepSetsAggregation", () -> {
                SequentialImpl local = new SequentialImpl();
                local.push_back(new LinearImpl(FEATURES, FEATURES));
                local.push_back(new ReLUImpl());
                SequentialImpl global = new SequentialImpl();
                global.push_back(new LinearImpl(FEATURES, FEATURES));
                return new DeepSetsAggregation(local, global);
            }, x, index, NUM_NODES, out -> expectShape(out, NUM_NODES, FEATURES));

            // DegreeScalerAggregation wraps a base aggregator and expands channels
            bench("DegreeScalerAggregation", () -> {
                List<String> scalers = Arrays.asList("identity", "amplification", "attenuation");
                return new DegreeScalerAggregation(4.0, scalers, new MeanAggregation());
            }, x, index, NUM_NODES, out -> {
                if (out.size(0) != NUM_NODES) throw new AssertionError("rows");
                if (out.size(1) != 3 * FEATURES) {
                    throw new AssertionError("cols expected " + (3 * FEATURES) + " got " + out.size(1));
                }
            });

            // MultiAggregation concatenates multiple aggregators
            bench("MultiAggregation", () -> new MultiAggregation(
                    new SumAggregation(), new MeanAggregation(), new MaxAggregation()
            ), x, index, NUM_NODES, out -> {
                if (out.size(0) != NUM_NODES) throw new AssertionError("rows");
                if (out.size(1) != 3 * FEATURES) {
                    throw new AssertionError("cols expected " + (3 * FEATURES) + " got " + out.size(1));
                }
            });
        }

        System.out.println();
        System.out.println("----- Results -----");
        System.out.printf(Locale.ROOT, "%-36s %10s %12s %s%n", "name", "status", "ms/iter", "notes");
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

    @FunctionalInterface
    interface Checker {
        void check(Tensor out);
    }

    private static void expectShape(Tensor out, long rows, long cols) {
        if (out.size(0) != rows || out.size(1) != cols) {
            throw new AssertionError("shape expected [" + rows + ", " + cols + "] got "
                    + Arrays.toString(out.shape()));
        }
        if (out.isnan().any().item().toBool()) {
            throw new AssertionError("output contains NaN");
        }
        if (out.isinf().any().item().toBool()) {
            throw new AssertionError("output contains Inf");
        }
    }

    private static void bench(String name, Supplier<Aggregation> factory,
                              Tensor x, Tensor index, long dimSize, Checker checker) {
        run(name, () -> {
            Aggregation aggr = factory.get();
            // warmup
            for (int i = 0; i < WARMUP; i++) {
                Tensor o = aggr.forward(x, index, dimSize);
                checker.check(o);
            }
            long t0 = System.nanoTime();
            Tensor last = null;
            for (int i = 0; i < ITERS; i++) {
                last = aggr.forward(x, index, dimSize);
            }
            long t1 = System.nanoTime();
            checker.check(last);
            double ms = (t1 - t0) / 1e6 / ITERS;
            rows.add(String.format(Locale.ROOT, "%-36s %10s %10.3f  shape=%s",
                    name, "PASS", ms, Arrays.toString(last.shape())));
        });
    }

    private static void run(String name, Runnable body) {
        try {
            body.run();
            // if body already added a row (bench), don't double-count notes
            if (rows.stream().noneMatch(r -> r.startsWith(name + " "))) {
                rows.add(String.format(Locale.ROOT, "%-36s %10s %10s  ok", name, "PASS", "-"));
            }
            passed++;
            System.out.println("PASS  " + name);
        } catch (Throwable t) {
            failed++;
            String msg = t.getClass().getSimpleName() + ": " + t.getMessage();
            failures.add(name + " -> " + msg);
            rows.add(String.format(Locale.ROOT, "%-36s %10s %10s  %s",
                    name, "FAIL", "-", abbreviate(msg, 80)));
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
