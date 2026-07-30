package samples.demo.transform;
import org.bytedeco.pytorch.data.transforms.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.*;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

import static org.bytedeco.pytorch.global.torch.kCPU;

/**
 * Smoke + micro-benchmark for {@code org.bytedeco.pytorch.geometric.transforms}.
 * Uses only public constructors / {@link BaseTransform#apply(GraphData)}.
 */
public class TransformBenchmark {

    private static final int WARMUP = 1;
    private static final int ITERS = 3;
    private static final long N = 24;
    private static final long F = 8;
    private static final long E = 48;

    private static int passed = 0;
    private static int failed = 0;
    private static final List<String> failures = new ArrayList<>();
    private static final List<String> rows = new ArrayList<>();

    public static void main(String[] args) {
        torch.manual_seed(11);
        System.out.println("=== Transform Benchmark ===");
        System.out.printf(Locale.ROOT, "N=%d E=%d F=%d warmup=%d iters=%d%n%n",
                N, E, F, WARMUP, ITERS);

        try (PointerScope scope = new PointerScope()) {
            // ---- feature / structure transforms on generic graph ----
            bench("NormalizeFeatures", NormalizeFeatures::new, TransformBenchmark::baseGraph, d -> {
                requireX(d);
                checkFinite(d.x);
            });
            bench("SVDFeatureReduction", () -> new SVDFeatureReduction(4),
                    TransformBenchmark::baseGraph, d -> {
                        requireX(d);
                        if (d.x.size(1) != 4) {
                            throw new AssertionError("SVD out dim expected 4 got " + d.x.size(1));
                        }
                    });
            bench("Constant", () -> new Constant(1.0), TransformBenchmark::baseGraph, d -> {
                requireX(d);
                if (d.x.size(1) != F + 1) {
                    throw new AssertionError("Constant should append 1 col, got " + d.x.size(1));
                }
            });
            bench("OneHotDegree", () -> new OneHotDegree(8), TransformBenchmark::baseGraph, d -> {
                requireX(d);
                // appends degree one-hot → F + (maxDegree+1)
                if (d.x.size(1) <= F) {
                    throw new AssertionError("OneHotDegree should expand features, got " + d.x.size(1));
                }
            });
            bench("LocalDegreeProfile", LocalDegreeProfile::new, TransformBenchmark::baseGraph, d -> {
                requireX(d);
                if (d.x.size(1) <= F) {
                    throw new AssertionError("LDP should expand features");
                }
            });
            bench("AddSelfLoops", AddSelfLoops::new, TransformBenchmark::baseGraph, d -> {
                requireEdge(d);
                if (d.edge_index.size(1) < E) {
                    throw new AssertionError("AddSelfLoops should not shrink edges");
                }
            });
            bench("ToUndirected", ToUndirected::new, TransformBenchmark::baseGraph, d -> {
                requireEdge(d);
            });
            bench("ToSparseTensor", ToSparseTensor::new, TransformBenchmark::baseGraph, d -> {
                // may store adj_t in attributes
                if (d.edge_index == null && d.get("adj_t") == null && d.adj == null) {
                    throw new AssertionError("ToSparseTensor left no adjacency");
                }
            });
            bench("RandomNodeSplit", () -> new RandomNodeSplit(0.6, 0.2, 0.2),
                    TransformBenchmark::baseGraph, d -> {
                        if (d.get("train_mask") == null || !d.get("train_mask").defined()) {
                            throw new AssertionError("missing train_mask");
                        }
                    });
            bench("RandomLinkSplit", () -> new RandomLinkSplit(0.1, 0.1),
                    () -> {
                        // need more edges for split
                        GraphData g = baseGraph();
                        g.edge_index = randomEdges(N, 80);
                        return g;
                    }, d -> requireEdge(d));
            bench("Pad", () -> new Pad(N + 4), TransformBenchmark::baseGraph, d -> {
                requireX(d);
                if (d.x.size(0) < N + 4) {
                    throw new AssertionError("Pad target rows " + (N + 4) + " got " + d.x.size(0));
                }
            });
            bench("IndexToMask", () -> new IndexToMask(N), () -> {
                GraphData g = baseGraph();
                g.put("train_indices", torch.tensor(new long[]{0, 1, 2, 5}));
                g.put("val_indices", torch.tensor(new long[]{3, 4}));
                g.put("test_indices", torch.tensor(new long[]{6, 7, 8}));
                return g;
            }, d -> {
                if (d.get("train_mask") == null || !d.get("train_mask").defined()) {
                    throw new AssertionError("IndexToMask missing train_mask");
                }
            });
            bench("MaskToIndex", MaskToIndex::new, () -> {
                GraphData g = baseGraph();
                Tensor ar = torch.arange(new org.bytedeco.pytorch.Scalar(N));
                g.put("train_mask", ar.lt(new org.bytedeco.pytorch.Scalar(5)));
                g.put("val_mask", ar.ge(new org.bytedeco.pytorch.Scalar(5))
                        .logical_and(ar.lt(new org.bytedeco.pytorch.Scalar(10))));
                g.put("test_mask", ar.ge(new org.bytedeco.pytorch.Scalar(10)));
                return g;
            }, d -> {
                if (d.get("train_indices") == null) {
                    throw new AssertionError("MaskToIndex missing train_indices");
                }
            });
            bench("RemoveTrainingClasses", () -> new RemoveTrainingClasses(Arrays.asList(0, 1)),
                    () -> {
                        GraphData g = baseGraph();
                        g.y = torch.tensor(new long[]{
                                0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                                1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
                                2, 0, 1, 2
                        });
                        g.put("train_mask", torch.ones(new long[]{N},
                                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Bool))));
                        return g;
                    }, d -> {
                        if (d.get("train_mask") == null) {
                            throw new AssertionError("train_mask missing");
                        }
                    });
            bench("NodePropertySplit", () -> new NodePropertySplit(0.5, 0.25, true),
                    () -> {
                        GraphData g = baseGraph();
                        g.put("node_prop", torch.randn(N));
                        return g;
                    }, d -> {
                        if (d.get("train_mask") == null) {
                            throw new AssertionError("NodePropertySplit missing train_mask");
                        }
                    });
            bench("Compose", () -> new Compose(new NormalizeFeatures(), new Constant(0.5)),
                    TransformBenchmark::baseGraph, d -> {
                        requireX(d);
                        if (d.x.size(1) != F + 1) {
                            throw new AssertionError("Compose dim " + d.x.size(1));
                        }
                    });
            bench("ToDevice", () -> new ToDevice(new Device(kCPU())),
                    TransformBenchmark::baseGraph, d -> requireX(d));

            // nested structural
            bench("GCNNorm", AdvancedStructuralTransforms.GCNNorm::new,
                    TransformBenchmark::baseGraph, d -> requireEdge(d));
            bench("TwoHop", AdvancedStructuralTransforms.TwoHop::new,
                    TransformBenchmark::baseGraph, d -> requireEdge(d));
            bench("ToDense", AdvancedStructuralTransforms.ToDense::new,
                    TransformBenchmark::baseGraph, d -> {
                        if (d.adj == null && d.get("adj") == null) {
                            // may set adj field
                            try {
                                d.initDenseAdj();
                            } catch (Throwable ignore) {
                            }
                        }
                    });
            bench("SIGN", () -> new AdvancedStructuralTransforms.SIGN(2),
                    TransformBenchmark::baseGraph, d -> requireX(d));
            bench("VirtualNode", HighOrderTransforms.VirtualNode::new,
                    TransformBenchmark::baseGraph, d -> {
                        requireX(d);
                        if (d.x.size(0) != N + 1) {
                            throw new AssertionError("VirtualNode nodes expected " + (N + 1)
                                    + " got " + d.x.size(0));
                        }
                    });
            bench("RemoveSelfLoops", TopologyTransforms.RemoveSelfLoops::new, () -> {
                GraphData g = baseGraph();
                // force self loops
                g.edge_index = torch.cat(new org.bytedeco.pytorch.TensorVector(
                        g.edge_index,
                        torch.tensor(new long[]{0, 1, 0, 1}).reshape(2, 2)
                ), 1);
                return g;
            }, d -> requireEdge(d));
            bench("AddRemainingSelfLoops", TopologyTransforms.AddRemainingSelfLoops::new,
                    TransformBenchmark::baseGraph, d -> requireEdge(d));
            bench("KNNGraph", () -> new TopologyTransforms.KNNGraph(3), () -> {
                GraphData g = baseGraph();
                g.pos = torch.randn(N, 3);
                g.edge_index = null; // will be built
                return g;
            }, d -> requireEdge(d));
            bench("RadiusGraph", () -> new TopologyTransforms.RadiusGraph(1.5), () -> {
                GraphData g = baseGraph();
                g.pos = torch.randn(N, 3);
                g.edge_index = null;
                return g;
            }, d -> {
                // may produce empty edges if radius too small — still must not crash
            });
            bench("AddRandomWalkPE", () -> new SpectralAndStructuralTransforms.AddRandomWalkPE(4),
                    TransformBenchmark::baseGraph, d -> requireX(d));
            bench("AddLaplacianEigenvectorPE",
                    () -> new SpectralAndStructuralTransforms.AddLaplacianEigenvectorPE(3),
                    TransformBenchmark::baseGraph, d -> requireX(d));
            bench("LaplacianLambdaMax", SpectralAndStructuralTransforms.LaplacianLambdaMax::new,
                    TransformBenchmark::baseGraph, d -> {
                    });
            bench("FeaturePropagation", () -> new AdvancedStructureTransforms.FeaturePropagation(4),
                    () -> {
                        GraphData g = baseGraph();
                        // mask some features missing
                        Tensor mask = torch.ones(new long[]{N, F},
                                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Bool)));
                        // zero out last 4 nodes features via putting missing mask if required
                        g.put("feature_mask", mask);
                        return g;
                    }, d -> requireX(d));
            bench("HalfHop", AdvancedStructureTransforms.HalfHop::new,
                    TransformBenchmark::baseGraph, d -> requireX(d));

            // ---- point-cloud / geometric ----
            bench("Center", Center::new, TransformBenchmark::pointCloud, d -> {
                requirePos(d);
                // mean should be near zero after centering
                Tensor mean = d.pos.mean(new long[]{0}, false, new ScalarTypeOptional(torch.ScalarType.Float));
                float meanAbs = mean.abs().max().item().toFloat();
                if (meanAbs > 1e-3f) {
                    throw new AssertionError("Center mean abs max=" + meanAbs);
                }
            });
            bench("NormalizeScale", NormalizeScale::new, TransformBenchmark::pointCloud, d -> {
                requirePos(d);
            });
            bench("NormalizeRotation", NormalizeRotation::new, TransformBenchmark::pointCloud, d -> {
                requirePos(d);
            });
            bench("RandomJitter", () -> new RandomJitter(0.01f),
                    TransformBenchmark::pointCloud, d -> requirePos(d));
            bench("RandomFlip", () -> new RandomFlip(0, 1.0),
                    TransformBenchmark::pointCloud, d -> requirePos(d));
            bench("RandomScale", () -> new RandomScale(0.5f, 1.5f),
                    TransformBenchmark::pointCloud, d -> requirePos(d));
            bench("RandomRotate", () -> new RandomRotate(30f, 2),
                    TransformBenchmark::pointCloud, d -> requirePos(d));
            bench("RandomShear", () -> new RandomShear(0.1f),
                    TransformBenchmark::pointCloud, d -> requirePos(d));
            bench("LinearTransformation", () -> {
                Tensor M = torch.eye(3);
                return new LinearTransformation(M);
            }, TransformBenchmark::pointCloud, d -> requirePos(d));
            bench("FixedPoints", () -> new FixedPoints(10),
                    TransformBenchmark::pointCloud, d -> {
                        requirePos(d);
                        if (d.pos.size(0) != 10) {
                            throw new AssertionError("FixedPoints expected 10 got " + d.pos.size(0));
                        }
                    });
            bench("GridSampling", () -> new GridSampling(0.5f),
                    TransformBenchmark::pointCloud, d -> requirePos(d));
            bench("Distance", () -> new Distance(false), TransformBenchmark::pointCloud, d -> {
                if (d.edge_attr == null) throw new AssertionError("Distance should set edge_attr");
                checkFinite(d.edge_attr);
            });
            bench("Cartesian", Cartesian::new, TransformBenchmark::pointCloud, d -> {
                if (d.edge_attr == null || d.edge_attr.size(1) != 3) {
                    throw new AssertionError("Cartesian edge_attr dim");
                }
            });
            bench("LocalCartesian", LocalCartesian::new, TransformBenchmark::pointCloud, d -> {
                if (d.edge_attr == null || d.edge_attr.size(1) != 3) {
                    throw new AssertionError("LocalCartesian edge_attr dim");
                }
            });
            bench("Polar", Polar::new, () -> {
                GraphData g = pointCloud();
                g.pos = torch.randn(N, 2); // Polar needs 2D
                return g;
            }, d -> {
                if (d.edge_attr == null || d.edge_attr.size(1) != 2) {
                    throw new AssertionError("Polar edge_attr expected 2 got "
                            + (d.edge_attr == null ? null : d.edge_attr.size(1)));
                }
            });
            bench("Spherical", Spherical::new, TransformBenchmark::pointCloud, d -> {
                if (d.edge_attr == null || d.edge_attr.size(1) != 3) {
                    throw new AssertionError("Spherical edge_attr dim");
                }
            });
            bench("PointPairFeatures", PointPairFeatures::new, () -> {
                GraphData g = pointCloud();
                g.put("norm", torch.randn(N, 3));
                return g;
            }, d -> {
                if (d.edge_attr == null || d.edge_attr.size(1) < 4) {
                    throw new AssertionError("PPF edge_attr expected >=4 dims");
                }
            });
            bench("FaceToEdge", () -> new FaceToEdge(true), () -> {
                GraphData g = new GraphData(torch.randn(4, F), null);
                // one tetrahedron-ish face list [3, F]
                g.put("face", torch.tensor(new long[]{
                        0, 1, 2,
                        0, 1, 3,
                        0, 2, 3,
                        1, 2, 3
                }).reshape(3, 4));
                return g;
            }, d -> requireEdge(d));
            bench("Delaunay", Delaunay::new, () -> {
                GraphData g = new GraphData(torch.randn(8, F), null);
                g.pos = torch.randn(8, 2);
                return g;
            }, d -> {
                // may set face or edge_index
            });
            bench("GenerateMeshNormals", GenerateMeshNormals::new, () -> {
                GraphData g = new GraphData(torch.randn(4, F), null);
                g.pos = torch.tensor(new float[]{
                        0, 0, 0,
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1
                }).reshape(4, 3);
                g.put("face", torch.tensor(new long[]{
                        0, 1, 2,
                        0, 1, 3,
                        0, 2, 3,
                        1, 2, 3
                }).reshape(3, 4));
                return g;
            }, d -> {
                if (d.get("norm") == null || !d.get("norm").defined()) {
                    throw new AssertionError("GenerateMeshNormals should set data['norm']");
                }
                if (d.get("norm").size(0) != 4 || d.get("norm").size(1) != 3) {
                    throw new AssertionError("norm shape " + Arrays.toString(d.get("norm").shape()));
                }
            });
            bench("SamplePoints", () -> new SamplePoints(16), () -> {
                GraphData g = new GraphData(null, null);
                g.pos = torch.tensor(new float[]{
                        0, 0, 0,
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1
                }).reshape(4, 3);
                g.put("face", torch.tensor(new long[]{
                        0, 1, 2,
                        0, 1, 3,
                        0, 2, 3,
                        1, 2, 3
                }).reshape(3, 4));
                return g;
            }, d -> {
                requirePos(d);
                if (d.pos.size(0) != 16) {
                    throw new AssertionError("SamplePoints expected 16 got " + d.pos.size(0));
                }
            });
            bench("ToSLIC", () -> new ToSLIC(4, 10f), () -> {
                // image-like: often expects pos + x as pixels; keep minimal
                GraphData g = pointCloud();
                return g;
            }, d -> {
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

    private static GraphData baseGraph() {
        GraphData g = new GraphData(torch.randn(N, F), randomEdges(N, E));
        return g;
    }

    private static GraphData pointCloud() {
        GraphData g = new GraphData(torch.randn(N, F), randomEdges(N, E));
        g.pos = torch.randn(N, 3);
        return g;
    }

    private static Tensor randomEdges(long n, long e) {
        java.util.Random rnd = new java.util.Random(11);
        long[] flat = new long[(int) (2 * e)];
        for (int i = 0; i < e; i++) {
            flat[i] = rnd.nextInt((int) n);
            flat[(int) e + i] = rnd.nextInt((int) n);
        }
        return torch.tensor(flat).reshape(2, e);
    }

    private static void requireX(GraphData d) {
        if (d.x == null) throw new AssertionError("x is null");
        checkFinite(d.x);
    }

    private static void requirePos(GraphData d) {
        if (d.pos == null) throw new AssertionError("pos is null");
        checkFinite(d.pos);
    }

    private static void requireEdge(GraphData d) {
        if (d.edge_index == null) throw new AssertionError("edge_index is null");
    }

    private static void checkFinite(Tensor t) {
        if (t.isnan().any().item().toBool()) throw new AssertionError("NaN");
        if (t.isinf().any().item().toBool()) throw new AssertionError("Inf");
    }

    private static void bench(String name, Supplier<BaseTransform> factory,
                              Supplier<GraphData> dataFactory,
                              java.util.function.Consumer<GraphData> checker) {
        run(name, () -> {
            BaseTransform t = factory.get();
            GraphData last = null;
            for (int i = 0; i < WARMUP; i++) {
                last = t.apply(dataFactory.get());
                checker.accept(last);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                last = t.apply(dataFactory.get());
            }
            long t1 = System.nanoTime();
            checker.accept(last);
            double ms = (t1 - t0) / 1e6 / ITERS;
            String note = summarize(last);
            rows.add(String.format(Locale.ROOT, "%-32s %8s %8.3f  %s",
                    name, "PASS", ms, note));
        });
    }

    private static String summarize(GraphData d) {
        StringBuilder sb = new StringBuilder();
        if (d.x != null) sb.append("x=").append(Arrays.toString(d.x.shape())).append(' ');
        if (d.pos != null) sb.append("pos=").append(Arrays.toString(d.pos.shape())).append(' ');
        if (d.edge_index != null) sb.append("E=").append(d.edge_index.size(1)).append(' ');
        if (d.edge_attr != null) sb.append("ea=").append(Arrays.toString(d.edge_attr.shape()));
        return sb.toString().trim();
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
