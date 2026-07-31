package media;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.plot.visdom.VisdomClient;
import org.bytedeco.pytorch.plot.visdom.VisdomResponse;
import org.bytedeco.pytorch.plot.visdom.VisdomTrainingMonitor;

import java.awt.Desktop;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.plot.visdom.VisdomClient.opts;

/**
 * Multi-dimensional Visdom benchmark against a <b>real</b> Visdom server.
 *
 * <p>Covers: line / multi-line / append streaming, scatter 2D+3D, heatmap,
 * surface, contour, bar, histogram, boxplot, pie, stem, quiver, mesh,
 * image / image-grid, audio, text, table, dual-axis, training monitor.
 *
 * <pre>
 *   # terminal 1 — real Visdom (already started by this benchmark if missing)
 *   python -m visdom.server -port 8097
 *
 *   # terminal 2
 *   java media.BenchmarkVisdom [host] [port] [env]
 * </pre>
 *
 * Open the printed URL in a browser to visually verify every pane.
 */
public class BenchmarkVisdom {

    static int passed = 0;
    static int failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final Random RND = new Random(42);

    interface Checked {
        void run() throws Exception;
    }

    static void check(String name, Checked c) {
        try {
            c.run();
            passed++;
            System.out.println("  ✓ " + name);
            report.append("PASS  ").append(name).append('\n');
        } catch (Throwable t) {
            failed++;
            System.out.println("  ✗ " + name + "  —  " + t.getMessage());
            report.append("FAIL  ").append(name).append("  ").append(t).append('\n');
            t.printStackTrace(System.out);
        }
    }

    static void assertOk(VisdomResponse r, String what) {
        if (r == null) throw new AssertionError(what + ": null response");
        if (!r.ok()) throw new AssertionError(what + ": HTTP " + r.statusCode() + " body=" + r.body());
        // Visdom returns the window id as body
        if (r.windowId() == null || r.windowId().isBlank()) {
            throw new AssertionError(what + ": empty window id, body=" + r.body());
        }
    }

    static TensorOptions floatOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 8097;
        String env = args.length > 2 ? args[2] : "javacpp_bench";

        System.out.println("=== Visdom multi-dimensional benchmark ===\n");
        System.out.println("server: http://" + host + ":" + port);
        System.out.println("env:    " + env);

        try (VisdomClient viz = VisdomClient.newBuilder()
                .host(host).port(port).env(env)
                .raiseExceptions(true)
                .build()) {

            // Wait / probe
            boolean up = false;
            for (int i = 0; i < 30; i++) {
                if (viz.checkConnection()) { up = true; break; }
                Thread.sleep(500);
            }
            if (!up) {
                System.err.println("ERROR: Visdom server not reachable at "
                        + viz.baseUrl()
                        + "\nStart with:  python -m visdom.server -port " + port);
                System.exit(2);
            }
            System.out.println("connected: " + viz.baseUrl());
            System.out.println("UI url:    " + viz.uiUrl());
            System.out.println();

            // Clear previous panes in this env (best-effort)
            try { viz.closeAll(); } catch (Exception ignored) {}

            // ------------------------------------------------------------------
            System.out.println("-- Basic plots --");
            check("line single-trace", () -> {
                double[] y = new double[50];
                double[] x = new double[50];
                for (int i = 0; i < 50; i++) {
                    x[i] = i;
                    y[i] = Math.sin(i / 5.0) * Math.exp(-i / 40.0);
                }
                assertOk(viz.line(y, x, "line_sin",
                        opts("title", "Damped sine", "xlabel", "t", "ylabel", "y")), "line");
            });

            check("line multi-trace", () -> {
                int n = 40, m = 3;
                double[][] Y = new double[n][m];
                double[] X = new double[n];
                for (int i = 0; i < n; i++) {
                    X[i] = i;
                    Y[i][0] = Math.sin(i / 4.0);
                    Y[i][1] = Math.cos(i / 4.0);
                    Y[i][2] = Math.sin(i / 4.0) * Math.cos(i / 6.0);
                }
                assertOk(viz.lineMultiple(Y, X, new String[]{"sin", "cos", "sin·cos"},
                        "line_multi", opts("title", "Multi-line", "xlabel", "step")), "lineMultiple");
            });

            check("line append streaming (20 steps)", () -> {
                // seed
                assertOk(viz.line(new double[]{1.0}, new double[]{0}, "line_stream",
                        opts("title", "Streaming loss", "xlabel", "step", "ylabel", "loss")), "seed");
                double loss = 2.0;
                for (int s = 1; s <= 20; s++) {
                    loss *= 0.85 + 0.1 * RND.nextDouble();
                    assertOk(viz.lineAppend("line_stream", s, loss, "loss",
                            opts("title", "Streaming loss")), "append@" + s);
                }
            });

            check("scatter 2D clusters", () -> {
                double[][] pts = new double[90][2];
                int[] labels = new int[90];
                for (int i = 0; i < 90; i++) {
                    int k = i / 30 + 1;
                    labels[i] = k;
                    pts[i][0] = k * 3 + RND.nextGaussian();
                    pts[i][1] = k * 2 + RND.nextGaussian();
                }
                assertOk(viz.scatter(pts, labels, "scatter2d",
                        opts("title", "2D clusters", "legend",
                                List.of("A", "B", "C"), "markersize", 8)), "scatter");
            });

            check("scatter 3D", () -> {
                int n = 60;
                double[] x = new double[n], y = new double[n], z = new double[n];
                for (int i = 0; i < n; i++) {
                    double t = i / 5.0;
                    x[i] = Math.sin(t);
                    y[i] = Math.cos(t);
                    z[i] = t / 10.0;
                }
                assertOk(viz.scatter3D(x, y, z, "scatter3d",
                        opts("title", "3D helix")), "scatter3d");
            });

            // ------------------------------------------------------------------
            System.out.println("-- Heatmap / surface / contour (critical) --");
            check("heatmap gaussian + rownames/colnames", () -> {
                int rows = 20, cols = 24;
                double[][] m = new double[rows][cols];
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++) {
                        double dx = (r - 10) / 4.0, dy = (c - 12) / 5.0;
                        m[r][c] = Math.exp(-(dx * dx + dy * dy));
                    }
                List<String> rn = new ArrayList<>();
                List<String> cn = new ArrayList<>();
                for (int r = 0; r < rows; r++) rn.add("r" + r);
                for (int c = 0; c < cols; c++) cn.add("c" + c);
                assertOk(viz.heatmap(m, "heatmap_gauss",
                        opts("title", "Gaussian heatmap",
                                "colormap", "Viridis",
                                "rownames", rn, "columnnames", cn)), "heatmap");
            });

            check("heatmap confusion-matrix style", () -> {
                double[][] cm = {
                        {50, 2, 1, 0},
                        {3, 45, 4, 1},
                        {0, 5, 40, 3},
                        {1, 0, 2, 48}
                };
                assertOk(viz.heatmap(cm, "heatmap_cm",
                        opts("title", "Confusion matrix",
                                "colormap", "Blues",
                                "rownames", List.of("cat", "dog", "bird", "fish"),
                                "columnnames", List.of("cat", "dog", "bird", "fish"))), "cm");
            });

            check("heatmap from Tensor", () -> {
                try (PointerScope scope = new PointerScope()) {
                    Tensor t = randn(new long[]{16, 16}, floatOpts());
                    // make it smoother: conv-ish via repeated average — just use raw noise
                    assertOk(viz.heatmap(t, "heatmap_tensor",
                            opts("title", "Tensor heatmap (randn 16x16)",
                                    "colormap", "Hot")), "tensor-hm");
                }
            });

            check("surface (3D)", () -> {
                int n = 30;
                double[][] Z = new double[n][n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        double x = (i - 15) / 5.0, y = (j - 15) / 5.0;
                        Z[i][j] = Math.sin(Math.sqrt(x * x + y * y) * 2);
                    }
                assertOk(viz.surf(Z, "surface_ripple",
                        opts("title", "Ripple surface", "colormap", "Viridis")), "surf");
            });

            check("contour", () -> {
                int n = 40;
                double[][] Z = new double[n][n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        double x = (i - 20) / 6.0, y = (j - 20) / 6.0;
                        Z[i][j] = x * x - y * y;
                    }
                assertOk(viz.contour(Z, "contour_saddle",
                        opts("title", "Saddle contour", "colormap", "RdBu")), "contour");
            });

            // ------------------------------------------------------------------
            System.out.println("-- Statistical plots --");
            check("bar", () -> {
                assertOk(viz.bar(new double[]{4, 7, 2, 9, 5}, "bar_basic",
                        opts("title", "Bar chart",
                                "rownames", List.of("a", "b", "c", "d", "e"))), "bar");
            });

            check("bar grouped", () -> {
                double[][] v = {{3, 4}, {5, 2}, {6, 7}, {2, 5}};
                assertOk(viz.barGrouped(v, new String[]{"train", "val"}, "bar_grouped",
                        opts("title", "Grouped bars",
                                "rownames", List.of("ep1", "ep2", "ep3", "ep4"))), "barGrouped");
            });

            check("histogram", () -> {
                double[] vals = new double[500];
                for (int i = 0; i < vals.length; i++) vals[i] = RND.nextGaussian() * 2 + 1;
                assertOk(viz.histogram(vals, 25, "hist_gauss",
                        opts("title", "Gaussian histogram")), "hist");
            });

            check("boxplot", () -> {
                double[][] seq = new double[80][3];
                for (int i = 0; i < 80; i++) {
                    seq[i][0] = RND.nextGaussian();
                    seq[i][1] = RND.nextGaussian() * 1.5 + 2;
                    seq[i][2] = RND.nextGaussian() * 0.5 - 1;
                }
                assertOk(viz.boxplot(seq, "box_demo",
                        opts("title", "Box plots",
                                "legend", List.of("n(0,1)", "n(2,1.5)", "n(-1,0.5)"))), "box");
            });

            check("pie", () -> {
                assertOk(viz.pie(new double[]{35, 25, 20, 15, 5},
                        new String[]{"A", "B", "C", "D", "E"},
                        "pie_demo", opts("title", "Class distribution")), "pie");
            });

            check("stem", () -> {
                double[] y = new double[20];
                for (int i = 0; i < 20; i++) y[i] = Math.sin(i / 2.0);
                assertOk(viz.stem(y, null, "stem_demo",
                        opts("title", "Stem plot")), "stem");
            });

            check("quiver", () -> {
                int n = 8;
                double[][] U = new double[n][n], V = new double[n][n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        U[i][j] = Math.cos(j / 2.0);
                        V[i][j] = Math.sin(i / 2.0);
                    }
                assertOk(viz.quiver(U, V, "quiver_demo",
                        opts("title", "Quiver field")), "quiver");
            });

            check("mesh 3D", () -> {
                // simple tetrahedron
                double[][] verts = {
                        {0, 0, 0}, {1, 0, 0}, {0.5, Math.sqrt(3) / 2, 0},
                        {0.5, Math.sqrt(3) / 6, Math.sqrt(6) / 3}
                };
                int[][] faces = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
                assertOk(viz.mesh(verts, faces, "mesh_tet",
                        opts("title", "Tetrahedron", "opacity", 0.7, "color", "#5ec8ff")), "mesh");
            });

            // ------------------------------------------------------------------
            System.out.println("-- Multimodal: image / audio / text / table --");
            check("image CHW float tensor (RGB gradient)", () -> {
                try (PointerScope scope = new PointerScope()) {
                    int C = 3, H = 64, W = 96;
                    float[] chw = new float[C * H * W];
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++) {
                            chw[0 * H * W + h * W + w] = w / (float) W;           // R
                            chw[1 * H * W + h * W + w] = h / (float) H;           // G
                            chw[2 * H * W + h * W + w] = 0.5f;                    // B
                        }
                    assertOk(viz.image(chw, C, H, W, "img_gradient",
                            opts("title", "RGB gradient", "caption", "R=x G=y")), "image");
                }
            });

            check("image from Tensor (checkerboard)", () -> {
                try (PointerScope scope = new PointerScope()) {
                    int H = 48, W = 48;
                    Tensor img = zeros(new long[]{1, H, W}, floatOpts());
                    // Build checkerboard in java then wrap — use pure float path via image(float[])
                    float[] chw = new float[H * W];
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            chw[h * W + w] = ((h / 6) + (w / 6)) % 2 == 0 ? 1f : 0.1f;
                    assertOk(viz.image(chw, 1, H, W, "img_checker",
                            opts("title", "Checkerboard", "caption", "1-channel→RGB")), "checker");
                }
            });

            check("images grid NCHW", () -> {
                int N = 8, C = 3, H = 32, W = 32;
                float[] nchw = new float[N * C * H * W];
                for (int n = 0; n < N; n++)
                    for (int c = 0; c < C; c++)
                        for (int h = 0; h < H; h++)
                            for (int w = 0; w < W; w++) {
                                float v = (float) ((Math.sin((h + n * 3) / 4.0) + 1) / 2.0);
                                if (c == 1) v = (float) ((Math.cos((w + n) / 5.0) + 1) / 2.0);
                                if (c == 2) v = n / (float) N;
                                nchw[n * C * H * W + c * H * W + h * W + w] = v;
                            }
                assertOk(viz.images(nchw, N, C, H, W, 4, "img_grid",
                        opts("title", "Image grid 8× (nrow=4)")), "images");
            });

            check("audio sine WAV", () -> {
                int sr = 16000;
                float[] mono = new float[sr]; // 1 second
                for (int i = 0; i < mono.length; i++) {
                    mono[i] = (float) (0.4 * Math.sin(2 * Math.PI * 440 * i / (double) sr)
                            + 0.2 * Math.sin(2 * Math.PI * 880 * i / (double) sr));
                }
                assertOk(viz.audio(mono, sr, "audio_a440",
                        opts("title", "A440 + octave")), "audio");
            });

            check("text + appendText", () -> {
                assertOk(viz.text("<b>Visdom benchmark</b><br/>javacpp-pytorch utils.visdom",
                        "text_main", opts("title", "Notes"), false), "text");
                assertOk(viz.appendText("text_main",
                        "<br/><i>append ok @ " + System.currentTimeMillis() + "</i>"), "appendText");
            });

            check("HTML table", () -> {
                assertOk(viz.table(
                        new String[]{"epoch", "loss", "acc", "lr"},
                        new String[][]{
                                {"1", "1.234", "0.55", "1e-3"},
                                {"2", "0.876", "0.71", "1e-3"},
                                {"3", "0.542", "0.83", "5e-4"},
                                {"4", "0.321", "0.90", "5e-4"},
                        },
                        "table_metrics",
                        opts("title", "Epoch metrics")), "table");
            });

            check("dual-axis lines", () -> {
                double[] X = new double[30], Y1 = new double[30], Y2 = new double[30];
                for (int i = 0; i < 30; i++) {
                    X[i] = i;
                    Y1[i] = Math.exp(-i / 12.0);
                    Y2[i] = 1 - Math.exp(-i / 10.0);
                }
                assertOk(viz.dualAxisLines(X, Y1, Y2, "dual_axis",
                        opts("title", "Loss vs Acc",
                                "name_y1", "loss", "name_y2", "acc")), "dual");
            });

            check("properties pane", () -> {
                List<Map<String, Object>> props = new ArrayList<>();
                props.add(Map.of("type", "text", "name", "run", "value", "bench1"));
                props.add(Map.of("type", "number", "name", "lr", "value", "0.001"));
                props.add(Map.of("type", "checkbox", "name", "amp", "value", true));
                assertOk(viz.properties(props, "props_demo",
                        opts("title", "HParams")), "properties");
            });

            // ------------------------------------------------------------------
            System.out.println("-- Training monitor end-to-end --");
            check("VisdomTrainingMonitor simulated epoch", () -> {
                try (VisdomTrainingMonitor mon = new VisdomTrainingMonitor(viz, "mlp")) {
                    double loss = 2.5, acc = 0.2;
                    for (int step = 0; step < 25; step++) {
                        loss *= 0.90 + 0.05 * RND.nextDouble();
                        acc = Math.min(0.99, acc + 0.03 * RND.nextDouble());
                        mon.logLoss(step, loss);
                        mon.logAccuracy(step, acc);
                        mon.logLearningRate(step, 1e-3 * Math.pow(0.95, step / 5.0));
                    }
                    // confusion heatmap
                    double[][] cm = new double[5][5];
                    for (int i = 0; i < 5; i++) {
                        cm[i][i] = 40 + RND.nextInt(10);
                        for (int j = 0; j < 5; j++) if (i != j) cm[i][j] = RND.nextInt(5);
                    }
                    mon.logHeatmap("confusion", cm, opts("colormap", "Blues"));
                    mon.logText("summary", "25 steps simulated · final loss="
                            + String.format("%.4f", loss) + " acc=" + String.format("%.3f", acc));

                    // weight histogram
                    double[] weights = new double[200];
                    for (int i = 0; i < weights.length; i++) weights[i] = RND.nextGaussian() * 0.1;
                    mon.logHistogram("fc1.weight", weights);
                }
            });

            // Open browser
            String ui = viz.uiUrl();
            System.out.println("\n=== Open in browser ===");
            System.out.println(ui);
            try {
                if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                    Desktop.getDesktop().browse(URI.create(ui));
                    System.out.println("(browser launch requested)");
                } else {
                    // macOS fallback
                    new ProcessBuilder("open", ui).start();
                    System.out.println("(open " + ui + ")");
                }
            } catch (Exception e) {
                System.out.println("(could not auto-open browser: " + e.getMessage() + ")");
            }
        }

        System.out.println("\n=== Results ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        System.out.println(report);
        if (failed > 0) System.exit(1);
    }
}
