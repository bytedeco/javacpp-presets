package samples;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.utils.swanlab.SwanLabClient;
import org.bytedeco.pytorch.utils.swanlab.SwanLabLocalServer;
import org.bytedeco.pytorch.utils.swanlab.SwanLabTrainingMonitor;

import java.awt.Desktop;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Multi-dimensional SwanLab benchmark against the embedded {@link SwanLabLocalServer}.
 *
 * <p>Covers: init/finish, scalar logs, heatmap, histogram, scatter, bar, line
 * charts, images, text, audio, tables, summary, training monitor. Starts a real
 * local HTTP server and opens the UI URL for visual verification.
 *
 * <pre>
 *   java samples.BenchmarkSwanLab
 * </pre>
 */
public class BenchmarkSwanLab {

    static int passed = 0;
    static int failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final Random RND = new Random(11);

    interface Checked { void run() throws Exception; }

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

    static TensorOptions floatOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        System.out.println("=== SwanLab multi-dimensional benchmark (local offline server) ===\n");

        SwanLabLocalServer server = SwanLabLocalServer.start(0);
        System.out.println("local server: " + server.uiUrl());
        System.out.println("api base:     " + server.apiBase());

        try (SwanLabClient swan = SwanLabClient.newBuilder()
                .offline(server)
                .workspace("javacpp")
                .project("pytorch-bench")
                .experiment("swan-bench-" + (System.currentTimeMillis() % 100000))
                .apiKey("local")
                .build()) {

            check("init", () -> {
                Map<String, Object> cfg = new LinkedHashMap<>();
                cfg.put("lr", 1e-3);
                cfg.put("batch_size", 64);
                cfg.put("optimizer", "adamw");
                cfg.put("framework", "javacpp-pytorch");
                swan.init(cfg);
                if (swan.runId() == null || swan.runId().isBlank())
                    throw new AssertionError("empty experiment id");
                System.out.println("    exp id: " + swan.runId());
                System.out.println("    ui:     " + swan.uiUrl());
            });

            check("log scalars (50 steps)", () -> {
                double loss = 2.5, acc = 0.1, valLoss = 2.3;
                for (int s = 1; s <= 50; s++) {
                    loss *= 0.93 + 0.05 * RND.nextDouble();
                    valLoss *= 0.94 + 0.04 * RND.nextDouble();
                    acc = Math.min(0.99, acc + 0.018 * RND.nextDouble());
                    Map<String, Number> m = new LinkedHashMap<>();
                    m.put("train/loss", loss);
                    m.put("train/acc", acc);
                    m.put("val/loss", valLoss);
                    m.put("val/acc", Math.min(0.99, acc - 0.05 + 0.02 * RND.nextDouble()));
                    m.put("lr", 1e-3 * Math.pow(0.97, s / 5.0));
                    swan.log(m, s);
                }
            });

            check("logHeatmap attention-like", () -> {
                int n = 16;
                double[][] attn = new double[n][n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        double d = (i - j) / 3.0;
                        attn[i][j] = Math.exp(-d * d) + 0.05 * RND.nextDouble();
                    }
                // row-normalize
                for (int i = 0; i < n; i++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) sum += attn[i][j];
                    for (int j = 0; j < n; j++) attn[i][j] /= sum;
                }
                swan.logHeatmap("attention", attn, 50,
                        Map.of("title", "Self-attention", "colormap", "Viridis"));
            });

            check("logHeatmap confusion + Tensor", () -> {
                double[][] cm = {
                        {55, 2, 1, 0},
                        {3, 48, 4, 1},
                        {0, 4, 50, 2},
                        {1, 0, 3, 52}
                };
                swan.logHeatmap("confusion", cm, 50);
                try (PointerScope scope = new PointerScope()) {
                    Tensor t = randn(new long[]{10, 14}, floatOpts());
                    swan.logHeatmap("emb_hm", t, 50);
                }
            });

            check("logHistogram", () -> {
                double[] grads = new double[400];
                for (int i = 0; i < grads.length; i++) grads[i] = RND.nextGaussian() * 0.05;
                swan.logHistogram("grad_fc2", grads, 40, 50);
            });

            check("logScatter", () -> {
                double[][] pts = new double[100][2];
                for (int i = 0; i < 100; i++) {
                    double a = RND.nextGaussian(), b = RND.nextGaussian();
                    pts[i][0] = a;
                    pts[i][1] = 0.3 * a + 0.7 * b;
                }
                swan.logScatter("tsne_like", pts, 50);
            });

            check("logBar", () -> {
                swan.logBar("class_freq",
                        new double[]{120, 95, 80, 60, 40},
                        new String[]{"cat", "dog", "bird", "fish", "frog"},
                        50);
            });

            check("logLineChart multi-series", () -> {
                List<List<Double>> series = new ArrayList<>();
                List<Double> train = new ArrayList<>(), val = new ArrayList<>();
                for (int i = 0; i < 25; i++) {
                    train.add(2.0 * Math.exp(-i / 8.0));
                    val.add(2.1 * Math.exp(-i / 9.0) + 0.05);
                }
                series.add(train); series.add(val);
                swan.logLineChart("loss_curves", series, new String[]{"train", "val"}, 50);
            });

            check("logImage RGB Tensor", () -> {
                try (PointerScope scope = new PointerScope()) {
                    int C = 3, H = 40, W = 56;
                    Tensor img = zeros(new long[]{C, H, W}, floatOpts());
                    org.bytedeco.javacpp.FloatPointer p = img.data_ptr_float();
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++) {
                            p.put(0 * H * W + h * W + w, (float) (0.5 + 0.5 * Math.sin(w / 6.0)));
                            p.put(1 * H * W + h * W + w, (float) (0.5 + 0.5 * Math.cos(h / 5.0)));
                            p.put(2 * H * W + h * W + w, h / (float) H);
                        }
                    swan.logImage("sample", img, 50);
                }
            });

            check("logImage grayscale", () -> {
                try (PointerScope scope = new PointerScope()) {
                    int H = 28, W = 28;
                    Tensor img = zeros(new long[]{1, H, W}, floatOpts());
                    org.bytedeco.javacpp.FloatPointer p = img.data_ptr_float();
                    // fake digit-like blob
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++) {
                            double dx = h - 14, dy = w - 14;
                            p.put(h * W + w, (float) Math.exp(-(dx * dx + dy * dy) / 40.0));
                        }
                    swan.logImage("digit_blob", img, 50);
                }
            });

            check("logText", () -> {
                swan.logText("readme",
                        "SwanLab benchmark for org.bytedeco.pytorch.utils.swanlab\n"
                                + "multimodal: scalars, heatmap, image, audio, table, bar, scatter",
                        50);
            });

            check("logAudio", () -> {
                int sr = 8000;
                float[] mono = new float[sr];
                for (int i = 0; i < mono.length; i++) {
                    mono[i] = (float) (0.35 * Math.sin(2 * Math.PI * 523.25 * i / (double) sr) // C5
                            * (1.0 - i / (double) mono.length)); // decay
                }
                swan.logAudio("tone_c5", mono, sr, 50);
            });

            check("logTable", () -> {
                List<String[]> rows = new ArrayList<>();
                rows.add(new String[]{"resnet18", "11.7M", "91.2", "1.2ms"});
                rows.add(new String[]{"mobilenet", "3.5M", "88.4", "0.4ms"});
                rows.add(new String[]{"vit-tiny", "5.7M", "90.1", "0.9ms"});
                swan.logTable("model_zoo",
                        new String[]{"model", "params", "acc%", "latency"}, rows);
            });

            check("logSummary", () -> {
                Map<String, Object> sum = new LinkedHashMap<>();
                sum.put("best_val_acc", 0.93);
                sum.put("best_train_loss", 0.18);
                sum.put("epochs", 50);
                sum.put("device", "cpu");
                swan.logSummary(sum);
            });

            check("SwanLabTrainingMonitor e2e", () -> {
                try (SwanLabClient s2 = SwanLabClient.newBuilder()
                        .offline(server).workspace("javacpp").project("pytorch-bench")
                        .experiment("monitor-" + (System.currentTimeMillis() % 10000))
                        .apiKey("local").build();
                     SwanLabTrainingMonitor mon = new SwanLabTrainingMonitor(
                             s2, Map.of("arch", "mlp", "hidden", 256), false)) {
                    for (int s = 0; s < 12; s++) {
                        mon.log(Map.of(
                                "loss", 1.8 * Math.exp(-s / 4.0),
                                "acc", 1 - Math.exp(-s / 5.0)));
                    }
                    double[][] hm = new double[4][4];
                    for (int i = 0; i < 4; i++) {
                        hm[i][i] = 20;
                        for (int j = 0; j < 4; j++) if (i != j) hm[i][j] = RND.nextInt(4);
                    }
                    mon.logHeatmap("cm", hm);
                    mon.logText("status", "monitor e2e ok");
                    System.out.println("    monitor ui: " + mon.uiUrl());
                }
            });

            check("HTTP export + dashboard", () -> {
                HttpClient http = HttpClient.newHttpClient();
                String exportUrl = server.uiUrl() + "/api/v1/experiments/" + swan.runId() + "/export";
                HttpResponse<String> exp = http.send(
                        HttpRequest.newBuilder(URI.create(exportUrl)).GET()
                                .timeout(Duration.ofSeconds(5)).build(),
                        HttpResponse.BodyHandlers.ofString());
                if (exp.statusCode() != 200)
                    throw new AssertionError("export HTTP " + exp.statusCode());
                if (!exp.body().contains("logs") && !exp.body().contains("n_logs"))
                    throw new AssertionError("export missing logs");

                HttpResponse<String> index = http.send(
                        HttpRequest.newBuilder(URI.create(server.uiUrl() + "/")).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
                if (index.statusCode() != 200 || !index.body().contains("SwanLab"))
                    throw new AssertionError("index bad");

                HttpResponse<String> page = http.send(
                        HttpRequest.newBuilder(URI.create(swan.uiUrl())).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
                if (page.statusCode() != 200)
                    throw new AssertionError("exp page HTTP " + page.statusCode());
                if (!page.body().contains("plotly") && !page.body().contains("Plotly"))
                    throw new AssertionError("exp page missing plotly");
                System.out.println("    export bytes: " + exp.body().length()
                        + "  page bytes: " + page.body().length());
            });

            swan.finish();

            String ui = swan.uiUrl();
            System.out.println("\n=== Open in browser ===");
            System.out.println(server.uiUrl());
            System.out.println(ui);
            try {
                if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                    Desktop.getDesktop().browse(URI.create(ui));
                } else {
                    new ProcessBuilder("open", ui).start();
                }
                System.out.println("(browser launch requested)");
            } catch (Exception e) {
                System.out.println("(could not auto-open: " + e.getMessage() + ")");
            }
        }

        System.out.println("\n=== Results ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        System.out.println(report);

        if (failed > 0) {
            server.close();
            System.exit(1);
        }

        int keepSec = 120;
        try {
            String ks = System.getenv("BENCH_KEEPALIVE_SEC");
            if (ks != null && !ks.isBlank()) keepSec = Integer.parseInt(ks.trim());
        } catch (Exception ignored) { }
        if (keepSec > 0) {
            System.out.println("Server kept alive " + keepSec + "s for manual verification at " + server.uiUrl());
            System.out.println("(set BENCH_KEEPALIVE_SEC=0 to skip wait)");
            try { Thread.sleep(keepSec * 1000L); } catch (InterruptedException ignored) {}
        }
        server.close();
    }
}
