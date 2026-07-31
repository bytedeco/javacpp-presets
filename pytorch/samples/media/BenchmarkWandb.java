package media;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.plot.wandb.WandbClient;
import org.bytedeco.pytorch.plot.wandb.WandbLocalServer;
import org.bytedeco.pytorch.plot.wandb.WandbTrainingMonitor;

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
 * Multi-dimensional WandB benchmark against the embedded {@link WandbLocalServer}.
 *
 * <p>Exercises: run lifecycle, scalar metrics, heatmap, histogram, scatter,
 * line charts, images (Tensor → PNG), text, audio, tables, summary, training
 * monitor. Starts a real local HTTP server and opens the UI URL.
 *
 * <pre>
 *   java media.BenchmarkWandb
 *   # open the printed http://127.0.0.1:&lt;port&gt;/runs/... URL
 * </pre>
 */
public class BenchmarkWandb {

    static int passed = 0;
    static int failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final Random RND = new Random(7);

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
        System.out.println("=== WandB multi-dimensional benchmark (local offline server) ===\n");

        // Keep server alive after main so user can browse (daemon=false via non-daemon keep?)
        // Local server uses daemon threads; we sleep at the end.
        WandbLocalServer server = WandbLocalServer.start(0);
        System.out.println("local server: " + server.uiUrl());
        System.out.println("api base:     " + server.apiBase());

        try (WandbClient wb = WandbClient.newBuilder()
                .offline(server)
                .entity("javacpp")
                .project("pytorch-bench")
                .apiKey("local")
                .build()) {

            check("initRun", () -> {
                Map<String, Object> cfg = new LinkedHashMap<>();
                cfg.put("lr", "1e-3");
                cfg.put("batch_size", "32");
                cfg.put("model", "mlp");
                cfg.put("framework", "javacpp-pytorch");
                wb.initRun("bench-" + System.currentTimeMillis(), cfg);
                if (wb.runId() == null || wb.runId().isBlank())
                    throw new AssertionError("empty run id");
                System.out.println("    run id: " + wb.runId());
                System.out.println("    ui:     " + wb.uiUrl());
            });

            check("log scalar metrics (40 steps)", () -> {
                double loss = 2.2, acc = 0.15, lr = 1e-3;
                for (int s = 1; s <= 40; s++) {
                    loss *= 0.92 + 0.06 * RND.nextDouble();
                    acc = Math.min(0.99, acc + 0.02 * RND.nextDouble());
                    if (s % 10 == 0) lr *= 0.5;
                    Map<String, Number> m = new LinkedHashMap<>();
                    m.put("loss", loss);
                    m.put("accuracy", acc);
                    m.put("lr", lr);
                    m.put("grad_norm", 0.5 + RND.nextDouble());
                    wb.log(m, s);
                }
            });

            check("logHeatmap confusion matrix", () -> {
                double[][] cm = {
                        {42, 3, 1, 0, 2},
                        {2, 38, 4, 1, 0},
                        {1, 5, 35, 3, 1},
                        {0, 1, 2, 40, 2},
                        {3, 0, 1, 2, 39}
                };
                wb.logHeatmap("confusion_matrix", cm, 40,
                        Map.of("title", "5-class confusion", "colormap", "Blues"));
            });

            check("logHeatmap from Tensor", () -> {
                try (PointerScope scope = new PointerScope()) {
                    Tensor t = randn(new long[]{12, 16}, floatOpts());
                    wb.logHeatmap("weight_hm", t, 40);
                }
            });

            check("logHistogram", () -> {
                double[] w = new double[300];
                for (int i = 0; i < w.length; i++) w[i] = RND.nextGaussian() * 0.15;
                wb.logHistogram("fc1.weight", w, 30, 40);
            });

            check("logScatter", () -> {
                double[][] pts = new double[80][2];
                for (int i = 0; i < 80; i++) {
                    pts[i][0] = RND.nextGaussian();
                    pts[i][1] = 0.6 * pts[i][0] + 0.4 * RND.nextGaussian();
                }
                wb.logScatter("latent_2d", pts, 40);
            });

            check("logChart LINE multi-series", () -> {
                List<List<Double>> series = new ArrayList<>();
                List<Double> s1 = new ArrayList<>(), s2 = new ArrayList<>();
                for (int i = 0; i < 20; i++) {
                    s1.add(Math.sin(i / 3.0));
                    s2.add(Math.cos(i / 3.0));
                }
                series.add(s1); series.add(s2);
                wb.logChart("trig", WandbClient.ChartType.LINE, series,
                        new String[]{"sin", "cos"}, 40);
            });

            check("logImage from Tensor (RGB)", () -> {
                try (PointerScope scope = new PointerScope()) {
                    int C = 3, H = 48, W = 64;
                    float[] chw = new float[C * H * W];
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++) {
                            chw[0 * H * W + h * W + w] = w / (float) W;
                            chw[1 * H * W + h * W + w] = h / (float) H;
                            chw[2 * H * W + h * W + w] = ((h + w) % 16) / 16f;
                        }
                    // wrap as tensor CHW
                    Tensor img = zeros(new long[]{C, H, W}, floatOpts());
                    org.bytedeco.javacpp.FloatPointer p = img.data_ptr_float();
                    for (int i = 0; i < chw.length; i++) p.put(i, chw[i]);
                    wb.logImage("sample_rgb", img, 40);
                }
            });

            check("logImage checkerboard grayscale", () -> {
                try (PointerScope scope = new PointerScope()) {
                    int H = 32, W = 32;
                    float[] hw = new float[H * W];
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            hw[h * W + w] = ((h / 4) + (w / 4)) % 2 == 0 ? 1f : 0.2f;
                    Tensor img = zeros(new long[]{1, H, W}, floatOpts());
                    org.bytedeco.javacpp.FloatPointer p = img.data_ptr_float();
                    for (int i = 0; i < hw.length; i++) p.put(i, hw[i]);
                    wb.logImage("checker", img, 40);
                }
            });

            check("logText", () -> {
                wb.logText("notes",
                        "WandB benchmark · javacpp-pytorch\nsteps=40\nmultimodal: metrics/heatmap/image/audio/table",
                        40);
            });

            check("logAudio", () -> {
                int sr = 8000;
                float[] mono = new float[sr / 2];
                for (int i = 0; i < mono.length; i++)
                    mono[i] = (float) (0.3 * Math.sin(2 * Math.PI * 440 * i / (double) sr));
                wb.logAudio("tone_a440", mono, sr, 40);
            });

            check("logTable", () -> {
                List<String[]> rows = new ArrayList<>();
                rows.add(new String[]{"1", "1.20", "0.55"});
                rows.add(new String[]{"10", "0.80", "0.72"});
                rows.add(new String[]{"20", "0.45", "0.84"});
                rows.add(new String[]{"40", "0.22", "0.91"});
                wb.logTable("epoch_metrics",
                        new String[]{"epoch", "loss", "acc"}, rows);
            });

            check("logSummary", () -> {
                Map<String, Object> sum = new LinkedHashMap<>();
                sum.put("best_acc", 0.91);
                sum.put("best_loss", 0.22);
                sum.put("total_steps", 40);
                wb.logSummary(sum);
            });

            check("WandbTrainingMonitor e2e", () -> {
                try (WandbClient wb2 = WandbClient.newBuilder()
                        .offline(server).entity("javacpp").project("pytorch-bench")
                        .apiKey("local").build();
                     WandbTrainingMonitor mon = new WandbTrainingMonitor(
                             wb2, "monitor-run",
                             Map.of("opt", "adam", "hidden", "128"), false)) {
                    for (int s = 0; s < 15; s++) {
                        mon.log(Map.of(
                                "loss", 1.5 * Math.exp(-s / 5.0),
                                "acc", 1 - Math.exp(-s / 6.0)));
                    }
                    double[][] hm = new double[6][6];
                    for (int i = 0; i < 6; i++) {
                        hm[i][i] = 10;
                        for (int j = 0; j < 6; j++) if (i != j) hm[i][j] = RND.nextInt(3);
                    }
                    mon.logHeatmap("cm", hm);
                    mon.logText("cfg", "monitor e2e ok");
                    System.out.println("    monitor ui: " + mon.uiUrl());
                }
            });

            check("HTTP export endpoint", () -> {
                HttpClient http = HttpClient.newHttpClient();
                String url = server.uiUrl() + "/api/runs/" + wb.runId() + "/export";
                HttpResponse<String> resp = http.send(
                        HttpRequest.newBuilder(URI.create(url)).GET()
                                .timeout(Duration.ofSeconds(5)).build(),
                        HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() != 200)
                    throw new AssertionError("export HTTP " + resp.statusCode());
                if (!resp.body().contains("\"n_metrics\"") && !resp.body().contains("metrics"))
                    throw new AssertionError("export missing metrics: " + resp.body().substring(0, Math.min(200, resp.body().length())));
                System.out.println("    export bytes: " + resp.body().length());
            });

            check("dashboard HTML reachable", () -> {
                HttpClient http = HttpClient.newHttpClient();
                HttpResponse<String> index = http.send(
                        HttpRequest.newBuilder(URI.create(server.uiUrl() + "/")).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
                if (index.statusCode() != 200 || !index.body().contains("WandB"))
                    throw new AssertionError("index page bad");
                HttpResponse<String> run = http.send(
                        HttpRequest.newBuilder(URI.create(wb.uiUrl())).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
                if (run.statusCode() != 200)
                    throw new AssertionError("run page HTTP " + run.statusCode());
                if (!run.body().contains("Plotly") && !run.body().contains("plotly"))
                    throw new AssertionError("run page missing plotly");
                if (!run.body().contains("heatmap") && !run.body().contains("Metrics"))
                    throw new AssertionError("run page missing expected sections");
                System.out.println("    run page bytes: " + run.body().length());
            });

            wb.finish();

            String ui = wb.uiUrl();
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

        // Keep server up for manual inspection
        // Keep-alive: BENCH_KEEPALIVE_SEC (default 120). Set 0 to exit immediately after checks.
        int keepSec = 120;
        try {
            String env = System.getenv("BENCH_KEEPALIVE_SEC");
            if (env != null && !env.isBlank()) keepSec = Integer.parseInt(env.trim());
        } catch (Exception ignored) { /* keep default */ }
        if (keepSec > 0) {
            System.out.println("Server kept alive " + keepSec + "s for manual verification at " + server.uiUrl());
            System.out.println("Press Ctrl+C to stop earlier. (BENCH_KEEPALIVE_SEC=0 to skip)");
            try { Thread.sleep(keepSec * 1000L); } catch (InterruptedException ignored) {}
        }
        server.close();
    }
}
