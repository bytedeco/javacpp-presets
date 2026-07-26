package org.bytedeco.pytorch.utils.wandb;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.utils.tensorboard.PngEncoder;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Lightweight WandB-compatible experiment tracker for JavaCPP / LibTorch.
 *
 * <p>Two modes:
 * <ul>
 *   <li><b>Remote</b> — HTTP REST against a WandB-compatible backend
 *       ({@code host:port/api/...}).</li>
 *   <li><b>Offline / local</b> — spin {@link WandbLocalServer} (embedded JDK
 *       {@link com.sun.net.httpserver.HttpServer}) that stores runs on disk and
 *       serves a simple HTML dashboard so you can open a browser URL and verify
 *       metrics, heatmaps, images, tables without a cloud API key.</li>
 * </ul>
 *
 * <pre>{@code
 * // Offline demo (no API key required):
 * try (WandbLocalServer server = WandbLocalServer.start(0);   // ephemeral port
 *      WandbClient wb = WandbClient.newBuilder()
 *              .offline(server)
 *              .entity("local").project("demo").build()) {
 *     wb.initRun("exp1", Map.of("lr", "1e-3"));
 *     wb.log(Map.of("loss", 0.42, "acc", 0.91), 1);
 *     wb.logHeatmap("cm", matrix, 1);
 *     wb.logImage("sample", imageTensor, 1);
 *     System.out.println("open " + server.uiUrl());
 * }
 * }</pre>
 */
public final class WandbClient implements AutoCloseable {

    public enum ChartType { LINE, BAR, HISTOGRAM, SURFACE, HEATMAP, SCATTER, PIE }

    private final HttpClient http;
    private final URI baseUri;                 // …/api
    private final String apiKey;
    private final String entity;
    private final String project;
    private final WandbLocalServer local;      // non-null in offline mode
    private final Path runDir;                 // offline artifact dir (optional)
    private String runId;
    private String runName;
    private final AtomicLong stepCounter = new AtomicLong(0);
    private boolean finished;

    private WandbClient(Builder b) {
        this.local = b.localServer;
        this.apiKey = b.apiKey == null ? "local-key" : b.apiKey;
        this.entity = Objects.requireNonNull(b.entity, "entity");
        this.project = Objects.requireNonNull(b.project, "project");
        this.runDir = b.runDir;
        this.http = HttpClient.newBuilder().connectTimeout(b.connectTimeout).build();
        if (local != null) {
            this.baseUri = URI.create(local.apiBase());
        } else {
            String scheme = b.useHttps ? "https" : "http";
            this.baseUri = URI.create(String.format("%s://%s:%d/api", scheme, b.host, b.port));
        }
    }

    public static Builder newBuilder() { return new Builder(); }

    public String runId() { return runId; }
    public String runName() { return runName; }
    public String entity() { return entity; }
    public String project() { return project; }
    public String uiUrl() {
        if (local != null && runId != null) {
            return local.uiUrl() + "/runs/" + entity + "/" + project + "/" + runId;
        }
        return baseUri.resolve("../" + entity + "/" + project).toString();
    }

    // =========================================================================
    // Run lifecycle
    // =========================================================================

    public void initRun(String name) throws IOException, InterruptedException {
        initRun(name, Map.of());
    }

    public void initRun(String name, Map<String, ?> config) throws IOException, InterruptedException {
        this.runName = name == null ? "run-" + System.currentTimeMillis() : name;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("name", runName);
        payload.put("entity", entity);
        payload.put("project", project);
        payload.put("config", config == null ? Map.of() : config);
        payload.put("created_at", Instant.now().toString());
        Map<String, Object> resp = post("/runs", payload);
        this.runId = String.valueOf(resp.getOrDefault("id",
                UUID.randomUUID().toString().replace("-", "").substring(0, 8)));
        this.finished = false;
        if (runDir != null) {
            Files.createDirectories(runDir.resolve(runId));
            Files.writeString(runDir.resolve(runId).resolve("config.json"),
                    Json.encode(payload), StandardCharsets.UTF_8);
        }
    }

    public void finish() throws IOException, InterruptedException {
        if (runId == null || finished) return;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("status", "finished");
        payload.put("finished_at", Instant.now().toString());
        post("/runs/" + runId, payload);
        finished = true;
    }

    // =========================================================================
    // Metrics
    // =========================================================================

    public void log(Map<String, ? extends Number> metrics, long step)
            throws IOException, InterruptedException {
        requireRun();
        stepCounter.set(Math.max(stepCounter.get(), step));
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("step", step);
        payload.put("metrics", metrics);
        payload.put("timestamp", Instant.now().toString());
        post("/metrics", payload);
    }

    public void log(Map<String, ? extends Number> metrics) throws IOException, InterruptedException {
        log(metrics, stepCounter.incrementAndGet());
    }

    public void logMetrics(Map<String, Number> metrics, long step, Map<String, Object> chartOpts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("step", step);
        payload.put("metrics", metrics);
        if (chartOpts != null) payload.put("chart", chartOpts);
        payload.put("timestamp", Instant.now().toString());
        post("/metrics", payload);
    }

    // =========================================================================
    // Charts
    // =========================================================================

    public void logChart(String name, ChartType type, List<List<Double>> series,
                         String[] legends, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", type.name().toLowerCase());
        payload.put("step", step);
        payload.put("series", series);
        if (legends != null) payload.put("legend", legends);
        post("/charts", payload);
    }

    public void logHeatmap(String name, double[][] matrix, long step)
            throws IOException, InterruptedException {
        logHeatmap(name, matrix, step, null);
    }

    public void logHeatmap(String name, double[][] matrix, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "heatmap");
        payload.put("step", step);
        payload.put("matrix", matrix);
        if (opts != null) payload.put("opts", opts);
        post("/charts", payload);
    }

    public void logHeatmap(String name, Tensor t, long step)
            throws IOException, InterruptedException {
        logHeatmap(name, tensorToMatrix(t), step, null);
    }

    public void logHistogram(String name, double[] values, int bins, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "histogram");
        payload.put("step", step);
        payload.put("values", values);
        payload.put("bins", bins);
        post("/charts", payload);
    }

    public void logScatter(String name, double[][] points, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "scatter");
        payload.put("step", step);
        payload.put("points", points);
        post("/charts", payload);
    }

    // =========================================================================
    // Tables / images / text / audio
    // =========================================================================

    public void logTable(String tableName, String[] columns, List<String[]> rows)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", tableName);
        payload.put("columns", columns);
        payload.put("rows", rows);
        post("/tables", payload);
    }

    public void logImage(String name, byte[] pngBytes, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("bytes", pngBytes); // Json encodes as base64
        payload.put("format", "png");
        if (opts != null) payload.put("opts", opts);
        post("/images", payload);
        if (runDir != null) {
            Path img = runDir.resolve(runId).resolve("images");
            Files.createDirectories(img);
            Files.write(img.resolve(name.replace('/', '_') + "_s" + step + ".png"), pngBytes);
        }
    }

    public void logImage(String name, Tensor image, long step)
            throws IOException, InterruptedException {
        ImageBuf buf = tensorToPng(image);
        Map<String, Object> opts = new LinkedHashMap<>();
        opts.put("height", buf.height);
        opts.put("width", buf.width);
        opts.put("channels", buf.channels);
        logImage(name, buf.png, step, opts);
    }

    public void logImages(String name, Tensor batchNchw, long step)
            throws IOException, InterruptedException {
        // Take first image of batch for simplicity; full grid optional
        if (batchNchw.dim() == 4) {
            logImage(name, batchNchw.select(0, 0), step);
        } else {
            logImage(name, batchNchw, step);
        }
    }

    public void logText(String name, String text, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("text", text);
        post("/text", payload);
    }

    public void logAudio(String name, float[] mono, int sampleRate, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("sample_rate", sampleRate);
        // store as list (JSON) — local server can render a sparkline; full WAV optional
        List<Double> samples = new ArrayList<>(Math.min(mono.length, 4000));
        int stride = Math.max(1, mono.length / 4000);
        for (int i = 0; i < mono.length; i += stride) samples.add((double) mono[i]);
        payload.put("waveform", samples);
        payload.put("n_samples", mono.length);
        post("/audio", payload);
    }

    public void logAudio(String name, Tensor waveform, int sampleRate, long step)
            throws IOException, InterruptedException {
        logAudio(name, toFloatArray(waveform), sampleRate, step);
    }

    public void logSummary(Map<String, ?> summary) throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("summary", summary);
        post("/summary", payload);
    }

    // =========================================================================
    // HTTP
    // =========================================================================

    private void requireRun() {
        if (runId == null) throw new IllegalStateException("call initRun() first");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> post(String path, Map<String, Object> payload)
            throws IOException, InterruptedException {
        // Fast path: in-process local server (no real HTTP roundtrip needed, but we still
        // go through HTTP so the protocol is exercised end-to-end).
        String json = Json.encode(payload);
        HttpRequest req = HttpRequest.newBuilder(baseUri.resolve(path.startsWith("/") ? path.substring(1) : path))
                .timeout(Duration.ofSeconds(15))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) {
            throw new IOException("WandB " + path + " failed HTTP " + resp.statusCode()
                    + ": " + resp.body());
        }
        if (resp.body() == null || resp.body().isBlank()) return new LinkedHashMap<>();
        Object decoded = Json.decode(resp.body());
        if (decoded instanceof Map) return (Map<String, Object>) decoded;
        Map<String, Object> wrap = new LinkedHashMap<>();
        wrap.put("value", decoded);
        return wrap;
    }

    @Override
    public void close() {
        try { finish(); } catch (Exception ignored) { /* best-effort */ }
    }

    // =========================================================================
    // Tensor helpers
    // =========================================================================

    static float[] toFloatArray(Tensor tensor) {
        if (tensor == null || !tensor.defined()) return new float[0];
        Tensor c = tensor.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.kFloat()).flatten();
        long n = c.numel();
        float[] data = new float[(int) Math.min(n, Integer.MAX_VALUE)];
        org.bytedeco.javacpp.FloatPointer p = c.data_ptr_float();
        for (int i = 0; i < data.length; i++) data[i] = p.get(i);
        return data;
    }

    static double[][] tensorToMatrix(Tensor t) {
        if (t.dim() != 2) throw new IllegalArgumentException("expected 2D tensor");
        int rows = (int) t.size(0), cols = (int) t.size(1);
        float[] flat = toFloatArray(t);
        double[][] m = new double[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r][c] = flat[r * cols + c];
        return m;
    }

    static final class ImageBuf {
        final byte[] png; final int height, width, channels;
        ImageBuf(byte[] png, int h, int w, int c) { this.png = png; height = h; width = w; channels = c; }
    }

    static ImageBuf tensorToPng(Tensor t) {
        long nd = t.dim();
        int c, h, w;
        float[] chw;
        if (nd == 2) {
            h = (int) t.size(0); w = (int) t.size(1); c = 1;
            chw = toFloatArray(t);
        } else if (nd == 3) {
            long d0 = t.size(0);
            if (d0 == 1 || d0 == 3 || d0 == 4) {
                c = (int) d0; h = (int) t.size(1); w = (int) t.size(2);
                chw = toFloatArray(t);
            } else {
                h = (int) d0; w = (int) t.size(1); c = (int) t.size(2);
                float[] hwc = toFloatArray(t);
                chw = new float[c * h * w];
                for (int ci = 0; ci < c; ci++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            chw[ci * h * w + hi * w + wi] = hwc[(hi * w + wi) * c + ci];
            }
        } else {
            throw new IllegalArgumentException("image tensor must be 2D/3D");
        }
        float[] hwc = new float[h * w * c];
        for (int ci = 0; ci < c; ci++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    hwc[(hi * w + wi) * c + ci] = chw[ci * h * w + hi * w + wi];
        int outC = c;
        if (c == 1) {
            float[] rgb = new float[h * w * 3];
            for (int i = 0; i < h * w; i++) {
                float v = hwc[i];
                rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
            }
            hwc = rgb; outC = 3;
        }
        return new ImageBuf(PngEncoder.encodeFloatHWC(hwc, h, w, outC), h, w, outC);
    }

    // =========================================================================
    // Builder
    // =========================================================================

    public static final class Builder {
        private String host = "localhost";
        private int port = 8080;
        private boolean useHttps = false;
        private Duration connectTimeout = Duration.ofSeconds(5);
        private String apiKey;
        private String entity = "local";
        private String project = "pytorch";
        private WandbLocalServer localServer;
        private Path runDir;

        public Builder host(String host) { this.host = host; return this; }
        public Builder port(int port) { this.port = port; return this; }
        public Builder useHttps(boolean v) { this.useHttps = v; return this; }
        public Builder connectTimeout(Duration d) { this.connectTimeout = d; return this; }
        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder entity(String entity) { this.entity = entity; return this; }
        public Builder project(String project) { this.project = project; return this; }
        /** Attach an in-process {@link WandbLocalServer} (offline mode). */
        public Builder offline(WandbLocalServer server) { this.localServer = server; return this; }
        public Builder runDir(Path dir) { this.runDir = dir; return this; }
        /** Convenience: also accept WANDB_API_KEY from the environment. */
        public Builder fromEnv() {
            String k = System.getenv("WANDB_API_KEY");
            if (k != null && !k.isBlank()) this.apiKey = k;
            String e = System.getenv("WANDB_ENTITY");
            if (e != null && !e.isBlank()) this.entity = e;
            String p = System.getenv("WANDB_PROJECT");
            if (p != null && !p.isBlank()) this.project = p;
            return this;
        }

        public WandbClient build() { return new WandbClient(this); }
    }
}
