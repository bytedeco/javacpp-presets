package org.bytedeco.pytorch.utils.visdom;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.utils.tensorboard.PngEncoder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Pure-Java Visdom client matching the real Python {@code visdom.Visdom} HTTP protocol.
 *
 * <p>Posts JSON payloads to {@code /events} (create) and {@code /update} (append/replace).
 * Heatmaps use Plotly {@code type=heatmap} with a {@code z} matrix; images are sent as
 * base64 data-URIs; line/scatter share the Plotly scatter path.
 *
 * <pre>{@code
 * try (VisdomClient viz = VisdomClient.newBuilder().env("main").build()) {
 *     if (!viz.checkConnection()) throw new IllegalStateException("visdom not running");
 *     viz.line(new double[]{1,2,3}, new double[]{1,4,9}, "loss", opts("title", "Loss"));
 *     viz.heatmap(matrix, "cm", opts("title", "Confusion", "colormap", "Viridis"));
 *     viz.image(chwTensor, "sample");   // CxHxW float or byte Tensor
 * }
 * // open http://localhost:8097
 * }</pre>
 *
 * <p>Requires a running Visdom server: {@code python -m visdom.server -port 8097}.
 */
public final class VisdomClient implements AutoCloseable {

    private final HttpClient httpClient;
    private final String baseUrl;          // e.g. http://localhost:8097
    private final String env;
    private final Duration requestTimeout;
    private final ExecutorService executor;
    private final boolean raiseExceptions;
    private final boolean ownExecutor;

    private VisdomClient(Builder b) {
        this.ownExecutor = b.executor == null;
        this.executor = b.executor != null ? b.executor : Executors.newFixedThreadPool(Math.max(2, b.executorThreads));
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(b.connectTimeout)
                .executor(this.executor)
                .build();
        String scheme = b.useHttps ? "https" : "http";
        String path = b.basePath == null || b.basePath.isEmpty() ? "" :
                (b.basePath.startsWith("/") ? b.basePath : "/" + b.basePath);
        if (path.endsWith("/")) path = path.substring(0, path.length() - 1);
        this.baseUrl = String.format("%s://%s:%d%s", scheme, b.host, b.port, path);
        this.env = b.env;
        this.requestTimeout = b.requestTimeout;
        this.raiseExceptions = b.raiseExceptions;
    }

    public static Builder newBuilder() { return new Builder(); }

    public String baseUrl() { return baseUrl; }
    public String env() { return env; }
    public String uiUrl() { return baseUrl + "/env/" + env; }

    // =========================================================================
    // Connection
    // =========================================================================

    /** Probe {@code /} or {@code /env/<env>} — returns true if the server answers. */
    public boolean checkConnection() {
        return checkConnection(Duration.ofSeconds(2));
    }

    public boolean checkConnection(Duration timeout) {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/"))
                    .timeout(timeout)
                    .GET()
                    .build();
            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() >= 200 && resp.statusCode() < 500;
        } catch (Exception e) {
            return false;
        }
    }

    // =========================================================================
    // Lifecycle helpers
    // =========================================================================

    public VisdomResponse closeWindow(String win) throws IOException, InterruptedException {
        Map<String, Object> msg = base(win);
        return send(msg, "close");
    }

    public VisdomResponse closeAll() throws IOException, InterruptedException {
        Map<String, Object> msg = new LinkedHashMap<>();
        msg.put("win", null);
        msg.put("eid", env);
        return send(msg, "close");
    }

    public VisdomResponse save(String... envs) throws IOException, InterruptedException {
        Map<String, Object> msg = new LinkedHashMap<>();
        msg.put("data", Arrays.asList(envs));
        return send(msg, "save");
    }

    // =========================================================================
    // Text / properties / table
    // =========================================================================

    public VisdomResponse text(String content) throws IOException, InterruptedException {
        return text(content, null, null, false);
    }

    public VisdomResponse text(String content, String win) throws IOException, InterruptedException {
        return text(content, win, null, false);
    }

    public VisdomResponse text(String content, String win, Map<String, Object> opts, boolean append)
            throws IOException, InterruptedException {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("content", content);
        data.put("type", "text");
        Map<String, Object> msg = base(win);
        msg.put("data", List.of(data));
        msg.put("opts", cleanOpts(opts));
        return send(msg, append ? "update" : "events");
    }

    public VisdomResponse appendText(String win, String content) throws IOException, InterruptedException {
        return text(content, win, null, true);
    }

    /**
     * Render a simple HTML table (Visdom has no first-class table endpoint —
     * we send HTML via {@link #text}).
     */
    public VisdomResponse table(String[] columns, String[][] rows, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        StringBuilder html = new StringBuilder();
        html.append("<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font-family:monospace'>");
        if (columns != null) {
            html.append("<thead><tr>");
            for (String c : columns) html.append("<th>").append(escHtml(c)).append("</th>");
            html.append("</tr></thead>");
        }
        html.append("<tbody>");
        if (rows != null) {
            for (String[] row : rows) {
                html.append("<tr>");
                for (String cell : row) html.append("<td>").append(escHtml(cell)).append("</td>");
                html.append("</tr>");
            }
        }
        html.append("</tbody></table>");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.putIfAbsent("title", "table");
        return text(html.toString(), win, o, false);
    }

    public VisdomResponse properties(List<Map<String, Object>> props, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("content", props);
        data.put("type", "properties");
        Map<String, Object> msg = base(win);
        msg.put("data", List.of(data));
        msg.put("opts", cleanOpts(opts));
        return send(msg, "events");
    }

    // =========================================================================
    // Line
    // =========================================================================

    /**
     * Single-trace line: {@code Y} length N, optional {@code X} length N.
     * Returns the window id string in {@link VisdomResponse#windowId()}.
     */
    public VisdomResponse line(double[] Y, double[] X, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return line(Y, X, win, opts, null, null);
    }

    public VisdomResponse line(double[] Y, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return line(Y, null, win, opts, null, null);
    }

    /**
     * Full line API matching Python {@code vis.line(Y, X=…, win=…, update=…, name=…)}.
     *
     * @param update {@code null}, {@code "append"}, {@code "replace"}, or {@code "remove"}
     * @param name   trace name when updating a single series
     */
    public VisdomResponse line(double[] Y, double[] X, String win, Map<String, Object> opts,
                               String update, String name)
            throws IOException, InterruptedException {
        Objects.requireNonNull(Y, "Y");
        if (X == null) {
            X = linspace(0, 1, Y.length);
        }
        if (X.length != Y.length) {
            throw new IllegalArgumentException("X and Y must have same length, got "
                    + X.length + " vs " + Y.length);
        }
        // Build Nx2 points and delegate to scatter with mode=lines
        double[][] pts = new double[Y.length][2];
        for (int i = 0; i < Y.length; i++) {
            pts[i][0] = X[i];
            pts[i][1] = Y[i];
        }
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.putIfAbsent("markers", Boolean.FALSE);
        o.putIfAbsent("mode", Boolean.TRUE.equals(o.get("markers")) ? "lines+markers" : "lines");
        return scatter(pts, null, win, o, update, name);
    }

    /**
     * Multi-trace line. {@code Y} is N×M (rows=points, cols=traces); {@code X} is N or N×M.
     */
    public VisdomResponse lineMultiple(double[][] Y, double[] X, String[] legends,
                                       String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(Y, "Y");
        if (Y.length == 0) throw new IllegalArgumentException("Y empty");
        int n = Y.length;
        int m = Y[0].length;
        if (X == null) X = linspace(0, 1, n);
        if (X.length != n) throw new IllegalArgumentException("X length must equal Y rows");

        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.putIfAbsent("mode", "lines");
        if (legends != null) o.put("legend", Arrays.asList(legends));

        List<Map<String, Object>> data = new ArrayList<>();
        for (int k = 0; k < m; k++) {
            double[] xs = new double[n];
            double[] ys = new double[n];
            for (int i = 0; i < n; i++) {
                xs[i] = X[i];
                ys[i] = Y[i][k];
            }
            Map<String, Object> trace = new LinkedHashMap<>();
            trace.put("x", xs);
            trace.put("y", ys);
            trace.put("type", "scatter");
            trace.put("mode", o.getOrDefault("mode", "lines"));
            String name = legends != null && k < legends.length ? legends[k] : String.valueOf(k + 1);
            trace.put("name", name);
            data.add(trace);
        }
        Map<String, Object> msg = base(win);
        msg.put("data", data);
        msg.put("layout", opts2layout(o, false));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    /**
     * Append a single (x,y) point to an existing line window (efficient streaming).
     */
    public VisdomResponse lineAppend(String win, double x, double y, String name, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return line(new double[]{y}, new double[]{x}, win, opts, "append", name);
    }

    // =========================================================================
    // Scatter (2D / 3D)
    // =========================================================================

    /**
     * Scatter. {@code points} is N×2 or N×3. Optional {@code labels} length N with values ≥ 1.
     */
    public VisdomResponse scatter(double[][] points, int[] labels, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return scatter(points, labels, win, opts, null, null);
    }

    public VisdomResponse scatter(double[][] points, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return scatter(points, null, win, opts, null, null);
    }

    public VisdomResponse scatter(double[][] points, int[] labels, String win,
                                  Map<String, Object> opts, String update, String name)
            throws IOException, InterruptedException {
        if ("remove".equals(update)) {
            Objects.requireNonNull(win, "win");
            Objects.requireNonNull(name, "name");
            Map<String, Object> msg = base(win);
            msg.put("data", List.of());
            msg.put("name", name);
            msg.put("delete", true);
            return send(msg, "update");
        }
        Objects.requireNonNull(points, "points");
        if (points.length == 0) throw new IllegalArgumentException("points empty");
        int dim = points[0].length;
        if (dim != 2 && dim != 3) throw new IllegalArgumentException("points must be Nx2 or Nx3");
        boolean is3d = dim == 3;

        int n = points.length;
        int[] Y;
        if (labels == null) {
            Y = new int[n];
            Arrays.fill(Y, 1);
        } else {
            if (labels.length != n) throw new IllegalArgumentException("labels length mismatch");
            Y = labels;
        }

        // unique labels
        int maxLabel = 1;
        for (int y : Y) if (y > maxLabel) maxLabel = y;
        List<Integer> uniq = new ArrayList<>();
        boolean[] seen = new boolean[maxLabel + 1];
        for (int y : Y) {
            if (y < 1) throw new IllegalArgumentException("labels must be >= 1");
            if (!seen[y]) { seen[y] = true; uniq.add(y); }
        }

        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.putIfAbsent("mode", "markers");
        o.putIfAbsent("markersymbol", "dot");
        o.putIfAbsent("markersize", 10);
        o.putIfAbsent("markerborderwidth", 0.5);

        @SuppressWarnings("unchecked")
        List<String> legend = o.get("legend") instanceof List
                ? (List<String>) o.get("legend")
                : (o.get("legend") instanceof String[] sa ? Arrays.asList(sa) : null);

        List<Map<String, Object>> data = new ArrayList<>();
        for (int k : uniq) {
            List<Double> xs = new ArrayList<>();
            List<Double> ys = new ArrayList<>();
            List<Double> zs = is3d ? new ArrayList<>() : null;
            for (int i = 0; i < n; i++) {
                if (Y[i] != k) continue;
                xs.add(points[i][0]);
                ys.add(points[i][1]);
                if (is3d) zs.add(points[i][2]);
            }
            if (xs.isEmpty()) continue;
            String traceName;
            if (legend != null && k - 1 < legend.size()) traceName = legend.get(k - 1);
            else if (uniq.size() == 1 && name != null) traceName = name;
            else traceName = String.valueOf(k);

            Map<String, Object> trace = new LinkedHashMap<>();
            trace.put("x", xs);
            trace.put("y", ys);
            if (is3d) trace.put("z", zs);
            trace.put("name", traceName);
            trace.put("type", is3d ? "scatter3d" : "scatter");
            trace.put("mode", o.get("mode"));
            Map<String, Object> marker = new LinkedHashMap<>();
            marker.put("size", o.get("markersize"));
            marker.put("symbol", o.get("markersymbol"));
            Map<String, Object> mline = new LinkedHashMap<>();
            mline.put("color", "#000000");
            mline.put("width", o.get("markerborderwidth"));
            marker.put("line", mline);
            trace.put("marker", marker);
            if (Boolean.TRUE.equals(o.get("fillarea"))) trace.put("fill", "tonexty");
            data.add(trace);
        }

        Map<String, Object> msg = base(win);
        msg.put("data", data);
        if (update == null) {
            msg.put("layout", opts2layout(o, is3d));
        } else {
            msg.put("layout", Map.of());
            msg.put("name", name);
            msg.put("append", "append".equals(update));
        }
        msg.put("opts", cleanOpts(o));
        return send(msg, update == null ? "events" : "update");
    }

    public VisdomResponse scatter3D(double[] x, double[] y, double[] z, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        if (x.length != y.length || x.length != z.length) {
            throw new IllegalArgumentException("x,y,z length mismatch");
        }
        double[][] pts = new double[x.length][3];
        for (int i = 0; i < x.length; i++) {
            pts[i][0] = x[i]; pts[i][1] = y[i]; pts[i][2] = z[i];
        }
        return scatter(pts, null, win, opts);
    }

    // =========================================================================
    // Heatmap  ★ critical path
    // =========================================================================

    /**
     * Heatmap. {@code matrix} is rows × cols (row 0 at the top in rownames order).
     * Matches Python: {@code data: [{z: matrix, type: 'heatmap', colorscale: …}]}.
     */
    public VisdomResponse heatmap(double[][] matrix, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return heatmap(matrix, win, opts, null);
    }

    public VisdomResponse heatmap(double[][] matrix, String win, Map<String, Object> opts, String update)
            throws IOException, InterruptedException {
        Objects.requireNonNull(matrix, "matrix");
        if (matrix.length == 0) throw new IllegalArgumentException("matrix empty");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.putIfAbsent("colormap", "Viridis");

        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("z", matrix);
        if (o.get("columnnames") != null) trace.put("x", o.get("columnnames"));
        if (o.get("rownames") != null) trace.put("y", o.get("rownames"));
        if (o.get("xmin") != null) trace.put("zmin", o.get("xmin"));
        if (o.get("xmax") != null) trace.put("zmax", o.get("xmax"));
        trace.put("type", "heatmap");
        trace.put("colorscale", o.get("colormap"));

        List<Map<String, Object>> data = new ArrayList<>();
        data.add(trace);

        Map<String, Object> msg = base(win);
        msg.put("data", data);
        if (update == null) {
            msg.put("layout", opts2layout(o, false));
        } else {
            msg.put("layout", Map.of());
            boolean appending = update.startsWith("append") || update.startsWith("prepend");
            msg.put("append", appending);
            msg.put("updateDir", update);
        }
        msg.put("opts", cleanOpts(o));
        return send(msg, update == null ? "events" : "update");
    }

    /** Convenience: build heatmap from a 2-D float Tensor (rows × cols). */
    public VisdomResponse heatmap(Tensor t, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return heatmap(tensorToMatrix(t), win, opts);
    }

    // =========================================================================
    // Surface / contour
    // =========================================================================

    public VisdomResponse surf(double[][] Z, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return surfaceLike(Z, "surface", win, opts, true);
    }

    public VisdomResponse surface(double[][] Z, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return surf(Z, win, opts);
    }

    public VisdomResponse contour(double[][] Z, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return surfaceLike(Z, "contour", win, opts, false);
    }

    private VisdomResponse surfaceLike(double[][] Z, String stype, String win,
                                       Map<String, Object> opts, boolean is3d)
            throws IOException, InterruptedException {
        Objects.requireNonNull(Z, "Z");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (double[] row : Z) for (double v : row) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        o.putIfAbsent("xmin", min);
        o.putIfAbsent("xmax", max);
        o.putIfAbsent("colormap", "Viridis");

        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("z", Z);
        trace.put("cmin", o.get("xmin"));
        trace.put("cmax", o.get("xmax"));
        trace.put("type", stype);
        trace.put("colorscale", o.get("colormap"));

        Map<String, Object> msg = base(win);
        msg.put("data", List.of(trace));
        msg.put("layout", opts2layout(o, is3d));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    // =========================================================================
    // Bar / histogram / boxplot / pie
    // =========================================================================

    public VisdomResponse bar(double[] values, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return bar(values, null, win, opts);
    }

    public VisdomResponse bar(double[] values, double[] x, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(values, "values");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        Object xx = o.get("rownames");
        if (xx == null) {
            if (x != null) xx = x;
            else {
                double[] auto = new double[values.length];
                for (int i = 0; i < auto.length; i++) auto[i] = i + 1;
                xx = auto;
            }
        }
        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("y", values);
        trace.put("x", xx);
        trace.put("type", "bar");
        if (o.get("legend") instanceof List<?> leg && !leg.isEmpty()) {
            trace.put("name", String.valueOf(leg.get(0)));
        }
        Map<String, Object> msg = base(win);
        msg.put("data", List.of(trace));
        msg.put("layout", opts2layout(o, false));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    /** Grouped / stacked bar: columns of {@code values} (N×M) become series. */
    public VisdomResponse barGrouped(double[][] values, String[] legends, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(values, "values");
        int n = values.length;
        int m = values[0].length;
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.putIfAbsent("stacked", false);
        if (legends != null) o.put("legend", Arrays.asList(legends));

        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = i + 1;

        List<Map<String, Object>> data = new ArrayList<>();
        for (int k = 0; k < m; k++) {
            double[] y = new double[n];
            for (int i = 0; i < n; i++) y[i] = values[i][k];
            Map<String, Object> trace = new LinkedHashMap<>();
            trace.put("y", y);
            trace.put("x", o.getOrDefault("rownames", x));
            trace.put("type", "bar");
            if (legends != null && k < legends.length) trace.put("name", legends[k]);
            data.add(trace);
        }
        Map<String, Object> msg = base(win);
        msg.put("data", data);
        msg.put("layout", opts2layout(o, false));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    public VisdomResponse histogram(double[] values, int bins, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(values, "values");
        if (bins <= 0) bins = Math.min(30, values.length);
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (double v : values) { if (v < min) min = v; if (v > max) max = v; }
        if (min == max) { min -= 0.5; max += 0.5; }
        double width = (max - min) / bins;
        double[] counts = new double[bins];
        double[] centers = new double[bins];
        for (int i = 0; i < bins; i++) centers[i] = min + (i + 0.5) * width;
        for (double v : values) {
            int b = (int) ((v - min) / width);
            if (b < 0) b = 0;
            if (b >= bins) b = bins - 1;
            counts[b]++;
        }
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.put("numbins", bins);
        return bar(counts, centers, win, o);
    }

    public VisdomResponse histogram(double[] values, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return histogram(values, 30, win, opts);
    }

    /** Boxplot: each column of {@code sequences} (N×M) is one box. */
    public VisdomResponse boxplot(double[][] sequences, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(sequences, "sequences");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        int n = sequences.length;
        int m = sequences[0].length;
        @SuppressWarnings("unchecked")
        List<String> legend = o.get("legend") instanceof List
                ? (List<String>) o.get("legend") : null;

        List<Map<String, Object>> data = new ArrayList<>();
        for (int k = 0; k < m; k++) {
            double[] y = new double[n];
            for (int i = 0; i < n; i++) y[i] = sequences[i][k];
            Map<String, Object> trace = new LinkedHashMap<>();
            trace.put("y", y);
            trace.put("type", "box");
            trace.put("name", legend != null && k < legend.size() ? legend.get(k) : "column " + k);
            data.add(trace);
        }
        Map<String, Object> msg = base(win);
        msg.put("data", data);
        msg.put("layout", opts2layout(o, false));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    public VisdomResponse pie(double[] values, String[] labels, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(values, "values");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        if (labels != null) o.putIfAbsent("legend", Arrays.asList(labels));
        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("values", values);
        trace.put("labels", labels != null ? labels : o.get("legend"));
        trace.put("type", "pie");
        Map<String, Object> msg = base(win);
        msg.put("data", List.of(trace));
        msg.put("layout", opts2layout(o, false));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    // =========================================================================
    // Stem / quiver / mesh
    // =========================================================================

    public VisdomResponse stem(double[] Y, double[] X, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(Y, "Y");
        if (X == null) {
            X = new double[Y.length];
            for (int i = 0; i < X.length; i++) X[i] = i + 1;
        }
        // stem = scatter with lines+markers from 0 to y
        double[][] pts = new double[Y.length * 3][2];
        int[] labels = new int[Y.length * 3];
        for (int i = 0; i < Y.length; i++) {
            int b = i * 3;
            pts[b][0] = X[i]; pts[b][1] = 0; labels[b] = 1;
            pts[b + 1][0] = X[i]; pts[b + 1][1] = Y[i]; labels[b + 1] = 1;
            pts[b + 2][0] = Double.NaN; pts[b + 2][1] = Double.NaN; labels[b + 2] = 1;
        }
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.put("mode", "lines+markers");
        return scatter(pts, labels, win, o);
    }

    public VisdomResponse quiver(double[][] U, double[][] V, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(U, "U");
        Objects.requireNonNull(V, "V");
        int rows = U.length, cols = U[0].length;
        // Build line segments for each arrow
        List<Double> xs = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        double maxLen = 0;
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
            double len = Math.hypot(U[r][c], V[r][c]);
            if (len > maxLen) maxLen = len;
        }
        double scale = maxLen > 0 ? 0.8 / maxLen : 1.0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double x0 = c, y0 = r;
                double x1 = x0 + U[r][c] * scale;
                double y1 = y0 + V[r][c] * scale;
                xs.add(x0); ys.add(y0);
                xs.add(x1); ys.add(y1);
                xs.add(Double.NaN); ys.add(Double.NaN);
            }
        }
        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("x", xs);
        trace.put("y", ys);
        trace.put("type", "scatter");
        trace.put("mode", "lines");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        Map<String, Object> msg = base(win);
        msg.put("data", List.of(trace));
        msg.put("layout", opts2layout(o, false));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    /**
     * Mesh. {@code vertices} is N×2 or N×3; optional {@code faces} is M×3 (0-based indices).
     */
    public VisdomResponse mesh(double[][] vertices, int[][] faces, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(vertices, "vertices");
        int dim = vertices[0].length;
        boolean is3d = dim == 3;
        double[] x = new double[vertices.length];
        double[] y = new double[vertices.length];
        double[] z = is3d ? new double[vertices.length] : null;
        for (int i = 0; i < vertices.length; i++) {
            x[i] = vertices[i][0];
            y[i] = vertices[i][1];
            if (is3d) z[i] = vertices[i][2];
        }
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("x", x);
        trace.put("y", y);
        if (is3d) trace.put("z", z);
        if (faces != null && faces.length > 0) {
            int[] ii = new int[faces.length];
            int[] jj = new int[faces.length];
            int[] kk = new int[faces.length];
            for (int f = 0; f < faces.length; f++) {
                ii[f] = faces[f][0];
                jj[f] = faces[f][1];
                kk[f] = faces[f].length > 2 ? faces[f][2] : 0;
            }
            trace.put("i", ii);
            trace.put("j", jj);
            if (is3d) trace.put("k", kk);
        }
        if (o.get("color") != null) trace.put("color", o.get("color"));
        if (o.get("opacity") != null) trace.put("opacity", o.get("opacity"));
        trace.put("type", is3d ? "mesh3d" : "mesh");

        Map<String, Object> msg = base(win);
        msg.put("data", List.of(trace));
        msg.put("layout", opts2layout(o, is3d));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    // =========================================================================
    // Image / images  (multimodal)
    // =========================================================================

    /**
     * Image from raw CHW float pixels in [0,1] or [0,255]. Encoded as PNG data-URI —
     * matches real Visdom protocol ({@code type: image, content.src: data:image/png;base64,…}).
     */
    public VisdomResponse image(float[] chw, int channels, int height, int width,
                                String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(chw, "chw");
        if (chw.length != (long) channels * height * width) {
            throw new IllegalArgumentException("pixel buffer size != C*H*W");
        }
        // CHW → HWC
        float[] hwc = new float[height * width * channels];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    hwc[(h * width + w) * channels + c] = chw[c * height * width + h * width + w];
                }
            }
        }
        // grayscale → RGB for Visdom compatibility
        int outC = channels;
        if (channels == 1) {
            float[] rgb = new float[height * width * 3];
            for (int i = 0; i < height * width; i++) {
                float v = hwc[i];
                rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
            }
            hwc = rgb;
            outC = 3;
        }
        byte[] png = PngEncoder.encodeFloatHWC(hwc, height, width, outC);
        return imagePng(png, win, opts, height, width);
    }

    /** Image from a CHW or HWC Tensor (float/byte). */
    public VisdomResponse image(Tensor t, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        float[] chw; int c, h, w;
        long nd = t.dim();
        if (nd == 2) {
            h = (int) t.size(0); w = (int) t.size(1); c = 1;
            chw = tensorToFloat(t);
        } else if (nd == 3) {
            // Heuristic: if dim0 ∈ {1,3,4} treat as CHW, else HWC
            long d0 = t.size(0);
            if (d0 == 1 || d0 == 3 || d0 == 4) {
                c = (int) d0; h = (int) t.size(1); w = (int) t.size(2);
                chw = tensorToFloat(t);
            } else {
                h = (int) d0; w = (int) t.size(1); c = (int) t.size(2);
                float[] hwc = tensorToFloat(t);
                chw = new float[c * h * w];
                for (int ci = 0; ci < c; ci++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            chw[ci * h * w + hi * w + wi] = hwc[(hi * w + wi) * c + ci];
            }
        } else {
            throw new IllegalArgumentException("image tensor must be 2D or 3D, got dim=" + nd);
        }
        return image(chw, c, h, w, win, opts);
    }

    public VisdomResponse imagePng(byte[] pngBytes, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return imagePng(pngBytes, win, opts, -1, -1);
    }

    private VisdomResponse imagePng(byte[] pngBytes, String win, Map<String, Object> opts,
                                    int height, int width)
            throws IOException, InterruptedException {
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        if (width > 0) o.putIfAbsent("width", width);
        if (height > 0) o.putIfAbsent("height", height);
        String b64 = Base64.getEncoder().encodeToString(pngBytes);
        Map<String, Object> content = new LinkedHashMap<>();
        content.put("src", "data:image/png;base64," + b64);
        if (o.get("caption") != null) content.put("caption", o.get("caption"));
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("content", content);
        data.put("type", Boolean.TRUE.equals(o.get("store_history")) ? "image_history" : "image");
        Map<String, Object> msg = base(win);
        msg.put("data", List.of(data));
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    /**
     * Image grid from NCHW batch (float). Arranges into {@code nrow} columns.
     */
    public VisdomResponse images(float[] nchw, int n, int c, int h, int w, int nrow,
                                 String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        if (nchw.length != (long) n * c * h * w) {
            throw new IllegalArgumentException("buffer size != N*C*H*W");
        }
        if (nrow <= 0) nrow = Math.min(8, n);
        int padding = 2;
        int xmaps = Math.min(nrow, n);
        int ymaps = (n + xmaps - 1) / xmaps;
        int gh = h + 2 * padding;
        int gw = w + 2 * padding;
        int outC = c == 1 ? 3 : c;
        float[] grid = new float[outC * (gh * ymaps) * (gw * xmaps)];
        // fill white-ish background
        Arrays.fill(grid, 1f);
        for (int i = 0; i < n; i++) {
            int row = i / xmaps, col = i % xmaps;
            int top = row * gh + padding;
            int left = col * gw + padding;
            for (int ci = 0; ci < c; ci++) {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        float v = nchw[i * c * h * w + ci * h * w + y * w + x];
                        int oc = c == 1 ? 0 : ci;
                        // write to all 3 channels if grayscale
                        if (c == 1) {
                            for (int k = 0; k < 3; k++) {
                                grid[k * (gh * ymaps) * (gw * xmaps)
                                        + (top + y) * (gw * xmaps) + (left + x)] = v;
                            }
                        } else {
                            grid[oc * (gh * ymaps) * (gw * xmaps)
                                    + (top + y) * (gw * xmaps) + (left + x)] = v;
                        }
                    }
                }
            }
        }
        return image(grid, outC, gh * ymaps, gw * xmaps, win, opts);
    }

    // =========================================================================
    // Audio (WAV data-URI embedded in text pane — matches Visdom)
    // =========================================================================

    public VisdomResponse audio(float[] mono, int sampleRate, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(mono, "mono");
        if (sampleRate <= 0) sampleRate = 44100;
        byte[] wav = encodeWav(mono, sampleRate);
        String b64 = Base64.getEncoder().encodeToString(wav);
        String html = "<audio controls><source type=\"audio/wav\" src=\"data:audio/wav;base64,"
                + b64 + "\">Your browser does not support the audio tag.</audio>";
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        o.putIfAbsent("height", 80);
        o.putIfAbsent("width", 330);
        o.putIfAbsent("sample_frequency", sampleRate);
        return text(html, win, o, false);
    }

    public VisdomResponse audio(Tensor waveform, int sampleRate, String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        return audio(tensorToFloat(waveform), sampleRate, win, opts);
    }

    // =========================================================================
    // Dual-axis lines helper
    // =========================================================================

    public VisdomResponse dualAxisLines(double[] X, double[] Y1, double[] Y2,
                                        String win, Map<String, Object> opts)
            throws IOException, InterruptedException {
        Objects.requireNonNull(X, "X");
        Objects.requireNonNull(Y1, "Y1");
        Objects.requireNonNull(Y2, "Y2");
        Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
        Map<String, Object> t1 = new LinkedHashMap<>();
        t1.put("x", X); t1.put("y", Y1);
        t1.put("name", o.getOrDefault("name_y1", "Y1"));
        t1.put("type", "scatter");
        Map<String, Object> t2 = new LinkedHashMap<>();
        t2.put("x", X); t2.put("y", Y2);
        t2.put("yaxis", "y2");
        t2.put("name", o.getOrDefault("name_y2", "Y2"));
        t2.put("type", "scatter");

        Map<String, Object> layout = opts2layout(o, false);
        Map<String, Object> yaxis2 = new LinkedHashMap<>();
        yaxis2.put("title", o.getOrDefault("name_y2", "Y2"));
        yaxis2.put("overlaying", "y");
        yaxis2.put("side", o.getOrDefault("side", "right"));
        layout.put("yaxis2", yaxis2);

        Map<String, Object> msg = base(win);
        msg.put("data", List.of(t1, t2));
        msg.put("layout", layout);
        msg.put("opts", cleanOpts(o));
        return send(msg, "events");
    }

    // =========================================================================
    // Low-level send
    // =========================================================================

    public VisdomResponse sendRaw(Map<String, Object> payload, String endpoint)
            throws IOException, InterruptedException {
        return send(payload, endpoint == null ? "events" : endpoint);
    }

    public CompletableFuture<VisdomResponse> sendAsync(Map<String, Object> payload, String endpoint) {
        try {
            HttpRequest req = buildRequest(payload, endpoint);
            return httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                    .thenApply(r -> new VisdomResponse(r.statusCode(), r.body(), endpoint));
        } catch (IOException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    private VisdomResponse send(Map<String, Object> payload, String endpoint)
            throws IOException, InterruptedException {
        HttpRequest req = buildRequest(payload, endpoint);
        try {
            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            VisdomResponse vr = new VisdomResponse(resp.statusCode(), resp.body(), endpoint);
            if (raiseExceptions && !vr.ok()) {
                throw new IOException("Visdom " + endpoint + " failed: HTTP " + vr.statusCode()
                        + " body=" + vr.body());
            }
            return vr;
        } catch (IOException e) {
            if (raiseExceptions) throw e;
            return new VisdomResponse(0, "ERROR: " + e.getMessage(), endpoint);
        }
    }

    private HttpRequest buildRequest(Map<String, Object> payload, String endpoint) throws IOException {
        if (!payload.containsKey("eid") || payload.get("eid") == null) {
            payload.put("eid", env);
        }
        if (payload.containsKey("win") && payload.get("win") == null) {
            payload.put("win", "window_" + UUID.randomUUID().toString().replace("-", "").substring(0, 8));
        }
        String json = Json.encode(payload);
        return HttpRequest.newBuilder(URI.create(baseUrl + "/" + endpoint))
                .timeout(requestTimeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                .build();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private Map<String, Object> base(String win) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("win", win);
        m.put("eid", env);
        return m;
    }

    static Map<String, Object> opts2layout(Map<String, Object> opts, boolean is3d) {
        Map<String, Object> layout = new LinkedHashMap<>();
        boolean showLegend = opts.containsKey("legend")
                || Boolean.TRUE.equals(opts.get("showlegend"));
        layout.put("showlegend", showLegend);
        if (opts.get("title") != null) layout.put("title", opts.get("title"));
        Map<String, Object> margin = new LinkedHashMap<>();
        margin.put("l", opts.getOrDefault("marginleft", is3d ? 0 : 60));
        margin.put("r", opts.getOrDefault("marginright", 60));
        margin.put("t", opts.getOrDefault("margintop", is3d ? 20 : 60));
        margin.put("b", opts.getOrDefault("marginbottom", is3d ? 0 : 60));
        layout.put("margin", margin);
        if (is3d) {
            Map<String, Object> scene = new LinkedHashMap<>();
            scene.put("xaxis", axis3d("x", opts));
            scene.put("yaxis", axis3d("y", opts));
            scene.put("zaxis", axis3d("z", opts));
            layout.put("scene", scene);
        } else {
            layout.put("xaxis", axis2d("x", opts));
            layout.put("yaxis", axis2d("y", opts));
        }
        if (Boolean.TRUE.equals(opts.get("stacked"))) layout.put("barmode", "stack");
        if (opts.get("width") instanceof Number) layout.put("width", opts.get("width"));
        if (opts.get("height") instanceof Number) layout.put("height", opts.get("height"));
        return scrub(layout);
    }

    private static Map<String, Object> axis2d(String ax, Map<String, Object> opts) {
        Map<String, Object> a = new LinkedHashMap<>();
        Object title = opts.get(ax + "label");
        if (title == null) title = opts.get(ax + "title");
        if (title != null) a.put("title", title);
        if (opts.get(ax + "tickvals") != null) a.put("tickvals", opts.get(ax + "tickvals"));
        if (opts.get(ax + "ticktext") != null) a.put("ticktext", opts.get(ax + "ticktext"));
        if (opts.get(ax + "type") != null) a.put("type", opts.get(ax + "type"));
        return a;
    }

    private static Map<String, Object> axis3d(String ax, Map<String, Object> opts) {
        return axis2d(ax, opts);
    }

    static Map<String, Object> cleanOpts(Map<String, Object> opts) {
        if (opts == null) return new LinkedHashMap<>();
        Map<String, Object> o = new LinkedHashMap<>(opts);
        o.entrySet().removeIf(e -> e.getValue() == null);
        return o;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> scrub(Map<String, Object> m) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : m.entrySet()) {
            Object v = e.getValue();
            if (v == null) continue;
            if (v instanceof Map) v = scrub((Map<String, Object>) v);
            if (v instanceof Map && ((Map<?, ?>) v).isEmpty()) continue;
            out.put(e.getKey(), v);
        }
        return out;
    }

    static double[] linspace(double a, double b, int n) {
        double[] x = new double[n];
        if (n == 1) { x[0] = a; return x; }
        for (int i = 0; i < n; i++) x[i] = a + (b - a) * i / (n - 1.0);
        return x;
    }

    static String escHtml(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\"", "&quot;");
    }

    /** Tiny PCM16 mono WAV encoder (no deps). */
    static byte[] encodeWav(float[] mono, int sampleRate) {
        // normalize
        float max = 1e-8f;
        for (float v : mono) {
            float a = Math.abs(v);
            if (a > max) max = a;
        }
        int dataLen = mono.length * 2;
        ByteArrayOutputStream bos = new ByteArrayOutputStream(44 + dataLen);
        try {
            // RIFF header
            bos.write("RIFF".getBytes(StandardCharsets.US_ASCII));
            writeIntLE(bos, 36 + dataLen);
            bos.write("WAVE".getBytes(StandardCharsets.US_ASCII));
            bos.write("fmt ".getBytes(StandardCharsets.US_ASCII));
            writeIntLE(bos, 16);           // PCM chunk size
            writeShortLE(bos, (short) 1);  // audio format = PCM
            writeShortLE(bos, (short) 1);  // mono
            writeIntLE(bos, sampleRate);
            writeIntLE(bos, sampleRate * 2); // byte rate
            writeShortLE(bos, (short) 2);  // block align
            writeShortLE(bos, (short) 16); // bits
            bos.write("data".getBytes(StandardCharsets.US_ASCII));
            writeIntLE(bos, dataLen);
            for (float v : mono) {
                int s = Math.round(v / max * 32767f);
                if (s > 32767) s = 32767;
                if (s < -32768) s = -32768;
                writeShortLE(bos, (short) s);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return bos.toByteArray();
    }

    private static void writeIntLE(ByteArrayOutputStream bos, int v) {
        bos.write(v & 0xff);
        bos.write((v >> 8) & 0xff);
        bos.write((v >> 16) & 0xff);
        bos.write((v >> 24) & 0xff);
    }

    private static void writeShortLE(ByteArrayOutputStream bos, short v) {
        bos.write(v & 0xff);
        bos.write((v >> 8) & 0xff);
    }

    static float[] tensorToFloat(Tensor t) {
        if (t == null || !t.defined()) return new float[0];
        // Match SummaryWriter.toFloatArray: contiguous → cpu → float → flatten → data_ptr_float
        Tensor c = t.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.kFloat()).flatten();
        long n = c.numel();
        float[] data = new float[(int) Math.min(n, Integer.MAX_VALUE)];
        org.bytedeco.javacpp.FloatPointer p = c.data_ptr_float();
        for (int i = 0; i < data.length; i++) data[i] = p.get(i);
        return data;
    }

    static double[][] tensorToMatrix(Tensor t) {
        if (t.dim() != 2) throw new IllegalArgumentException("heatmap tensor must be 2D");
        int rows = (int) t.size(0);
        int cols = (int) t.size(1);
        float[] flat = tensorToFloat(t);
        double[][] m = new double[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r][c] = flat[r * cols + c];
        return m;
    }

    /** Build an opts map quickly: {@code opts("title","Loss","xlabel","step")}. */
    public static Map<String, Object> opts(Object... kv) {
        if (kv.length % 2 != 0) throw new IllegalArgumentException("opts requires even #args");
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put(String.valueOf(kv[i]), kv[i + 1]);
        }
        return m;
    }

    @Override
    public void close() {
        if (ownExecutor) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(5, TimeUnit.SECONDS)) executor.shutdownNow();
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    // =========================================================================
    // Builder
    // =========================================================================

    public static final class Builder {
        private String host = "localhost";
        private int port = 8097;
        private String env = "main";
        private String basePath = "";
        private boolean useHttps = false;
        private Duration connectTimeout = Duration.ofSeconds(5);
        private Duration requestTimeout = Duration.ofSeconds(30);
        private int executorThreads = 4;
        private ExecutorService executor;
        private boolean raiseExceptions = true;

        public Builder host(String host) { this.host = Objects.requireNonNull(host); return this; }
        public Builder port(int port) { this.port = port; return this; }
        public Builder env(String env) { this.env = Objects.requireNonNull(env); return this; }
        public Builder basePath(String basePath) { this.basePath = basePath; return this; }
        public Builder useHttps(boolean v) { this.useHttps = v; return this; }
        public Builder connectTimeout(Duration d) { this.connectTimeout = d; return this; }
        public Builder requestTimeout(Duration d) { this.requestTimeout = d; return this; }
        public Builder executorThreads(int n) { this.executorThreads = n; return this; }
        public Builder executor(ExecutorService ex) { this.executor = ex; return this; }
        public Builder raiseExceptions(boolean v) { this.raiseExceptions = v; return this; }

        public VisdomClient build() { return new VisdomClient(this); }
    }
}
