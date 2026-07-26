package org.bytedeco.pytorch.utils.wandb;

import org.bytedeco.pytorch.utils.json.Json;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

/**
 * Embedded offline WandB-compatible server.
 *
 * <p>Exposes:
 * <ul>
 *   <li>{@code POST /api/runs} — create run</li>
 *   <li>{@code POST /api/runs/{id}} — finish / update</li>
 *   <li>{@code POST /api/metrics|charts|tables|images|text|audio|summary}</li>
 *   <li>{@code GET  /} — HTML dashboard listing runs</li>
 *   <li>{@code GET  /runs/{entity}/{project}/{id}} — per-run dashboard with
 *       Plotly charts (line metrics, heatmaps, images, tables)</li>
 *   <li>{@code GET  /api/runs/{id}/export} — full JSON export of a run</li>
 * </ul>
 *
 * <p>Start with {@link #start(int)} (port {@code 0} = ephemeral). Open
 * {@link #uiUrl()} in a browser to verify.
 */
public final class WandbLocalServer implements AutoCloseable {

    private final HttpServer server;
    private final int port;
    private final Map<String, RunState> runs = new ConcurrentHashMap<>();

    private WandbLocalServer(HttpServer server, int port) {
        this.server = server;
        this.port = port;
    }

    /** Bind {@code 127.0.0.1:port}. Pass {@code 0} for an ephemeral free port. */
    public static WandbLocalServer start(int port) throws IOException {
        HttpServer hs = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        WandbLocalServer local = new WandbLocalServer(hs, hs.getAddress().getPort());
        local.installRoutes();
        hs.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "wandb-local");
            t.setDaemon(true);
            return t;
        }));
        hs.start();
        return local;
    }

    public int port() { return port; }
    public String uiUrl() { return "http://127.0.0.1:" + port; }
    public String apiBase() { return uiUrl() + "/api/"; }
    public Map<String, RunState> runs() { return runs; }

    // =========================================================================
    // Routes
    // =========================================================================

    private void installRoutes() {
        server.createContext("/api/runs", this::handleRuns);
        server.createContext("/api/metrics", ex -> handleIngest(ex, "metrics"));
        server.createContext("/api/charts", ex -> handleIngest(ex, "charts"));
        server.createContext("/api/tables", ex -> handleIngest(ex, "tables"));
        server.createContext("/api/images", ex -> handleIngest(ex, "images"));
        server.createContext("/api/text", ex -> handleIngest(ex, "text"));
        server.createContext("/api/audio", ex -> handleIngest(ex, "audio"));
        server.createContext("/api/summary", ex -> handleIngest(ex, "summary"));
        server.createContext("/", this::handleUi);
    }

    private void handleRuns(HttpExchange ex) throws IOException {
        String method = ex.getRequestMethod();
        String path = ex.getRequestURI().getPath(); // /api/runs or /api/runs/{id}[/export]
        try {
            if ("POST".equalsIgnoreCase(method) && path.equals("/api/runs")) {
                Map<String, Object> body = readJson(ex);
                String id = UUID.randomUUID().toString().replace("-", "").substring(0, 8);
                RunState run = new RunState(id,
                        str(body.get("name"), "run"),
                        str(body.get("entity"), "local"),
                        str(body.get("project"), "pytorch"));
                run.config = body.get("config") instanceof Map
                        ? (Map<String, Object>) body.get("config") : Map.of();
                runs.put(id, run);
                Map<String, Object> resp = new LinkedHashMap<>();
                resp.put("id", id);
                resp.put("name", run.name);
                writeJson(ex, 200, resp);
                return;
            }
            if (path.startsWith("/api/runs/")) {
                String rest = path.substring("/api/runs/".length());
                boolean export = rest.endsWith("/export");
                String id = export ? rest.substring(0, rest.length() - "/export".length()) : rest;
                // strip trailing slash
                if (id.endsWith("/")) id = id.substring(0, id.length() - 1);
                RunState run = runs.get(id);
                if (run == null) { writeText(ex, 404, "run not found: " + id); return; }
                if ("POST".equalsIgnoreCase(method)) {
                    Map<String, Object> body = readJson(ex);
                    if ("finished".equals(String.valueOf(body.get("status")))) {
                        run.status = "finished";
                        run.finishedAt = str(body.get("finished_at"), "");
                    }
                    writeJson(ex, 200, Map.of("id", id, "status", run.status));
                    return;
                }
                if ("GET".equalsIgnoreCase(method)) {
                    writeJson(ex, 200, export ? run.toExport() : run.toSummary());
                    return;
                }
            }
            writeText(ex, 405, "method not allowed");
        } catch (Exception e) {
            writeText(ex, 500, e.getMessage() == null ? e.toString() : e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private void handleIngest(HttpExchange ex, String kind) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            writeText(ex, 405, "POST only"); return;
        }
        try {
            Map<String, Object> body = readJson(ex);
            String runId = str(body.get("run_id"), "");
            RunState run = runs.get(runId);
            if (run == null) { writeText(ex, 404, "unknown run_id " + runId); return; }
            switch (kind) {
                case "metrics" -> run.metrics.add(body);
                case "charts" -> run.charts.add(body);
                case "tables" -> run.tables.add(body);
                case "images" -> run.images.add(body);
                case "text" -> run.texts.add(body);
                case "audio" -> run.audios.add(body);
                case "summary" -> {
                    Object s = body.get("summary");
                    if (s instanceof Map) run.summary.putAll((Map<String, Object>) s);
                }
                default -> { }
            }
            writeJson(ex, 200, Map.of("ok", true, "kind", kind));
        } catch (Exception e) {
            writeText(ex, 500, e.getMessage() == null ? e.toString() : e.getMessage());
        }
    }

    private void handleUi(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
            writeText(ex, 405, "GET only"); return;
        }
        try {
            if (path.equals("/") || path.equals("/index.html")) {
                writeHtml(ex, renderIndex());
                return;
            }
            // /runs/{entity}/{project}/{id}
            if (path.startsWith("/runs/")) {
                String[] parts = path.substring(1).split("/");
                // runs, entity, project, id
                if (parts.length >= 4) {
                    String id = parts[3];
                    RunState run = runs.get(id);
                    if (run == null) { writeText(ex, 404, "run not found"); return; }
                    writeHtml(ex, renderRun(run));
                    return;
                }
            }
            // static image fetch: /img/{runId}/{index}
            if (path.startsWith("/img/")) {
                String[] parts = path.substring(5).split("/");
                if (parts.length >= 2) {
                    RunState run = runs.get(parts[0]);
                    int idx = Integer.parseInt(parts[1]);
                    if (run != null && idx >= 0 && idx < run.images.size()) {
                        Object b = run.images.get(idx).get("bytes");
                        byte[] png;
                        if (b instanceof String s) png = Base64.getDecoder().decode(s);
                        else if (b instanceof byte[] raw) png = raw;
                        else { writeText(ex, 404, "no image bytes"); return; }
                        Headers h = ex.getResponseHeaders();
                        h.set("Content-Type", "image/png");
                        ex.sendResponseHeaders(200, png.length);
                        try (OutputStream os = ex.getResponseBody()) { os.write(png); }
                        return;
                    }
                }
            }
            writeText(ex, 404, "not found: " + path);
        } catch (Exception e) {
            writeText(ex, 500, e.getMessage() == null ? e.toString() : e.getMessage());
        }
    }

    // =========================================================================
    // HTML dashboards (Plotly CDN)
    // =========================================================================

    private String renderIndex() {
        StringBuilder sb = new StringBuilder();
        sb.append("<!doctype html><html><head><meta charset='utf-8'>");
        sb.append("<title>WandB Local</title>");
        sb.append(CSS);
        sb.append("</head><body><div class='wrap'>");
        sb.append("<h1>WandB Local Dashboard</h1>");
        sb.append("<p class='muted'>Offline server · ").append(esc(uiUrl())).append("</p>");
        if (runs.isEmpty()) {
            sb.append("<p>No runs yet. Start a benchmark to populate.</p>");
        } else {
            sb.append("<table><thead><tr><th>ID</th><th>Name</th><th>Project</th>");
            sb.append("<th>Status</th><th>Metrics</th><th>Charts</th><th>Images</th><th></th></tr></thead><tbody>");
            for (RunState r : runs.values()) {
                String href = "/runs/" + enc(r.entity) + "/" + enc(r.project) + "/" + enc(r.id);
                sb.append("<tr>");
                sb.append("<td><code>").append(esc(r.id)).append("</code></td>");
                sb.append("<td>").append(esc(r.name)).append("</td>");
                sb.append("<td>").append(esc(r.entity)).append('/').append(esc(r.project)).append("</td>");
                sb.append("<td>").append(esc(r.status)).append("</td>");
                sb.append("<td>").append(r.metrics.size()).append("</td>");
                sb.append("<td>").append(r.charts.size()).append("</td>");
                sb.append("<td>").append(r.images.size()).append("</td>");
                sb.append("<td><a href='").append(href).append("'>open</a></td>");
                sb.append("</tr>");
            }
            sb.append("</tbody></table>");
        }
        sb.append("</div></body></html>");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private String renderRun(RunState run) {
        StringBuilder sb = new StringBuilder();
        sb.append("<!doctype html><html><head><meta charset='utf-8'>");
        sb.append("<title>").append(esc(run.name)).append(" · WandB Local</title>");
        sb.append("<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>");
        sb.append(CSS);
        sb.append("</head><body><div class='wrap'>");
        sb.append("<p><a href='/'>← all runs</a></p>");
        sb.append("<h1>").append(esc(run.name)).append("</h1>");
        sb.append("<p class='muted'><code>").append(esc(run.id)).append("</code> · ");
        sb.append(esc(run.entity)).append('/').append(esc(run.project));
        sb.append(" · status=<b>").append(esc(run.status)).append("</b></p>");

        // ---- metrics as multi-series line chart ----
        // Collect metric keys → (step,value) lists
        Map<String, List<double[]>> series = new LinkedHashMap<>();
        for (Map<String, Object> m : run.metrics) {
            long step = toLong(m.get("step"), 0);
            Object metrics = m.get("metrics");
            if (!(metrics instanceof Map)) continue;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) metrics).entrySet()) {
                String key = String.valueOf(e.getKey());
                double val = toDouble(e.getValue(), Double.NaN);
                series.computeIfAbsent(key, k -> new ArrayList<>()).add(new double[]{step, val});
            }
        }
        if (!series.isEmpty()) {
            sb.append("<h2>Metrics</h2><div id='metrics' class='plot'></div>");
            sb.append("<script>(function(){\n");
            sb.append("var traces=[];\n");
            for (Map.Entry<String, List<double[]>> e : series.entrySet()) {
                sb.append("traces.push({name:").append(jsStr(e.getKey())).append(",x:[");
                for (int i = 0; i < e.getValue().size(); i++) {
                    if (i > 0) sb.append(',');
                    sb.append(e.getValue().get(i)[0]);
                }
                sb.append("],y:[");
                for (int i = 0; i < e.getValue().size(); i++) {
                    if (i > 0) sb.append(',');
                    sb.append(e.getValue().get(i)[1]);
                }
                sb.append("],mode:'lines+markers',type:'scatter'});\n");
            }
            sb.append("Plotly.newPlot('metrics',traces,{margin:{t:30},xaxis:{title:'step'},yaxis:{title:'value'},height:360});\n");
            sb.append("})();</script>\n");
        }

        // ---- charts (heatmap / scatter / histogram …) ----
        int ci = 0;
        for (Map<String, Object> c : run.charts) {
            String type = str(c.get("type"), "line");
            String name = str(c.get("name"), "chart-" + ci);
            String divId = "chart_" + ci;
            sb.append("<h2>").append(esc(name)).append(" <span class='muted'>(").append(esc(type)).append(")</span></h2>");
            sb.append("<div id='").append(divId).append("' class='plot'></div>");
            sb.append("<script>(function(){\n");
            if ("heatmap".equals(type) && c.get("matrix") instanceof List) {
                sb.append("var z=").append(Json.encode(c.get("matrix"))).append(";\n");
                sb.append("Plotly.newPlot('").append(divId).append("',[{z:z,type:'heatmap',colorscale:'Viridis'}],");
                sb.append("{margin:{t:30},height:420,title:").append(jsStr(name)).append("});\n");
            } else if ("scatter".equals(type) && c.get("points") instanceof List) {
                List<?> pts = (List<?>) c.get("points");
                sb.append("var xs=[],ys=[];\n");
                for (Object p : pts) {
                    if (p instanceof List<?> xy && xy.size() >= 2) {
                        sb.append("xs.push(").append(xy.get(0)).append(");ys.push(").append(xy.get(1)).append(");\n");
                    }
                }
                sb.append("Plotly.newPlot('").append(divId).append("',[{x:xs,y:ys,mode:'markers',type:'scatter'}],");
                sb.append("{margin:{t:30},height:360,title:").append(jsStr(name)).append("});\n");
            } else if ("histogram".equals(type) && c.get("values") instanceof List) {
                sb.append("var v=").append(Json.encode(c.get("values"))).append(";\n");
                sb.append("Plotly.newPlot('").append(divId).append("',[{x:v,type:'histogram',nbinsx:")
                        .append(toLong(c.get("bins"), 30)).append("}],");
                sb.append("{margin:{t:30},height:360,title:").append(jsStr(name)).append("});\n");
            } else if (c.get("series") instanceof List) {
                sb.append("var series=").append(Json.encode(c.get("series"))).append(";\n");
                sb.append("var legends=").append(Json.encode(c.getOrDefault("legend", List.of()))).append(";\n");
                sb.append("var traces=series.map(function(s,i){return {y:s,name:legends[i]||('s'+i),type:'scatter',mode:'lines'};});\n");
                sb.append("Plotly.newPlot('").append(divId).append("',traces,{margin:{t:30},height:360});\n");
            } else {
                sb.append("document.getElementById('").append(divId)
                        .append("').innerText='(unsupported chart payload)';\n");
            }
            sb.append("})();</script>\n");
            ci++;
        }

        // ---- images ----
        if (!run.images.isEmpty()) {
            sb.append("<h2>Images</h2><div class='imgs'>");
            for (int i = 0; i < run.images.size(); i++) {
                Map<String, Object> im = run.images.get(i);
                String name = str(im.get("name"), "img" + i);
                long step = toLong(im.get("step"), 0);
                sb.append("<figure><img src='/img/").append(enc(run.id)).append('/').append(i)
                        .append("' alt='").append(esc(name)).append("'/>");
                sb.append("<figcaption>").append(esc(name)).append(" @ step ").append(step)
                        .append("</figcaption></figure>");
            }
            sb.append("</div>");
        }

        // ---- tables ----
        for (Map<String, Object> t : run.tables) {
            String name = str(t.get("name"), "table");
            sb.append("<h2>").append(esc(name)).append("</h2><table>");
            Object cols = t.get("columns");
            if (cols instanceof List<?> cl) {
                sb.append("<thead><tr>");
                for (Object c : cl) sb.append("<th>").append(esc(String.valueOf(c))).append("</th>");
                sb.append("</tr></thead>");
            }
            sb.append("<tbody>");
            Object rows = t.get("rows");
            if (rows instanceof List<?> rl) {
                for (Object row : rl) {
                    sb.append("<tr>");
                    if (row instanceof List<?> cells) {
                        for (Object cell : cells)
                            sb.append("<td>").append(esc(String.valueOf(cell))).append("</td>");
                    } else if (row instanceof Object[] arr) {
                        for (Object cell : arr)
                            sb.append("<td>").append(esc(String.valueOf(cell))).append("</td>");
                    }
                    sb.append("</tr>");
                }
            }
            sb.append("</tbody></table>");
        }

        // ---- texts ----
        for (Map<String, Object> t : run.texts) {
            sb.append("<h2>").append(esc(str(t.get("name"), "text"))).append("</h2>");
            sb.append("<pre>").append(esc(str(t.get("text"), ""))).append("</pre>");
        }

        // ---- summary ----
        if (!run.summary.isEmpty()) {
            sb.append("<h2>Summary</h2><pre>").append(esc(Json.encode(run.summary))).append("</pre>");
        }

        sb.append("<p class='muted'>export: <a href='/api/runs/").append(enc(run.id))
                .append("/export'>/api/runs/").append(esc(run.id)).append("/export</a></p>");
        sb.append("</div></body></html>");
        return sb.toString();
    }

    private static final String CSS = """
            <style>
              :root { color-scheme: light dark; }
              body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
                     margin: 0; background: #0b1020; color: #e8ecf7; }
              .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
              h1 { margin: 0 0 8px; font-size: 1.6rem; }
              h2 { margin-top: 28px; font-size: 1.15rem; border-bottom: 1px solid #243049; padding-bottom: 6px; }
              a { color: #7db4ff; }
              .muted { color: #9aa6c2; }
              table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92rem; }
              th, td { border: 1px solid #243049; padding: 6px 10px; text-align: left; }
              th { background: #151c31; }
              .plot { background: #111827; border-radius: 8px; padding: 4px; }
              .imgs { display: flex; flex-wrap: wrap; gap: 12px; }
              .imgs figure { margin: 0; background: #151c31; padding: 8px; border-radius: 8px; }
              .imgs img { max-width: 240px; display: block; border-radius: 4px; }
              figcaption { font-size: 0.8rem; color: #9aa6c2; margin-top: 4px; }
              pre { background: #151c31; padding: 12px; border-radius: 8px; overflow: auto; }
              code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
            </style>
            """;

    // =========================================================================
    // IO helpers
    // =========================================================================

    private static Map<String, Object> readJson(HttpExchange ex) throws IOException {
        byte[] raw = readAll(ex.getRequestBody());
        if (raw.length == 0) return new LinkedHashMap<>();
        return Json.decodeObject(new String(raw, StandardCharsets.UTF_8));
    }

    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) >= 0) bos.write(buf, 0, n);
        return bos.toByteArray();
    }

    private static void writeJson(HttpExchange ex, int code, Object body) throws IOException {
        byte[] raw = Json.encode(body).getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "application/json; charset=utf-8");
        h.set("Access-Control-Allow-Origin", "*");
        ex.sendResponseHeaders(code, raw.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(raw); }
    }

    private static void writeText(HttpExchange ex, int code, String body) throws IOException {
        byte[] raw = body.getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "text/plain; charset=utf-8");
        ex.sendResponseHeaders(code, raw.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(raw); }
    }

    private static void writeHtml(HttpExchange ex, String html) throws IOException {
        byte[] raw = html.getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, raw.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(raw); }
    }

    private static String str(Object o, String dft) {
        return o == null ? dft : String.valueOf(o);
    }

    private static long toLong(Object o, long dft) {
        if (o instanceof Number n) return n.longValue();
        if (o == null) return dft;
        try { return Long.parseLong(String.valueOf(o)); } catch (Exception e) { return dft; }
    }

    private static double toDouble(Object o, double dft) {
        if (o instanceof Number n) return n.doubleValue();
        if (o == null) return dft;
        try { return Double.parseDouble(String.valueOf(o)); } catch (Exception e) { return dft; }
    }

    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\"", "&quot;");
    }

    private static String enc(String s) {
        return java.net.URLEncoder.encode(s, StandardCharsets.UTF_8);
    }

    private static String jsStr(String s) {
        return "\"" + Json.escape(s) + "\"";
    }

    @Override
    public void close() {
        server.stop(0);
    }

    // =========================================================================
    // Run state
    // =========================================================================

    public static final class RunState {
        public final String id, name, entity, project;
        public String status = "running";
        public String finishedAt = "";
        public Map<String, Object> config = Map.of();
        public final List<Map<String, Object>> metrics = new ArrayList<>();
        public final List<Map<String, Object>> charts = new ArrayList<>();
        public final List<Map<String, Object>> tables = new ArrayList<>();
        public final List<Map<String, Object>> images = new ArrayList<>();
        public final List<Map<String, Object>> texts = new ArrayList<>();
        public final List<Map<String, Object>> audios = new ArrayList<>();
        public final Map<String, Object> summary = new LinkedHashMap<>();

        RunState(String id, String name, String entity, String project) {
            this.id = id; this.name = name; this.entity = entity; this.project = project;
        }

        Map<String, Object> toSummary() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id); m.put("name", name); m.put("entity", entity);
            m.put("project", project); m.put("status", status);
            m.put("n_metrics", metrics.size());
            m.put("n_charts", charts.size());
            m.put("n_images", images.size());
            m.put("n_tables", tables.size());
            return m;
        }

        Map<String, Object> toExport() {
            Map<String, Object> m = toSummary();
            m.put("config", config);
            m.put("metrics", metrics);
            m.put("charts", charts);
            m.put("tables", tables);
            // images: drop raw bytes in export listing size only
            List<Map<String, Object>> imgs = new ArrayList<>();
            for (Map<String, Object> im : images) {
                Map<String, Object> c = new LinkedHashMap<>(im);
                Object b = c.get("bytes");
                if (b instanceof String s) c.put("bytes_len", s.length());
                else if (b instanceof byte[] raw) c.put("bytes_len", raw.length);
                c.remove("bytes");
                imgs.add(c);
            }
            m.put("images", imgs);
            m.put("texts", texts);
            m.put("audios", audios);
            m.put("summary", summary);
            m.put("finished_at", finishedAt);
            return m;
        }
    }
}
