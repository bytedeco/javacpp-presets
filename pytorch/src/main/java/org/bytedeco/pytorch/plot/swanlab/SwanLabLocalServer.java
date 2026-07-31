package org.bytedeco.pytorch.plot.swanlab;

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
 * Embedded offline SwanLab-compatible server with a Plotly HTML dashboard.
 *
 * <p>Endpoints mirror the subset used by {@link SwanLabClient}:
 * <ul>
 *   <li>{@code POST /api/v1/experiments} — create</li>
 *   <li>{@code POST /api/v1/experiments/{id}} — finish</li>
 *   <li>{@code POST /api/v1/logs}</li>
 *   <li>{@code POST /api/v1/media/{charts,images,text,audio,tables}}</li>
 *   <li>{@code POST /api/v1/summary}</li>
 *   <li>{@code GET  /} and {@code /experiments/.../{id}} — dashboards</li>
 * </ul>
 */
public final class SwanLabLocalServer implements AutoCloseable {

    private final HttpServer server;
    private final int port;
    private final Map<String, ExpState> experiments = new ConcurrentHashMap<>();

    private SwanLabLocalServer(HttpServer server, int port) {
        this.server = server;
        this.port = port;
    }

    public static SwanLabLocalServer start(int port) throws IOException {
        HttpServer hs = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        SwanLabLocalServer local = new SwanLabLocalServer(hs, hs.getAddress().getPort());
        local.installRoutes();
        hs.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "swanlab-local");
            t.setDaemon(true);
            return t;
        }));
        hs.start();
        return local;
    }

    public int port() { return port; }
    public String uiUrl() { return "http://127.0.0.1:" + port; }
    public String apiBase() { return uiUrl() + "/api/v1/"; }
    public Map<String, ExpState> experiments() { return experiments; }

    private void installRoutes() {
        server.createContext("/api/v1/experiments", this::handleExperiments);
        server.createContext("/api/v1/logs", ex -> handleIngest(ex, "logs"));
        server.createContext("/api/v1/media/charts", ex -> handleIngest(ex, "charts"));
        server.createContext("/api/v1/media/images", ex -> handleIngest(ex, "images"));
        server.createContext("/api/v1/media/text", ex -> handleIngest(ex, "text"));
        server.createContext("/api/v1/media/audio", ex -> handleIngest(ex, "audio"));
        server.createContext("/api/v1/media/tables", ex -> handleIngest(ex, "tables"));
        server.createContext("/api/v1/summary", ex -> handleIngest(ex, "summary"));
        server.createContext("/", this::handleUi);
    }

    @SuppressWarnings("unchecked")
    private void handleExperiments(HttpExchange ex) throws IOException {
        String method = ex.getRequestMethod();
        String path = ex.getRequestURI().getPath();
        try {
            if ("POST".equalsIgnoreCase(method) && path.equals("/api/v1/experiments")) {
                Map<String, Object> body = readJson(ex);
                String id = UUID.randomUUID().toString().replace("-", "").substring(0, 8);
                ExpState exp = new ExpState(id,
                        str(body.get("experiment"), "exp"),
                        str(body.get("workspace"), "local"),
                        str(body.get("project"), "pytorch"));
                exp.config = body.get("config") instanceof Map
                        ? (Map<String, Object>) body.get("config") : Map.of();
                experiments.put(id, exp);
                writeJson(ex, 200, Map.of("id", id, "name", exp.name));
                return;
            }
            if (path.startsWith("/api/v1/experiments/")) {
                String rest = path.substring("/api/v1/experiments/".length());
                boolean export = rest.endsWith("/export");
                String id = export ? rest.substring(0, rest.length() - "/export".length()) : rest;
                if (id.endsWith("/")) id = id.substring(0, id.length() - 1);
                ExpState exp = experiments.get(id);
                if (exp == null) { writeText(ex, 404, "experiment not found: " + id); return; }
                if ("POST".equalsIgnoreCase(method)) {
                    Map<String, Object> body = readJson(ex);
                    if ("finished".equals(String.valueOf(body.get("status")))) {
                        exp.status = "finished";
                        exp.finishedAt = str(body.get("finished_at"), "");
                    }
                    writeJson(ex, 200, Map.of("id", id, "status", exp.status));
                    return;
                }
                if ("GET".equalsIgnoreCase(method)) {
                    writeJson(ex, 200, export ? exp.toExport() : exp.toSummary());
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
            String expId = str(body.get("experiment_id"),
                    str(body.get("run_id"), ""));
            ExpState exp = experiments.get(expId);
            if (exp == null) { writeText(ex, 404, "unknown experiment_id " + expId); return; }
            switch (kind) {
                case "logs" -> exp.logs.add(body);
                case "charts" -> exp.charts.add(body);
                case "images" -> exp.images.add(body);
                case "text" -> exp.texts.add(body);
                case "audio" -> exp.audios.add(body);
                case "tables" -> exp.tables.add(body);
                case "summary" -> {
                    Object s = body.get("summary");
                    if (s instanceof Map) exp.summary.putAll((Map<String, Object>) s);
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
            if (path.startsWith("/experiments/")) {
                String[] parts = path.substring(1).split("/");
                // experiments, workspace, project, id
                if (parts.length >= 4) {
                    ExpState exp = experiments.get(parts[3]);
                    if (exp == null) { writeText(ex, 404, "experiment not found"); return; }
                    writeHtml(ex, renderExp(exp));
                    return;
                }
            }
            if (path.startsWith("/img/")) {
                String[] parts = path.substring(5).split("/");
                if (parts.length >= 2) {
                    ExpState exp = experiments.get(parts[0]);
                    int idx = Integer.parseInt(parts[1]);
                    if (exp != null && idx >= 0 && idx < exp.images.size()) {
                        Object b = exp.images.get(idx).get("bytes");
                        byte[] png;
                        if (b instanceof String s) png = Base64.getDecoder().decode(s);
                        else if (b instanceof byte[] raw) png = raw;
                        else { writeText(ex, 404, "no image"); return; }
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

    private String renderIndex() {
        StringBuilder sb = new StringBuilder();
        sb.append("<!doctype html><html><head><meta charset='utf-8'>");
        sb.append("<title>SwanLab Local</title>").append(CSS).append("</head><body><div class='wrap'>");
        sb.append("<h1>SwanLab Local Dashboard</h1>");
        sb.append("<p class='muted'>Offline server · ").append(esc(uiUrl())).append("</p>");
        if (experiments.isEmpty()) {
            sb.append("<p>No experiments yet.</p>");
        } else {
            sb.append("<table><thead><tr><th>ID</th><th>Name</th><th>Project</th>");
            sb.append("<th>Status</th><th>Logs</th><th>Charts</th><th>Images</th><th></th></tr></thead><tbody>");
            for (ExpState e : experiments.values()) {
                String href = "/experiments/" + enc(e.workspace) + "/" + enc(e.project) + "/" + enc(e.id);
                sb.append("<tr><td><code>").append(esc(e.id)).append("</code></td>");
                sb.append("<td>").append(esc(e.name)).append("</td>");
                sb.append("<td>").append(esc(e.workspace)).append('/').append(esc(e.project)).append("</td>");
                sb.append("<td>").append(esc(e.status)).append("</td>");
                sb.append("<td>").append(e.logs.size()).append("</td>");
                sb.append("<td>").append(e.charts.size()).append("</td>");
                sb.append("<td>").append(e.images.size()).append("</td>");
                sb.append("<td><a href='").append(href).append("'>open</a></td></tr>");
            }
            sb.append("</tbody></table>");
        }
        sb.append("</div></body></html>");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private String renderExp(ExpState exp) {
        StringBuilder sb = new StringBuilder();
        sb.append("<!doctype html><html><head><meta charset='utf-8'>");
        sb.append("<title>").append(esc(exp.name)).append(" · SwanLab Local</title>");
        sb.append("<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>");
        sb.append(CSS).append("</head><body><div class='wrap'>");
        sb.append("<p><a href='/'>← all experiments</a></p>");
        sb.append("<h1>").append(esc(exp.name)).append("</h1>");
        sb.append("<p class='muted'><code>").append(esc(exp.id)).append("</code> · ");
        sb.append(esc(exp.workspace)).append('/').append(esc(exp.project));
        sb.append(" · status=<b>").append(esc(exp.status)).append("</b></p>");

        // metrics from logs
        Map<String, List<double[]>> series = new LinkedHashMap<>();
        for (Map<String, Object> m : exp.logs) {
            long step = toLong(m.get("step"), 0);
            Object metrics = m.get("metrics");
            if (!(metrics instanceof Map)) continue;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) metrics).entrySet()) {
                series.computeIfAbsent(String.valueOf(e.getKey()), k -> new ArrayList<>())
                        .add(new double[]{step, toDouble(e.getValue(), Double.NaN)});
            }
        }
        if (!series.isEmpty()) {
            sb.append("<h2>Metrics</h2><div id='metrics' class='plot'></div><script>(function(){\n");
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
            sb.append("Plotly.newPlot('metrics',traces,{margin:{t:30},xaxis:{title:'step'},height:360});\n");
            sb.append("})();</script>\n");
        }

        int ci = 0;
        for (Map<String, Object> c : exp.charts) {
            String type = str(c.get("type"), "line");
            String name = str(c.get("name"), "chart-" + ci);
            String divId = "chart_" + ci;
            sb.append("<h2>").append(esc(name)).append(" <span class='muted'>(")
                    .append(esc(type)).append(")</span></h2>");
            sb.append("<div id='").append(divId).append("' class='plot'></div><script>(function(){\n");
            if ("heatmap".equals(type) && c.get("matrix") instanceof List) {
                sb.append("var z=").append(Json.encode(c.get("matrix"))).append(";\n");
                sb.append("Plotly.newPlot('").append(divId)
                        .append("',[{z:z,type:'heatmap',colorscale:'Viridis'}],{margin:{t:30},height:420});\n");
            } else if ("scatter".equals(type) && c.get("points") instanceof List) {
                sb.append("var xs=[],ys=[];\n");
                for (Object p : (List<?>) c.get("points")) {
                    if (p instanceof List<?> xy && xy.size() >= 2) {
                        sb.append("xs.push(").append(xy.get(0)).append(");ys.push(")
                                .append(xy.get(1)).append(");\n");
                    }
                }
                sb.append("Plotly.newPlot('").append(divId)
                        .append("',[{x:xs,y:ys,mode:'markers',type:'scatter'}],{margin:{t:30},height:360});\n");
            } else if ("histogram".equals(type) && c.get("values") instanceof List) {
                sb.append("var v=").append(Json.encode(c.get("values"))).append(";\n");
                sb.append("Plotly.newPlot('").append(divId).append("',[{x:v,type:'histogram',nbinsx:")
                        .append(toLong(c.get("bins"), 30)).append("}],{margin:{t:30},height:360});\n");
            } else if ("bar".equals(type) && c.get("values") instanceof List) {
                sb.append("var v=").append(Json.encode(c.get("values"))).append(";\n");
                sb.append("var lab=").append(Json.encode(c.getOrDefault("labels", List.of()))).append(";\n");
                sb.append("Plotly.newPlot('").append(divId)
                        .append("',[{y:v,x:lab,type:'bar'}],{margin:{t:30},height:360});\n");
            } else if (c.get("series") instanceof List) {
                sb.append("var series=").append(Json.encode(c.get("series"))).append(";\n");
                sb.append("var legends=").append(Json.encode(c.getOrDefault("legend", List.of()))).append(";\n");
                sb.append("var traces=series.map(function(s,i){return {y:s,name:legends[i]||('s'+i),type:'scatter',mode:'lines'};});\n");
                sb.append("Plotly.newPlot('").append(divId).append("',traces,{margin:{t:30},height:360});\n");
            } else {
                sb.append("document.getElementById('").append(divId)
                        .append("').innerText='(unsupported chart)';\n");
            }
            sb.append("})();</script>\n");
            ci++;
        }

        if (!exp.images.isEmpty()) {
            sb.append("<h2>Images</h2><div class='imgs'>");
            for (int i = 0; i < exp.images.size(); i++) {
                Map<String, Object> im = exp.images.get(i);
                sb.append("<figure><img src='/img/").append(enc(exp.id)).append('/').append(i)
                        .append("' alt='").append(esc(str(im.get("name"), "img"))).append("'/>");
                sb.append("<figcaption>").append(esc(str(im.get("name"), "img")))
                        .append(" @ step ").append(toLong(im.get("step"), 0))
                        .append("</figcaption></figure>");
            }
            sb.append("</div>");
        }

        for (Map<String, Object> t : exp.tables) {
            sb.append("<h2>").append(esc(str(t.get("name"), "table"))).append("</h2><table>");
            if (t.get("columns") instanceof List<?> cl) {
                sb.append("<thead><tr>");
                for (Object c : cl) sb.append("<th>").append(esc(String.valueOf(c))).append("</th>");
                sb.append("</tr></thead>");
            }
            sb.append("<tbody>");
            if (t.get("rows") instanceof List<?> rl) {
                for (Object row : rl) {
                    sb.append("<tr>");
                    if (row instanceof List<?> cells)
                        for (Object cell : cells)
                            sb.append("<td>").append(esc(String.valueOf(cell))).append("</td>");
                    else if (row instanceof Object[] arr)
                        for (Object cell : arr)
                            sb.append("<td>").append(esc(String.valueOf(cell))).append("</td>");
                    sb.append("</tr>");
                }
            }
            sb.append("</tbody></table>");
        }

        for (Map<String, Object> t : exp.texts) {
            sb.append("<h2>").append(esc(str(t.get("name"), "text"))).append("</h2>");
            sb.append("<pre>").append(esc(str(t.get("text"), ""))).append("</pre>");
        }

        if (!exp.summary.isEmpty()) {
            sb.append("<h2>Summary</h2><pre>").append(esc(Json.encode(exp.summary))).append("</pre>");
        }

        sb.append("<p class='muted'>export: <a href='/api/v1/experiments/").append(enc(exp.id))
                .append("/export'>JSON</a></p>");
        sb.append("</div></body></html>");
        return sb.toString();
    }

    private static final String CSS = """
            <style>
              :root { color-scheme: light dark; }
              body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
                     margin: 0; background: #0a1628; color: #e7f0ff; }
              .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
              h1 { margin: 0 0 8px; font-size: 1.6rem; }
              h2 { margin-top: 28px; font-size: 1.15rem; border-bottom: 1px solid #1e3a5f; padding-bottom: 6px; }
              a { color: #5ec8ff; }
              .muted { color: #8fb0d0; }
              table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92rem; }
              th, td { border: 1px solid #1e3a5f; padding: 6px 10px; text-align: left; }
              th { background: #12233a; }
              .plot { background: #0d1b2e; border-radius: 8px; padding: 4px; }
              .imgs { display: flex; flex-wrap: wrap; gap: 12px; }
              .imgs figure { margin: 0; background: #12233a; padding: 8px; border-radius: 8px; }
              .imgs img { max-width: 240px; display: block; border-radius: 4px; }
              figcaption { font-size: 0.8rem; color: #8fb0d0; margin-top: 4px; }
              pre { background: #12233a; padding: 12px; border-radius: 8px; overflow: auto; }
              code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
            </style>
            """;

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

    private static String str(Object o, String dft) { return o == null ? dft : String.valueOf(o); }
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
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;");
    }
    private static String enc(String s) { return java.net.URLEncoder.encode(s, StandardCharsets.UTF_8); }
    private static String jsStr(String s) { return "\"" + Json.escape(s) + "\""; }

    @Override
    public void close() { server.stop(0); }

    public static final class ExpState {
        public final String id, name, workspace, project;
        public String status = "running";
        public String finishedAt = "";
        public Map<String, Object> config = Map.of();
        public final List<Map<String, Object>> logs = new ArrayList<>();
        public final List<Map<String, Object>> charts = new ArrayList<>();
        public final List<Map<String, Object>> images = new ArrayList<>();
        public final List<Map<String, Object>> texts = new ArrayList<>();
        public final List<Map<String, Object>> audios = new ArrayList<>();
        public final List<Map<String, Object>> tables = new ArrayList<>();
        public final Map<String, Object> summary = new LinkedHashMap<>();

        ExpState(String id, String name, String workspace, String project) {
            this.id = id; this.name = name; this.workspace = workspace; this.project = project;
        }

        Map<String, Object> toSummary() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id); m.put("name", name); m.put("workspace", workspace);
            m.put("project", project); m.put("status", status);
            m.put("n_logs", logs.size()); m.put("n_charts", charts.size());
            m.put("n_images", images.size()); m.put("n_tables", tables.size());
            return m;
        }

        Map<String, Object> toExport() {
            Map<String, Object> m = toSummary();
            m.put("config", config);
            m.put("logs", logs);
            m.put("charts", charts);
            m.put("tables", tables);
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
