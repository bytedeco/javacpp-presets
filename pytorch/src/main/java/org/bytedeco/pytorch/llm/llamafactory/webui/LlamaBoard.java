/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.llamafactory.webui;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.bytedeco.pytorch.llm.llamafactory.DefaultFinetuneJob;
import org.bytedeco.pytorch.llm.llamafactory.FactoryVersion;
import org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Embedded LLaMA-Factory training dashboard (Gradio-like port, zero npm).
 *
 * <p>Endpoints:
 * <ul>
 *   <li>{@code GET /} — HTML dashboard (Plotly CDN charts + log tail)</li>
 *   <li>{@code GET /api/runs} — list runs</li>
 *   <li>{@code GET /api/runs/{id}} — run snapshot</li>
 *   <li>{@code GET /api/runs/{id}/metrics} — time-series</li>
 *   <li>{@code GET /api/runs/{id}/config} — hyper-parameters</li>
 *   <li>{@code POST /api/runs} — create/start run from FactoryArgs JSON</li>
 *   <li>{@code POST /api/runs/{id}/stop} — cooperative cancel</li>
 * </ul>
 */
public final class LlamaBoard implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(LlamaBoard.class.getName());

    private final HttpServer server;
    private final int port;
    private final BoardState sharedState;
    private final FactoryArgs defaultArgs;
    private final ConcurrentHashMap<String, RunHandle> runs = new ConcurrentHashMap<>();
    private final ExecutorService workers;
    private final AtomicReference<FinetuneAdapter> attached = new AtomicReference<>();

    private LlamaBoard(HttpServer server, int port, BoardState state, FactoryArgs defaultArgs) {
        this.server = server;
        this.port = port;
        this.sharedState = state == null ? new BoardState() : state;
        this.defaultArgs = defaultArgs;
        this.workers = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "llamaboard-worker");
            t.setDaemon(true);
            return t;
        });
    }

    public static LlamaBoard start(int port) throws IOException {
        return start(port, null, null);
    }

    public static LlamaBoard start(int port, FactoryArgs defaultArgs) throws IOException {
        return start(port, defaultArgs, null);
    }

    public static LlamaBoard start(int port, FactoryArgs defaultArgs, BoardState state)
            throws IOException {
        int p = port > 0 ? port : 7860;
        HttpServer hs = HttpServer.create(new InetSocketAddress("0.0.0.0", p), 0);
        LlamaBoard board = new LlamaBoard(hs, hs.getAddress().getPort(), state, defaultArgs);
        board.installRoutes();
        hs.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "llamaboard-http");
            t.setDaemon(true);
            return t;
        }));
        hs.start();
        LOG.info("LlamaBoard listening on " + board.uiUrl());
        return board;
    }

    public int port() { return port; }
    public String uiUrl() { return "http://127.0.0.1:" + port; }
    public BoardState state() { return sharedState; }

    /** Attach an existing job so the dashboard mirrors its board state / stop. */
    public void attach(FinetuneAdapter job) {
        attached.set(job);
        if (job != null && job.board() != null) {
            // prefer job board for live metrics
        }
    }

    public FinetuneAdapter attached() {
        return attached.get();
    }

    private void installRoutes() {
        server.createContext("/", this::handleRoot);
        server.createContext("/api/runs", this::handleRuns);
        server.createContext("/api/health", ex -> writeJson(ex, 200, Map.of(
                "ok", true,
                "version", FactoryVersion.VERSION,
                "banner", FactoryVersion.BANNER)));
        server.createContext("/api/state", ex -> {
            BoardState s = effectiveState();
            writeJson(ex, 200, s.snapshot());
        });
    }

    private BoardState effectiveState() {
        FinetuneAdapter job = attached.get();
        if (job != null && job.board() != null) {
            return job.board();
        }
        return sharedState;
    }

    private void handleRoot(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
            writeText(ex, 405, "method not allowed");
            return;
        }
        String path = ex.getRequestURI().getPath();
        if (path == null || path.equals("/") || path.equals("/index.html")) {
            writeHtml(ex, 200, dashboardHtml());
            return;
        }
        writeText(ex, 404, "not found");
    }

    @SuppressWarnings("unchecked")
    private void handleRuns(HttpExchange ex) throws IOException {
        String method = ex.getRequestMethod();
        String path = ex.getRequestURI().getPath();
        try {
            if ("GET".equalsIgnoreCase(method) && path.equals("/api/runs")) {
                List<Map<String, Object>> list = new ArrayList<>();
                for (RunHandle h : runs.values()) {
                    list.add(h.summary());
                }
                // also surface attached job
                FinetuneAdapter job = attached.get();
                if (job != null) {
                    Map<String, Object> a = new LinkedHashMap<>();
                    a.put("id", "attached");
                    a.put("status", job.board() == null ? "unknown" : job.board().status().name());
                    a.put("global_step", job.globalStep());
                    a.put("metrics", job.lastMetrics());
                    list.add(a);
                }
                writeJson(ex, 200, Map.of("runs", list));
                return;
            }

            if (path.startsWith("/api/runs/")) {
                String rest = path.substring("/api/runs/".length());
                if (rest.endsWith("/")) rest = rest.substring(0, rest.length() - 1);
                String[] parts = rest.split("/");
                String id = parts[0];
                String action = parts.length > 1 ? parts[1] : "";

                if ("attached".equals(id)) {
                    FinetuneAdapter job = attached.get();
                    if (job == null) {
                        writeText(ex, 404, "no attached job");
                        return;
                    }
                    if ("stop".equals(action) && "POST".equalsIgnoreCase(method)) {
                        job.requestStop();
                        writeJson(ex, 200, Map.of("id", id, "stop_requested", true));
                        return;
                    }
                    if ("metrics".equals(action) && "GET".equalsIgnoreCase(method)) {
                        BoardState s = job.board();
                        writeJson(ex, 200, Map.of(
                                "metrics", job.lastMetrics(),
                                "loss_history", s == null ? List.of() : s.lossHistory(),
                                "global_step", job.globalStep()));
                        return;
                    }
                    if ("config".equals(action) && "GET".equalsIgnoreCase(method)) {
                        writeJson(ex, 200, job.args().toMap());
                        return;
                    }
                    if (action.isEmpty() && "GET".equalsIgnoreCase(method)) {
                        BoardState s = job.board();
                        Map<String, Object> snap = s == null ? new LinkedHashMap<>() : s.snapshot();
                        snap.put("id", "attached");
                        snap.put("global_step", job.globalStep());
                        snap.put("metrics", job.lastMetrics());
                        writeJson(ex, 200, snap);
                        return;
                    }
                }

                RunHandle h = runs.get(id);
                if (h == null) {
                    writeText(ex, 404, "run not found: " + id);
                    return;
                }
                if ("stop".equals(action) && "POST".equalsIgnoreCase(method)) {
                    h.requestStop();
                    writeJson(ex, 200, Map.of("id", id, "stop_requested", true));
                    return;
                }
                if ("metrics".equals(action) && "GET".equalsIgnoreCase(method)) {
                    writeJson(ex, 200, h.metricsPayload());
                    return;
                }
                if ("config".equals(action) && "GET".equalsIgnoreCase(method)) {
                    writeJson(ex, 200, h.args.toMap());
                    return;
                }
                if (action.isEmpty() && "GET".equalsIgnoreCase(method)) {
                    writeJson(ex, 200, h.snapshot());
                    return;
                }
            }

            if ("POST".equalsIgnoreCase(method) && path.equals("/api/runs")) {
                Map<String, Object> body = readJson(ex);
                FactoryArgs fa;
                if (body == null || body.isEmpty()) {
                    if (defaultArgs == null) {
                        writeText(ex, 400, "body required when no default FactoryArgs");
                        return;
                    }
                    fa = defaultArgs;
                } else {
                    fa = FactoryArgs.parse(body);
                }
                fa.validate();
                String id = UUID.randomUUID().toString().replace("-", "").substring(0, 10);
                BoardState st = new BoardState();
                RunHandle handle = new RunHandle(id, fa, st);
                runs.put(id, handle);
                handle.future = workers.submit(() -> {
                    st.setStatus(BoardState.Status.RUNNING);
                    st.setMessage("starting");
                    try (FinetuneAdapter job = new DefaultFinetuneJob(fa, null, st)) {
                        handle.job = job;
                        job.train();
                        handle.globalStep = job.globalStep();
                        handle.lastMetrics = job.lastMetrics();
                        if (!st.stopRequested()) {
                            st.setStatus(BoardState.Status.COMPLETED);
                        }
                    } catch (Throwable t) {
                        st.setStatus(BoardState.Status.FAILED);
                        st.setMessage(t.getMessage() == null ? t.toString() : t.getMessage());
                        st.log("[error] " + st.message());
                        LOG.log(Level.WARNING, "run " + id + " failed", t);
                    }
                });
                writeJson(ex, 200, Map.of("id", id, "status", "RUNNING", "ui", uiUrl()));
                return;
            }

            writeText(ex, 405, "method not allowed");
        } catch (Exception e) {
            writeText(ex, 500, e.getMessage() == null ? e.toString() : e.getMessage());
        }
    }

    private static String dashboardHtml() {
        return """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                  <meta charset="utf-8"/>
                  <title>LlamaBoard</title>
                  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
                  <style>
                    body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#0b1020;color:#e8eefc}
                    header{padding:16px 24px;border-bottom:1px solid #1e2a44;display:flex;gap:16px;align-items:center}
                    h1{font-size:18px;margin:0}
                    main{padding:16px 24px;display:grid;grid-template-columns:2fr 1fr;gap:16px}
                    .card{background:#121a2f;border:1px solid #1e2a44;border-radius:12px;padding:12px}
                    #loss{height:320px}
                    pre{background:#0b1020;border-radius:8px;padding:8px;max-height:280px;overflow:auto;font-size:12px}
                    button{background:#3b82f6;color:white;border:0;border-radius:8px;padding:8px 12px;cursor:pointer}
                    button.secondary{background:#334155}
                    .row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
                    .metric{background:#0b1020;border-radius:8px;padding:8px 10px;min-width:90px}
                    .metric b{display:block;font-size:16px}
                    .metric span{font-size:11px;opacity:.7}
                  </style>
                </head>
                <body>
                  <header>
                    <h1>LlamaBoard</h1>
                    <span id="status">IDLE</span>
                    <div style="flex:1"></div>
                    <button onclick="startRun()">Start run</button>
                    <button class="secondary" onclick="stopRun()">Stop</button>
                    <button class="secondary" onclick="refresh()">Refresh</button>
                  </header>
                  <main>
                    <section class="card">
                      <div class="row" id="metrics"></div>
                      <div id="loss"></div>
                    </section>
                    <section class="card">
                      <h3 style="margin-top:0">Logs</h3>
                      <pre id="logs"></pre>
                      <h3>Runs</h3>
                      <pre id="runs"></pre>
                    </section>
                  </main>
                  <script>
                    let currentId = 'attached';
                    async function jget(url){ const r = await fetch(url); return r.json(); }
                    async function jpost(url, body){
                      const r = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
                        body: body?JSON.stringify(body):'{}'});
                      return r.json();
                    }
                    function renderMetrics(m){
                      const el = document.getElementById('metrics');
                      el.innerHTML = '';
                      const keys = Object.keys(m||{});
                      keys.slice(0,8).forEach(k=>{
                        const d=document.createElement('div'); d.className='metric';
                        d.innerHTML = '<b>'+Number(m[k]).toFixed(4)+'</b><span>'+k+'</span>';
                        el.appendChild(d);
                      });
                    }
                    function renderLoss(hist){
                      const y = hist||[];
                      const x = y.map((_,i)=>i+1);
                      Plotly.newPlot('loss',[{x,y,type:'scatter',mode:'lines+markers',
                        line:{color:'#60a5fa'}}],{
                        margin:{t:24,r:16,b:40,l:48},
                        paper_bgcolor:'#121a2f', plot_bgcolor:'#121a2f',
                        font:{color:'#e8eefc'},
                        title:'loss'
                      }, {displayModeBar:false, responsive:true});
                    }
                    async function refresh(){
                      try{
                        const st = await jget('/api/state');
                        document.getElementById('status').textContent = st.status||'';
                        renderMetrics(st.metrics||{});
                        renderLoss(st.loss_history||[]);
                        document.getElementById('logs').textContent =
                          (st.logs_tail||[]).join('\\n');
                        const runs = await jget('/api/runs');
                        document.getElementById('runs').textContent =
                          JSON.stringify(runs.runs||[], null, 2);
                      }catch(e){ console.error(e); }
                    }
                    async function startRun(){
                      const res = await jpost('/api/runs', {});
                      currentId = res.id || currentId;
                      refresh();
                    }
                    async function stopRun(){
                      try{ await jpost('/api/runs/'+currentId+'/stop', {}); }catch(e){}
                      try{ await jpost('/api/runs/attached/stop', {}); }catch(e){}
                      refresh();
                    }
                    refresh();
                    setInterval(refresh, 2000);
                  </script>
                </body>
                </html>
                """;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> readJson(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            byte[] buf = in.readAllBytes();
            if (buf.length == 0) return Map.of();
            String s = new String(buf, StandardCharsets.UTF_8);
            Object o = Json.decode(s);
            if (o instanceof Map<?, ?> m) {
                return (Map<String, Object>) m;
            }
            return Map.of();
        }
    }

    private static void writeJson(HttpExchange ex, int code, Object body) throws IOException {
        byte[] data = Json.encode(body).getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "application/json; charset=utf-8");
        h.set("Access-Control-Allow-Origin", "*");
        ex.sendResponseHeaders(code, data.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(data);
        }
    }

    private static void writeHtml(HttpExchange ex, int code, String html) throws IOException {
        byte[] data = html.getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(code, data.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(data);
        }
    }

    private static void writeText(HttpExchange ex, int code, String text) throws IOException {
        byte[] data = (text == null ? "" : text).getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "text/plain; charset=utf-8");
        ex.sendResponseHeaders(code, data.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(data);
        }
    }

    @Override
    public void close() {
        try {
            server.stop(0);
        } catch (Exception ignored) {
        }
        workers.shutdownNow();
        for (RunHandle h : runs.values()) {
            h.requestStop();
        }
    }

    private static final class RunHandle {
        final String id;
        final FactoryArgs args;
        final BoardState state;
        volatile FinetuneAdapter job;
        volatile Future<?> future;
        volatile int globalStep;
        volatile Map<String, Double> lastMetrics = Map.of();

        RunHandle(String id, FactoryArgs args, BoardState state) {
            this.id = Objects.requireNonNull(id);
            this.args = Objects.requireNonNull(args);
            this.state = Objects.requireNonNull(state);
        }

        void requestStop() {
            state.requestStop();
            FinetuneAdapter j = job;
            if (j != null) j.requestStop();
            Future<?> f = future;
            if (f != null) f.cancel(true);
        }

        Map<String, Object> summary() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("status", state.status().name());
            m.put("global_step", job != null ? job.globalStep() : globalStep);
            m.put("message", state.message());
            return m;
        }

        Map<String, Object> snapshot() {
            Map<String, Object> m = state.snapshot();
            m.put("id", id);
            m.put("global_step", job != null ? job.globalStep() : globalStep);
            m.put("metrics", job != null ? job.lastMetrics() : lastMetrics);
            return m;
        }

        Map<String, Object> metricsPayload() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("metrics", job != null ? job.lastMetrics() : lastMetrics);
            m.put("loss_history", state.lossHistory());
            m.put("global_step", job != null ? job.globalStep() : globalStep);
            return m;
        }
    }
}
