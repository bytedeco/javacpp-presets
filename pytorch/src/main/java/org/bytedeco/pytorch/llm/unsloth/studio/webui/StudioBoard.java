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

package org.bytedeco.pytorch.llm.unsloth.studio.webui;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.bytedeco.pytorch.llm.unsloth.studio.StudioOptions;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingProgressEvent;
import org.bytedeco.pytorch.llm.unsloth.studio.observe.LiveGraphBuffer;
import org.bytedeco.pytorch.llm.unsloth.studio.train.TrainingProgressBus;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.concurrent.Executors;

/**
 * Pure-Java visual training board: HTML dashboard + SSE progress + SVG loss curves.
 */
public final class StudioBoard implements AutoCloseable {

    private final StudioOptions options;
    private final BoardState state;
    private final BoardSseHub sse = new BoardSseHub();
    private final LiveGraphBuffer graphs;
    private HttpServer server;
    private volatile boolean running;
    private int boundPort;

    public StudioBoard(StudioOptions options, LiveGraphBuffer graphs, TrainingProgressBus bus) {
        this.options = Objects.requireNonNull(options);
        this.graphs = graphs != null ? graphs : new LiveGraphBuffer();
        this.state = new BoardState(this.graphs);
        if (bus != null) {
            bus.subscribeAll(ev -> {
                state.onEvent(ev);
                sse.publish(ev);
            });
        }
    }

    public BoardState state() { return state; }
    public LiveGraphBuffer graphs() { return graphs; }
    public boolean isRunning() { return running; }
    public int port() { return boundPort > 0 ? boundPort : options.boardPort(); }

    public synchronized void open() throws IOException {
        if (running) return;
        server = HttpServer.create(new InetSocketAddress(options.apiBindHost(), options.boardPort()), 0);
        server.createContext("/", this::handleIndex);
        server.createContext("/api/snapshot", this::handleSnapshot);
        server.createContext("/api/svg", this::handleSvg);
        server.createContext("/board/events", this::handleSse);
        server.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "studio-board");
            t.setDaemon(true);
            return t;
        }));
        server.start();
        boundPort = server.getAddress().getPort();
        running = true;
    }

    public synchronized void close() {
        running = false;
        if (server != null) {
            server.stop(0);
            server = null;
        }
    }

    private void handleIndex(HttpExchange ex) throws IOException {
        byte[] body = BoardStaticAssets.indexHtml(options.boardPort()).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, body.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(body); }
    }

    private void handleSnapshot(HttpExchange ex) throws IOException {
        byte[] body = JsonMaps.stringify(state.snapshot()).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(200, body.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(body); }
    }

    private void handleSvg(HttpExchange ex) throws IOException {
        String q = ex.getRequestURI().getQuery();
        String key = "loss";
        if (q != null && q.startsWith("key=")) key = q.substring(4);
        // default to first run loss series if bare
        if (!key.contains("/")) {
            var runs = state.runs();
            if (!runs.isEmpty()) key = runs.get(0).runId() + "/" + key;
        }
        String svg = graphs.toSvg(key, 640, 200);
        byte[] body = svg.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "image/svg+xml");
        ex.sendResponseHeaders(200, body.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(body); }
    }

    private void handleSse(HttpExchange ex) throws IOException {
        ex.getResponseHeaders().add("Content-Type", "text/event-stream");
        ex.getResponseHeaders().add("Cache-Control", "no-cache");
        ex.getResponseHeaders().add("Connection", "keep-alive");
        ex.sendResponseHeaders(200, 0);
        OutputStream os = ex.getResponseBody();
        sse.add(os);
        os.write("event: hello\ndata: {\"ok\":true}\n\n".getBytes(StandardCharsets.UTF_8));
        os.flush();
        // keep open until client disconnects — simple block
        try {
            while (running) {
                Thread.sleep(15000);
                os.write(": keepalive\n\n".getBytes(StandardCharsets.UTF_8));
                os.flush();
            }
        } catch (Exception e) {
            sse.remove(os);
            try { os.close(); } catch (Exception ignored) {}
        }
    }
}
