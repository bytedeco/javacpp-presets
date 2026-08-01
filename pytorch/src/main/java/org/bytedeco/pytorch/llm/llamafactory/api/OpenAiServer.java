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
package org.bytedeco.pytorch.llm.llamafactory.api;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.bytedeco.pytorch.llm.llamafactory.FactoryVersion;
import org.bytedeco.pytorch.llm.llamafactory.LlamaFactory;
import org.bytedeco.pytorch.llm.llamafactory.chat.StreamCallback;
import org.bytedeco.pytorch.llm.llamafactory.hparams.InferArgs;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * OpenAI-compatible HTTP server ({@code com.sun.net.httpserver}).
 *
 * <p>Routes:
 * <ul>
 *   <li>{@code GET  /v1/models}</li>
 *   <li>{@code POST /v1/chat/completions}</li>
 *   <li>{@code POST /v1/completions}</li>
 *   <li>{@code GET  /health}</li>
 * </ul>
 *
 * <p>Started via {@link #start(InferArgs)} — also the reflective target of
 * {@link LlamaFactory#serveApi(InferArgs)}.
 */
public final class OpenAiServer implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(OpenAiServer.class.getName());

    private final HttpServer server;
    private final int port;
    private final CausalLmWorker worker;
    private final ApiAuth auth;
    private final boolean ownsWorker;

    private OpenAiServer(HttpServer server, int port, CausalLmWorker worker,
                         ApiAuth auth, boolean ownsWorker) {
        this.server = server;
        this.port = port;
        this.worker = worker;
        this.auth = auth == null ? ApiAuth.disabled() : auth;
        this.ownsWorker = ownsWorker;
    }

    public static OpenAiServer start(InferArgs args) throws IOException {
        Objects.requireNonNull(args, "args");
        CausalLmWorker worker = CausalLmWorker.open(args);
        try {
            return start(args, worker, true);
        } catch (IOException e) {
            worker.close();
            throw e;
        }
    }

    public static OpenAiServer start(InferArgs args, CausalLmWorker worker) throws IOException {
        return start(args, worker, false);
    }

    private static OpenAiServer start(InferArgs args, CausalLmWorker worker, boolean ownsWorker)
            throws IOException {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(worker, "worker");
        String host = args.host() == null || args.host().isBlank() ? "0.0.0.0" : args.host();
        int port = args.port() > 0 ? args.port() : 8000;
        HttpServer hs = HttpServer.create(new InetSocketAddress(host, port), 0);
        ApiAuth auth = new ApiAuth(args.apiKey());
        OpenAiServer api = new OpenAiServer(hs, hs.getAddress().getPort(), worker, auth, ownsWorker);
        api.installRoutes();
        hs.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "openai-api");
            t.setDaemon(true);
            return t;
        }));
        hs.start();
        LOG.info("OpenAiServer listening on http://" + host + ":" + api.port
                + " model=" + worker.modelId());
        return api;
    }

    public int port() { return port; }
    public String baseUrl() { return "http://127.0.0.1:" + port; }
    public CausalLmWorker worker() { return worker; }

    private void installRoutes() {
        server.createContext("/health", ex -> writeJson(ex, 200, Map.of(
                "ok", true,
                "version", FactoryVersion.VERSION)));
        server.createContext("/v1/models", this::handleModels);
        server.createContext("/v1/chat/completions", this::handleChat);
        server.createContext("/v1/completions", this::handleCompletions);
        server.createContext("/", ex -> {
            if ("GET".equalsIgnoreCase(ex.getRequestMethod())
                    && ("/".equals(ex.getRequestURI().getPath())
                    || "/v1".equals(ex.getRequestURI().getPath()))) {
                Map<String, Object> body = new LinkedHashMap<>();
                body.put("service", "llamafactory-openai");
                body.put("version", FactoryVersion.VERSION);
                body.put("endpoints", java.util.List.of(
                        "/v1/models", "/v1/chat/completions", "/v1/completions", "/health"));
                writeJson(ex, 200, body);
            } else {
                writeJson(ex, 404, OpenAiTypes.error("not found", "not_found_error", 404));
            }
        });
    }

    private boolean authorized(HttpExchange ex) throws IOException {
        String authH = header(ex, "Authorization");
        String keyH = header(ex, "X-Api-Key");
        if (auth.allow(authH, keyH)) {
            return true;
        }
        Headers h = ex.getResponseHeaders();
        auth.challenge().ifPresent(c -> h.set("WWW-Authenticate", c));
        writeJson(ex, 401, OpenAiTypes.error("Invalid API key", "authentication_error", 401));
        return false;
    }

    private void handleModels(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
            writeJson(ex, 405, OpenAiTypes.error("method not allowed", "invalid_request_error", 405));
            return;
        }
        if (!authorized(ex)) return;
        writeJson(ex, 200, OpenAiTypes.modelsList(worker.modelId()));
    }

    @SuppressWarnings("unchecked")
    private void handleChat(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            writeJson(ex, 405, OpenAiTypes.error("method not allowed", "invalid_request_error", 405));
            return;
        }
        if (!authorized(ex)) return;
        try {
            Map<String, Object> body = readJson(ex);
            OpenAiTypes.ChatCompletionRequest req = OpenAiTypes.ChatCompletionRequest.fromMap(body);
            if (req.messages.isEmpty()) {
                writeJson(ex, 400, OpenAiTypes.error("messages required", "invalid_request_error", 400));
                return;
            }
            if (req.stream) {
                streamChat(ex, req);
                return;
            }
            CausalLmWorker.GenResult r = worker.chat(req);
            writeJson(ex, 200, OpenAiTypes.chatCompletionResponse(
                    req.model == null || "default".equals(req.model) ? worker.modelId() : req.model,
                    r.text, r.promptTokens, r.completionTokens));
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            writeJson(ex, 503, OpenAiTypes.error("interrupted", "server_error", 503));
        } catch (Exception e) {
            LOG.log(Level.WARNING, "chat failed", e);
            writeJson(ex, 500, OpenAiTypes.error(
                    e.getMessage() == null ? e.toString() : e.getMessage(),
                    "server_error", 500));
        }
    }

    private void streamChat(HttpExchange ex, OpenAiTypes.ChatCompletionRequest req) throws IOException {
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "text/event-stream; charset=utf-8");
        h.set("Cache-Control", "no-cache");
        h.set("Access-Control-Allow-Origin", "*");
        ex.sendResponseHeaders(200, 0);
        String model = req.model == null || "default".equals(req.model) ? worker.modelId() : req.model;
        try (OutputStream os = ex.getResponseBody()) {
            StreamCallback cb = chunk -> {
                try {
                    String data = Json.encode(OpenAiTypes.chatChunk(model, chunk, false));
                    os.write(("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
                    os.flush();
                    return true;
                } catch (IOException e) {
                    return false;
                }
            };
            try {
                worker.chatStream(req, cb);
                String done = Json.encode(OpenAiTypes.chatChunk(model, "", true));
                os.write(("data: " + done + "\n\n").getBytes(StandardCharsets.UTF_8));
                os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                os.flush();
            } catch (Exception e) {
                String err = Json.encode(OpenAiTypes.error(
                        e.getMessage() == null ? e.toString() : e.getMessage(),
                        "server_error", 500));
                os.write(("data: " + err + "\n\n").getBytes(StandardCharsets.UTF_8));
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void handleCompletions(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            writeJson(ex, 405, OpenAiTypes.error("method not allowed", "invalid_request_error", 405));
            return;
        }
        if (!authorized(ex)) return;
        try {
            Map<String, Object> body = readJson(ex);
            OpenAiTypes.CompletionRequest req = OpenAiTypes.CompletionRequest.fromMap(body);
            CausalLmWorker.GenResult r = worker.complete(req);
            writeJson(ex, 200, OpenAiTypes.completionResponse(
                    req.model == null || "default".equals(req.model) ? worker.modelId() : req.model,
                    r.text, r.promptTokens, r.completionTokens));
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            writeJson(ex, 503, OpenAiTypes.error("interrupted", "server_error", 503));
        } catch (Exception e) {
            LOG.log(Level.WARNING, "completion failed", e);
            writeJson(ex, 500, OpenAiTypes.error(
                    e.getMessage() == null ? e.toString() : e.getMessage(),
                    "server_error", 500));
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> readJson(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            byte[] buf = in.readAllBytes();
            if (buf.length == 0) return Map.of();
            Object o = Json.decode(new String(buf, StandardCharsets.UTF_8));
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

    private static String header(HttpExchange ex, String name) {
        return ex.getRequestHeaders().getFirst(name);
    }

    @Override
    public void close() {
        try {
            server.stop(0);
        } catch (Exception ignored) {
        }
        if (ownsWorker) {
            try {
                worker.close();
            } catch (Exception ignored) {
            }
        }
    }
}
