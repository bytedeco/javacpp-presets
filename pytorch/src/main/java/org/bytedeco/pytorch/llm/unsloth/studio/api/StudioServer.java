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

package org.bytedeco.pytorch.llm.unsloth.studio.api;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.bytedeco.pytorch.llm.unsloth.studio.StudioOptions;
import org.bytedeco.pytorch.llm.unsloth.studio.StudioVersion;
import org.bytedeco.pytorch.llm.unsloth.studio.export.ExportOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.hardware.DeviceProbe;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioInventory;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioModelRegistry;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.InferenceOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ExportRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.LoadRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ModelCard;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingStartRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.train.StudioTrainingOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioValidationException;

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
import java.util.concurrent.Executors;

/**
 * OpenAI / Anthropic compatible + Studio control HTTP server.
 * Routes: /v1/chat/completions, /v1/models, /v1/messages, /v1/responses,
 * plus /studio/train|export|hardware|models.
 */
public final class StudioServer implements AutoCloseable {

    private final StudioOptions options;
    private final InferenceOrchestrator inference;
    private final StudioTrainingOrchestrator training;
    private final ExportOrchestrator export;
    private final StudioModelRegistry registry;
    private final StudioInventory inventory;
    private HttpServer server;
    private volatile boolean running;
    private int boundPort;

    public StudioServer(StudioOptions options,
                        InferenceOrchestrator inference,
                        StudioTrainingOrchestrator training,
                        ExportOrchestrator export,
                        StudioModelRegistry registry,
                        StudioInventory inventory) {
        this.options = Objects.requireNonNull(options);
        this.inference = inference;
        this.training = training;
        this.export = export;
        this.registry = registry;
        this.inventory = inventory;
    }

    public synchronized void start() throws IOException {
        if (running) return;
        server = HttpServer.create(new InetSocketAddress(options.apiBindHost(), options.apiPort()), 0);
        server.createContext("/health", ex -> writeJson(ex, 200, Map.of("ok", true, "version", StudioVersion.version())));
        server.createContext("/v1/models", this::handleModels);
        server.createContext("/v1/chat/completions", this::handleChatCompletions);
        server.createContext("/v1/messages", this::handleAnthropicMessages);
        server.createContext("/v1/responses", this::handleResponses);
        server.createContext("/studio/train/start", this::handleTrainStart);
        server.createContext("/studio/train/stop", this::handleTrainStop);
        server.createContext("/studio/train/list", this::handleTrainList);
        server.createContext("/studio/export", this::handleExport);
        server.createContext("/studio/hardware", ex -> writeJson(ex, 200, DeviceProbe.probe().toMap()));
        server.createContext("/studio/models", this::handleStudioModels);
        server.createContext("/studio/load", this::handleLoad);
        server.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "studio-api");
            t.setDaemon(true);
            return t;
        }));
        server.start();
        boundPort = server.getAddress().getPort();
        running = true;
    }

    public boolean isRunning() { return running; }
    public int port() { return boundPort > 0 ? boundPort : options.apiPort(); }

    public synchronized void close() {
        running = false;
        if (server != null) {
            server.stop(0);
            server = null;
        }
    }

    private void handleModels(HttpExchange ex) throws IOException {
        List<Map<String, Object>> data = new ArrayList<>();
        for (ModelCard c : registry.search("")) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", c.id());
            m.put("object", "model");
            m.put("owned_by", "unsloth-studio-java");
            data.add(m);
        }
        writeJson(ex, 200, Map.of("object", "list", "data", data));
    }

    private void handleChatCompletions(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            writeJson(ex, 405, Map.of("error", "POST required"));
            return;
        }
        try {
            Map<String, Object> body = readBody(ex);
            ChatCompletionRequest req = ChatCompletionRequest.fromMap(body);
            ChatCompletionResponse resp = inference.chatCompletions(req);
            writeJson(ex, 200, resp.toMap());
        } catch (StudioValidationException ve) {
            writeJson(ex, 422, Map.of("error", Map.of("message", ve.getMessage(), "type", "validation_error")));
        } catch (Exception e) {
            writeJson(ex, 500, Map.of("error", Map.of("message", String.valueOf(e.getMessage()), "type", "server_error")));
        }
    }

    private void handleAnthropicMessages(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            writeJson(ex, 405, Map.of("error", "POST required"));
            return;
        }
        try {
            Map<String, Object> body = readBody(ex);
            // Adapt Anthropic body → ChatCompletionRequest
            List<org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage> msgs = new ArrayList<>();
            if (body.get("system") != null) {
                msgs.add(org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage.system(String.valueOf(body.get("system"))));
            }
            if (body.get("messages") instanceof List<?> list) {
                for (Object o : list) {
                    if (o instanceof Map<?, ?> mm) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> m = (Map<String, Object>) mm;
                        msgs.add(org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage.fromMap(m));
                    }
                }
            }
            ChatCompletionRequest req = ChatCompletionRequest.builder()
                    .model(body.get("model") != null ? String.valueOf(body.get("model")) : null)
                    .messages(msgs)
                    .maxTokens(body.get("max_tokens") instanceof Number n ? n.intValue() : 256)
                    .build();
            ChatCompletionResponse resp = inference.chatCompletions(req);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("id", resp.id());
            out.put("type", "message");
            out.put("role", "assistant");
            out.put("content", List.of(Map.of("type", "text", "text", resp.firstContent())));
            out.put("model", resp.model());
            out.put("stop_reason", "end_turn");
            writeJson(ex, 200, out);
        } catch (Exception e) {
            writeJson(ex, 500, Map.of("error", Map.of("message", String.valueOf(e.getMessage()))));
        }
    }

    private void handleResponses(HttpExchange ex) throws IOException {
        // OpenAI Responses API minimal subset
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            writeJson(ex, 405, Map.of("error", "POST required"));
            return;
        }
        try {
            Map<String, Object> body = readBody(ex);
            String input = body.get("input") != null ? String.valueOf(body.get("input")) : "";
            ChatCompletionRequest req = ChatCompletionRequest.of(null, input);
            if (body.get("model") != null) {
                req = ChatCompletionRequest.builder()
                        .model(String.valueOf(body.get("model")))
                        .messages(req.messages())
                        .maxTokens(body.get("max_output_tokens") instanceof Number n ? n.intValue() : 256)
                        .build();
            }
            ChatCompletionResponse resp = inference.chatCompletions(req);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("id", "resp_" + resp.id());
            out.put("object", "response");
            out.put("status", "completed");
            out.put("model", resp.model());
            out.put("output_text", resp.firstContent());
            out.put("output", List.of(Map.of(
                    "type", "message",
                    "role", "assistant",
                    "content", List.of(Map.of("type", "output_text", "text", resp.firstContent())))));
            writeJson(ex, 200, out);
        } catch (Exception e) {
            writeJson(ex, 500, Map.of("error", Map.of("message", String.valueOf(e.getMessage()))));
        }
    }

    private void handleTrainStart(HttpExchange ex) throws IOException {
        try {
            Map<String, Object> body = readBody(ex);
            TrainingStartRequest req = TrainingStartRequest.fromMap(body);
            String runId = training.start(req);
            writeJson(ex, 200, Map.of("run_id", runId, "status", "started"));
        } catch (StudioValidationException ve) {
            writeJson(ex, 422, Map.of("error", ve.getMessage()));
        } catch (Exception e) {
            writeJson(ex, 500, Map.of("error", String.valueOf(e.getMessage())));
        }
    }

    private void handleTrainStop(HttpExchange ex) throws IOException {
        Map<String, Object> body = readBody(ex);
        String runId = String.valueOf(body.get("run_id"));
        training.stop(runId);
        writeJson(ex, 200, Map.of("run_id", runId, "status", "stop_requested"));
    }

    private void handleTrainList(HttpExchange ex) throws IOException {
        List<Map<String, Object>> runs = new ArrayList<>();
        training.list().forEach(r -> runs.add(r.toMap()));
        writeJson(ex, 200, Map.of("runs", runs));
    }

    private void handleExport(HttpExchange ex) throws IOException {
        try {
            Map<String, Object> body = readBody(ex);
            ExportRequest.Builder b = ExportRequest.builder();
            if (body.get("checkpoint_path") != null) b.checkpointPath(String.valueOf(body.get("checkpoint_path")));
            if (body.get("format") != null) b.format(String.valueOf(body.get("format")));
            if (body.get("save_directory") != null) b.saveDirectory(String.valueOf(body.get("save_directory")));
            var path = export.export(b.build());
            writeJson(ex, 200, Map.of("success", true, "output", path.toString(), "status", export.status()));
        } catch (Exception e) {
            writeJson(ex, 500, Map.of("success", false, "error", String.valueOf(e.getMessage())));
        }
    }

    private void handleStudioModels(HttpExchange ex) throws IOException {
        String q = ex.getRequestURI().getQuery();
        String query = "";
        if (q != null && q.startsWith("q=")) query = q.substring(2);
        List<Map<String, Object>> list = new ArrayList<>();
        for (ModelCard c : registry.search(query)) list.add(c.toMap());
        writeJson(ex, 200, Map.of("models", list));
    }

    private void handleLoad(HttpExchange ex) throws IOException {
        try {
            Map<String, Object> body = readBody(ex);
            LoadRequest.Builder b = LoadRequest.builder();
            if (body.get("model_path") != null) b.modelPath(String.valueOf(body.get("model_path")));
            if (body.containsKey("load_in_4bit")) b.loadIn4bit(Boolean.parseBoolean(String.valueOf(body.get("load_in_4bit"))));
            if (body.containsKey("max_seq_length") && body.get("max_seq_length") instanceof Number n) b.maxSeqLength(n.intValue());
            inference.load(b.build());
            writeJson(ex, 200, Map.of("loaded", true, "model", inference.loadedModelId().orElse("")));
        } catch (Exception e) {
            writeJson(ex, 500, Map.of("loaded", false, "error", String.valueOf(e.getMessage())));
        }
    }

    private Map<String, Object> readBody(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            return JsonCodec.readObject(in.readAllBytes());
        }
    }

    private void writeJson(HttpExchange ex, int code, Object body) throws IOException {
        byte[] bytes = JsonCodec.bytes(body);
        ex.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }
}
