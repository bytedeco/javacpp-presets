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

package org.bytedeco.pytorch.llm.unsloth.studio.mcp;

import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Minimal MCP JSON-RPC server over stdio (opt-in control endpoint).
 * Implements tools/list and tools/call.
 */
public final class McpServer implements AutoCloseable {

    private final McpToolRegistry registry;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "studio-mcp");
        t.setDaemon(true);
        return t;
    });

    public McpServer(McpToolRegistry registry) {
        this.registry = registry;
    }

    public McpToolRegistry registry() { return registry; }

    public void start() {
        if (!running.compareAndSet(false, true)) return;
        executor.submit(this::loopStdio);
    }

    public boolean isRunning() { return running.get(); }

    /** Handle one JSON-RPC request object (for tests / HTTP bridge). */
    @SuppressWarnings("unchecked")
    public Map<String, Object> handle(Map<String, Object> req) {
        Map<String, Object> resp = new LinkedHashMap<>();
        Object id = req.get("id");
        if (id != null) resp.put("id", id);
        resp.put("jsonrpc", "2.0");
        String method = req.get("method") != null ? String.valueOf(req.get("method")) : "";
        try {
            Object params = req.get("params");
            Map<String, Object> p = params instanceof Map<?, ?> mm ? (Map<String, Object>) mm : Map.of();
            switch (method) {
                case "initialize" -> resp.put("result", Map.of(
                        "protocolVersion", "2024-11-05",
                        "capabilities", Map.of("tools", Map.of()),
                        "serverInfo", Map.of("name", "unsloth-studio-java", "version", "1.0.0-beta")));
                case "tools/list" -> resp.put("result", Map.of("tools", registry.listTools()));
                case "tools/call" -> {
                    String name = String.valueOf(p.get("name"));
                    Map<String, Object> args = p.get("arguments") instanceof Map<?, ?> am
                            ? (Map<String, Object>) am : Map.of();
                    Object result = registry.call(name, args);
                    resp.put("result", Map.of(
                            "content", List.of(Map.of("type", "text", "text", JsonMaps.stringify(result))),
                            "isError", false));
                }
                case "ping" -> resp.put("result", Map.of());
                default -> resp.put("error", Map.of("code", -32601, "message", "Method not found: " + method));
            }
        } catch (Exception e) {
            resp.put("error", Map.of("code", -32000, "message", String.valueOf(e.getMessage())));
        }
        return resp;
    }

    private void loopStdio() {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
             BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8))) {
            String line;
            while (running.get() && (line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                try {
                    Map<String, Object> req = JsonMaps.parseObject(line);
                    Map<String, Object> resp = handle(req);
                    bw.write(JsonMaps.stringify(resp));
                    bw.newLine();
                    bw.flush();
                } catch (Exception e) {
                    bw.write(JsonMaps.stringify(Map.of(
                            "jsonrpc", "2.0",
                            "error", Map.of("code", -32700, "message", "Parse error: " + e.getMessage()))));
                    bw.newLine();
                    bw.flush();
                }
            }
        } catch (Exception ignored) {
        } finally {
            running.set(false);
        }
    }

    @Override
    public void close() {
        running.set(false);
        executor.shutdownNow();
    }
}
