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

package org.bytedeco.pytorch.llm.llamacpp;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Enterprise process-backed runtime: official llama-server + OpenAI-compatible HTTP.
 */
public final class ProcessLlamaRuntime implements LlamaEngine {

    private final LlamaRuntimeConfig config;
    private final AtomicBoolean loaded = new AtomicBoolean(false);
    private LlamaProcessManager manager;
    private LlamaServerClient client;
    private LlamaModel modelMeta; // metadata only
    private LlamaChatFormatter formatter;

    public ProcessLlamaRuntime(LlamaRuntimeConfig config) {
        this.config = Objects.requireNonNull(config);
    }

    @Override
    public LlamaBackend backend() { return LlamaBackend.PROCESS_SERVER; }

    @Override
    public LlamaRuntimeConfig config() { return config; }

    @Override
    public synchronized void load() throws Exception {
        if (loaded.get()) return;
        // parse GGUF metadata without full weight dequant
        this.modelMeta = GgufModelLoader.load(config.modelPath(), false);
        this.formatter = new LlamaChatFormatter(modelMeta.hparams().architecture(),
                config.chatTemplate().orElse(null));
        this.manager = new LlamaProcessManager(config);
        int port = manager.start();
        this.client = new LlamaServerClient(config.serverHost(), port);
        loaded.set(true);
    }

    @Override
    public boolean isLoaded() { return loaded.get() && manager != null && manager.isAlive(); }

    @Override
    public Optional<LlamaModel> model() { return Optional.ofNullable(modelMeta); }

    @Override
    public Optional<LlamaHParams> hparams() {
        return modelMeta != null ? Optional.of(modelMeta.hparams()) : Optional.empty();
    }

    @Override
    public synchronized String complete(String prompt, LlamaSamplingParams params) throws Exception {
        ensureLoaded();
        return client.completions(prompt, params != null ? params : LlamaSamplingParams.defaults());
    }

    @Override
    public synchronized int[] generate(int[] promptTokens, LlamaSamplingParams params) throws Exception {
        // process backend works on text; approximate by joining token ids as text prompt
        ensureLoaded();
        StringBuilder sb = new StringBuilder();
        for (int id : promptTokens) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(id);
        }
        String out = complete(sb.toString(), params);
        byte[] raw = out.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[promptTokens.length + Math.min(raw.length, params.maxTokens())];
        System.arraycopy(promptTokens, 0, ids, 0, promptTokens.length);
        for (int i = 0; i < ids.length - promptTokens.length; i++) {
            ids[promptTokens.length + i] = raw[i] & 0xff;
        }
        return ids;
    }

    @Override
    public synchronized String chat(List<Map<String, String>> messages, LlamaSamplingParams params) throws Exception {
        ensureLoaded();
        LlamaSamplingParams sp = params != null ? params : LlamaSamplingParams.defaults();
        try {
            return client.chatCompletions(messages, sp);
        } catch (Exception e) {
            // fallback: format locally and use /completion
            String prompt = formatter.format(messages);
            return client.completions(prompt, sp);
        }
    }

    @Override
    public void reset() {
        // server-side slots: best-effort no-op; new chat messages carry full history
    }

    @Override
    public Map<String, Object> stats() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("backend", backend().name());
        m.put("loaded", isLoaded());
        if (manager != null) {
            m.put("port", manager.boundPort());
            m.put("bin", manager.resolvedBin() != null ? manager.resolvedBin().toString() : null);
            m.put("alive", manager.isAlive());
        }
        if (client != null) m.put("base_url", client.baseUrl());
        if (modelMeta != null) m.put("model", modelMeta.summary());
        return m;
    }

    @Override
    public synchronized void unload() {
        loaded.set(false);
        if (manager != null) {
            manager.stop();
            manager = null;
        }
        client = null;
    }

    private void ensureLoaded() throws Exception {
        if (!loaded.get()) load();
        if (manager != null && !manager.isAlive()) {
            loaded.set(false);
            load();
        }
    }
}
