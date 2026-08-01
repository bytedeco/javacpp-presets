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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Pure-Java GGUF inference engine (llama.cpp behaviour surface).
 * Decode path goes through {@link LlamaContext} (KV + transformer + batch).
 */
public final class InProcessLlamaEngine implements LlamaEngine {

    private final LlamaRuntimeConfig config;
    private final AtomicBoolean loaded = new AtomicBoolean(false);
    private LlamaModel model;
    private LlamaHParams hparams;
    private LlamaContext context;
    private LlamaChatFormatter formatter;
    private final AtomicLong promptTokens = new AtomicLong();
    private final AtomicLong completionTokens = new AtomicLong();

    public InProcessLlamaEngine(LlamaRuntimeConfig config) {
        this.config = Objects.requireNonNull(config);
    }

    @Override
    public LlamaBackend backend() { return LlamaBackend.IN_PROCESS; }

    @Override
    public LlamaRuntimeConfig config() { return config; }

    @Override
    public synchronized void load() throws Exception {
        if (loaded.get()) return;
        this.model = GgufModelLoader.load(config.modelPath(), true);
        this.hparams = model.hparams();
        this.context = LlamaContext.create(model, config);
        this.formatter = new LlamaChatFormatter(hparams.architecture(),
                config.chatTemplate().orElse(null));
        loaded.set(true);
    }

    @Override
    public boolean isLoaded() { return loaded.get(); }

    @Override
    public Optional<LlamaModel> model() { return Optional.ofNullable(model); }

    @Override
    public Optional<LlamaHParams> hparams() { return Optional.ofNullable(hparams); }

    /** Exposed for advanced hosts that drive decode batches manually. */
    public Optional<LlamaContext> context() { return Optional.ofNullable(context); }

    @Override
    public synchronized String complete(String prompt, LlamaSamplingParams params) throws Exception {
        ensureLoaded();
        int[] ids = context.tokenizer().encode(prompt, true);
        int[] full = generate(ids, params);
        return context.tokenizer().decodeNew(full, ids.length);
    }

    @Override
    public synchronized int[] generate(int[] promptTokensIn, LlamaSamplingParams params) throws Exception {
        ensureLoaded();
        Objects.requireNonNull(promptTokensIn, "promptTokens");
        LlamaSamplingParams sp = params != null ? params : LlamaSamplingParams.defaults();
        LlamaSampler sampler = new LlamaSampler(sp);
        context.reset();

        List<Integer> out = new ArrayList<>(promptTokensIn.length + sp.maxTokens());
        for (int id : promptTokensIn) out.add(id);

        if (promptTokensIn.length == 0) {
            float[] logits = context.step(context.tokenizer().bosId());
            int next = sampler.sampleToken(logits);
            out.add(next);
            completionTokens.incrementAndGet();
            return toArray(out);
        }

        // Prefill all prompt tokens; sample from last-position logits
        float[] logits = context.prefill(promptTokensIn);
        promptTokens.addAndGet(promptTokensIn.length);

        int next = sampler.sampleToken(logits);
        out.add(next);
        int generated = 1;
        while (generated < sp.maxTokens()) {
            if (next == context.tokenizer().eosId() && generated > 1) break;
            if (context.nPast() >= config.nCtx() - 1) break;
            logits = context.step(next);
            next = sampler.sampleToken(logits);
            out.add(next);
            generated++;
        }
        completionTokens.addAndGet(generated);
        return toArray(out);
    }

    @Override
    public synchronized String chat(List<Map<String, String>> messages, LlamaSamplingParams params) throws Exception {
        ensureLoaded();
        String prompt = formatter.format(messages);
        return complete(prompt, params);
    }

    @Override
    public synchronized void reset() {
        if (context != null) context.reset();
    }

    @Override
    public Map<String, Object> stats() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("backend", backend().name());
        m.put("loaded", isLoaded());
        m.put("prompt_tokens", promptTokens.get());
        m.put("completion_tokens", completionTokens.get());
        if (model != null) m.put("model", model.summary());
        if (context != null) m.put("n_past", context.nPast());
        return m;
    }

    @Override
    public synchronized void unload() {
        loaded.set(false);
        if (context != null) {
            context.close();
            context = null;
        }
        model = null;
        hparams = null;
        formatter = null;
    }

    private void ensureLoaded() throws Exception {
        if (!loaded.get()) load();
    }

    private static int[] toArray(List<Integer> ids) {
        int[] a = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) a[i] = ids.get(i);
        return a;
    }
}
