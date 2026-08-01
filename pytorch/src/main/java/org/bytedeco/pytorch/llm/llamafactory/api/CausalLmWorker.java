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

import org.bytedeco.pytorch.llm.llamafactory.chat.Conversation;
import org.bytedeco.pytorch.llm.llamafactory.chat.StreamCallback;
import org.bytedeco.pytorch.llm.llamafactory.data.SimpleTokenizer;
import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.GeneratingArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.InferArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.transformers.CausalLM;

import java.util.Objects;
import java.util.concurrent.Semaphore;

/**
 * In-process generation worker over {@link CausalLM} (OpenAI API backend).
 *
 * <p>vLLM / SGLang workers can implement the same {@link #chat} / {@link #complete}
 * surface later; this is the default pure-Java path used by {@link OpenAiServer}.
 */
public final class CausalLmWorker implements AutoCloseable {

    private final LoadedModel loaded;
    private final CausalLM causal;
    private final SimpleTokenizer tokenizer;
    private final Template template;
    private final String modelId;
    private final GeneratingArgs defaults;
    private final Semaphore inflight;

    public CausalLmWorker(LoadedModel loaded, InferArgs infer) {
        this.loaded = Objects.requireNonNull(loaded, "loaded");
        this.causal = Objects.requireNonNull(loaded.causalLM(), "causalLM");
        this.tokenizer = SimpleTokenizer.defaults();
        InferArgs ia = infer == null ? InferArgs.defaults() : infer;
        this.template = TemplateRegistry.getOrDefault(ia.template());
        this.modelId = ia.modelNameOrPath() == null ? "default" : ia.modelNameOrPath();
        this.defaults = ia.generating() == null ? GeneratingArgs.defaults() : ia.generating();
        int maxConc = Math.max(1, ia.maxConcurrent());
        this.inflight = new Semaphore(maxConc);
    }

    public static CausalLmWorker open(InferArgs infer) {
        Objects.requireNonNull(infer, "infer");
        ModelArgs ma = ModelArgs.builder()
                .modelNameOrPath(infer.modelNameOrPath())
                .adapterNameOrPath(infer.adapterNameOrPath())
                .quantizationMethod(infer.quantizationMethod())
                .flashAttn(infer.flashAttn())
                .useUnsloth(infer.useUnsloth())
                .build();
        LoadedModel loaded = ModelLoader.load(
                FactoryArgs.builder()
                        .model(ma)
                        .generating(infer.generating())
                        .infer(infer)
                        .build());
        return new CausalLmWorker(loaded, infer);
    }

    public String modelId() { return modelId; }
    public LoadedModel loaded() { return loaded; }

    public static final class GenResult {
        public final String text;
        public final int promptTokens;
        public final int completionTokens;

        public GenResult(String text, int promptTokens, int completionTokens) {
            this.text = text == null ? "" : text;
            this.promptTokens = Math.max(0, promptTokens);
            this.completionTokens = Math.max(0, completionTokens);
        }
    }

    public GenResult chat(OpenAiTypes.ChatCompletionRequest req) throws InterruptedException {
        Objects.requireNonNull(req, "req");
        inflight.acquire();
        try {
            Conversation conv = new Conversation(defaults.defaultSystem());
            for (OpenAiTypes.ChatMessage m : req.messages) {
                if ("system".equalsIgnoreCase(m.role) && conv.size() == 0
                        && (conv.system() == null || conv.system().isBlank())) {
                    conv.setSystem(m.content);
                } else {
                    conv.add(m.role, m.content);
                }
            }
            String prompt = conv.render(template);
            return generate(prompt, req.maxTokens, req.temperature, req.topP);
        } finally {
            inflight.release();
        }
    }

    public GenResult complete(OpenAiTypes.CompletionRequest req) throws InterruptedException {
        Objects.requireNonNull(req, "req");
        inflight.acquire();
        try {
            return generate(req.prompt, req.maxTokens, req.temperature, req.topP);
        } finally {
            inflight.release();
        }
    }

    /**
     * Best-effort streaming: generates fully then emits coarse chunks (CausalLM
     * generate is non-streaming today). Still drives SSE clients correctly.
     */
    public GenResult chatStream(OpenAiTypes.ChatCompletionRequest req, StreamCallback cb)
            throws InterruptedException {
        GenResult r = chat(req);
        if (cb != null) {
            String text = r.text;
            int chunk = Math.max(1, text.length() / 8);
            for (int i = 0; i < text.length(); i += chunk) {
                int end = Math.min(text.length(), i + chunk);
                if (!cb.onChunk(text.substring(i, end))) {
                    break;
                }
            }
            cb.onComplete(text);
        }
        return r;
    }

    private GenResult generate(String prompt, int maxTokens, double temperature, double topP) {
        long[] ids = tokenizer.encode(prompt == null ? "" : prompt, false);
        int[] promptIds = new int[ids.length];
        for (int i = 0; i < ids.length; i++) promptIds[i] = (int) ids[i];

        int maxNew = maxTokens > 0 ? maxTokens
                : (defaults.maxNewTokens() > 0 ? defaults.maxNewTokens() : 64);

        CausalLM.GenerationConfig cfg;
        if (temperature <= 0.0) {
            cfg = CausalLM.GenerationConfig.greedy();
        } else {
            cfg = CausalLM.GenerationConfig.builder()
                    .doSample(true)
                    .temperature(temperature > 0 ? temperature : defaults.temperature())
                    .topK(defaults.topK() > 0 ? defaults.topK() : 50)
                    .topP(topP > 0 ? topP : defaults.topP())
                    .eosStop(true)
                    .build();
        }

        int[] out = causal.generate(promptIds, maxNew, cfg);
        int[] genOnly = stripPrompt(out, promptIds);
        String text = tokenizer.decode(toLong(genOnly));
        if (defaults.skipSpecialTokens()) {
            text = text.trim();
        }
        return new GenResult(text, promptIds.length, genOnly.length);
    }

    private static int[] stripPrompt(int[] full, int[] prompt) {
        if (full == null) return new int[0];
        if (prompt == null || prompt.length == 0 || full.length <= prompt.length) {
            return full == null ? new int[0] : full;
        }
        boolean prefix = true;
        for (int i = 0; i < prompt.length; i++) {
            if (full[i] != prompt[i]) {
                prefix = false;
                break;
            }
        }
        if (!prefix) return full;
        int[] gen = new int[full.length - prompt.length];
        System.arraycopy(full, prompt.length, gen, 0, gen.length);
        return gen;
    }

    private static long[] toLong(int[] ids) {
        long[] o = new long[ids.length];
        for (int i = 0; i < ids.length; i++) o[i] = ids[i];
        return o;
    }

    @Override
    public void close() {
        try {
            loaded.close();
        } catch (Exception ignored) {
        }
    }
}
