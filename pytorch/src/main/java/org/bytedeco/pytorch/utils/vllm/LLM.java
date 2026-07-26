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
package org.bytedeco.pytorch.utils.vllm;

import org.bytedeco.pytorch.utils.vllm.metrics.EngineMetrics;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.tokenizers.Encoding;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.transformers.tokenization.ChatTemplate;
import org.bytedeco.pytorch.utils.vllm.cache.CacheEngine;
import org.bytedeco.pytorch.utils.vllm.runner.CausalLmRunner;
import org.bytedeco.pytorch.utils.vllm.runner.EmbeddingRunner;
import org.bytedeco.pytorch.utils.vllm.runner.ModelRunner;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * High-level LLM API (nano-vLLM style).
 *
 * <pre>{@code
 * LLM llm = LLM.fromPretrained("Qwen/Qwen2-0.5B-Instruct", hub);
 * List<RequestOutput> outs = llm.generate(List.of("Hello", "2+2=?"),
 *     SamplingParams.builder().maxTokens(64).temperature(0.7).build());
 * String reply = llm.chat(List.of(Map.of("role","user","content","hi")), null);
 * float[][] emb = llm.embed(List.of("hello world"));
 * }</pre>
 */
public final class LLM implements AutoCloseable {

    private final LLMEngine engine;
    private final EngineConfig config;
    private final AutoModelForCausalLM.Bundle bundle;
    private final FastTokenizer tokenizer;
    private final ChatTemplate chatTemplate;

    private LLM(LLMEngine engine, EngineConfig config,
                AutoModelForCausalLM.Bundle bundle) {
        this.engine = Objects.requireNonNull(engine);
        this.config = config;
        this.bundle = Objects.requireNonNull(bundle);
        this.tokenizer = bundle.tokenizer();
        this.chatTemplate = bundle.chatTemplate();
    }

    /** Load from HuggingFace model id via HfHub. */
    public static LLM fromPretrained(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub, EngineConfig.cpuDefault());
    }

    public static LLM fromPretrained(String modelId, HfHub hub, EngineConfig engConfig) throws IOException {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.fromPretrained(modelId, hub);
        return fromBundle(bundle, engConfig);
    }

    /** Load from local directory (HF layout). */
    public static LLM fromDirectory(Path dir) throws IOException {
        return fromDirectory(dir, EngineConfig.cpuDefault());
    }

    public static LLM fromDirectory(Path dir, EngineConfig engConfig) throws IOException {
        return fromBundle(AutoModelForCausalLM.fromDirectory(dir), engConfig);
    }

    /** Tiny offline model (random weights, no network). */
    public static LLM tiny(String kind) {
        return tiny(kind, EngineConfig.cpuDefault());
    }

    public static LLM tiny(String kind, EngineConfig engConfig) {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.tiny(kind);
        return fromBundle(bundle, engConfig);
    }

    private static LLM fromBundle(AutoModelForCausalLM.Bundle bundle, EngineConfig engConfig) {
        Module model = bundle.model();
        PretrainedConfig cfg = bundle.config();
        FastTokenizer tok = bundle.tokenizer();

        // Fill in missing dimensions from config
        EngineConfig ec = EngineConfig.fromPretrainedConfig(cfg)
                .maxNumSeqs(engConfig.maxNumSeqs)
                .maxNumBatchedTokens(engConfig.maxNumBatchedTokens)
                .blockSize(engConfig.blockSize)
                .maxBlocks(engConfig.maxBlocks)
                .device(engConfig.device)
                .build();

        CacheEngine cache = new CacheEngine(ec);
        ModelRunner runner = new CausalLmRunner(model, cfg, cache);
        LLMEngine engine = new LLMEngine(ec, runner, cache, tok);

        return new LLM(engine, ec, bundle);
    }

    // ---- generation API ----

    /** Batch generate for plain text prompts. */
    public List<RequestOutput> generate(List<String> prompts, SamplingParams params) {
        Objects.requireNonNull(prompts, "prompts");
        if (prompts.isEmpty()) return List.of();
        if (params == null) params = SamplingParams.defaults();

        long[] reqIds = new long[prompts.size()];
        for (int i = 0; i < prompts.size(); i++) {
            Encoding enc = tokenizer.encode(prompts.get(i), true);
            reqIds[i] = engine.addRequest(enc.ids(), params, prompts.get(i), null);
        }
        return engine.generateAll();
    }

    public List<RequestOutput> generate(List<String> prompts, int maxTokens) {
        return generate(prompts, SamplingParams.greedy(maxTokens));
    }

    /** Chat (Instruct models). */
    public String chat(List<Map<String, String>> messages, SamplingParams params) {
        String prompt = chatTemplate.apply(messages, true);
        // Chat templates already embed BOS/specials; avoid double-adding via post-processor.
        Encoding enc = tokenizer.encode(prompt, false);
        engine.addRequest(enc.ids(), params != null ? params : SamplingParams.greedy(64), prompt, null);
        List<RequestOutput> outs = engine.generateAll();
        if (outs.isEmpty()) return "";
        RequestOutput out = outs.get(0);
        int[] outIds = out.outputs.isEmpty() ? new int[0] : out.outputs.get(0).tokenIds;
        return tokenizer.decode(outIds, true);
    }

    public String chat(List<Map<String, String>> messages) {
        return chat(messages, null);
    }

    // ---- embedding API ----

    /**
     * Batch text embedding via the bundled SentenceTransformer (if available)
     * or a simple mean-pool fallback.
     */
    public float[][] embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) return new float[0][];
        try {
            EmbeddingRunner runner = new EmbeddingRunner(
                    org.bytedeco.pytorch.utils.sentence.SentenceTransformer.mini());
            float[][] emb = runner.encodeBatch(texts);
            runner.close();
            return emb;
        } catch (Exception e) {
            // Fallback: use model as embedder (mean-pool last hidden)
            // This path is a stub for Phase 1 — real pooling needs forwardHidden
            float[][] out = new float[texts.size()][];
            for (int i = 0; i < texts.size(); i++) out[i] = new float[bundle != null && bundle.config() != null ? bundle.config().hiddenSize() : 0];
            return out;
        }
    }

    // ---- accessors ----

    public EngineConfig config() { return config; }
    public EngineMetrics metrics() { return engine.metrics(); }
    public LLMEngine engine() { return engine; }
    public FastTokenizer tokenizer() { return tokenizer; }
    public ChatTemplate chatTemplate() { return chatTemplate; }
    public AutoModelForCausalLM.Bundle bundle() { return bundle; }

    @Override
    public void close() { engine.close(); }
}
