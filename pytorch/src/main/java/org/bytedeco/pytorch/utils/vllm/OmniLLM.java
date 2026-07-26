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
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.transformers.tokenization.ChatTemplate;
import org.bytedeco.pytorch.utils.vllm.cache.CacheEngine;
import org.bytedeco.pytorch.utils.vllm.multimodal.*;
import org.bytedeco.pytorch.utils.vllm.runner.CausalLmRunner;
import org.bytedeco.pytorch.utils.vllm.runner.EmbeddingRunner;
import org.bytedeco.pytorch.utils.vllm.runner.ModelRunner;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Multimodal LLM facade (vllm-omni style).
 *
 * <p>Supports text + image / audio / video prompts through registered processors.
 * For non-TEXT modalities, falls back to stubs that reserve token budgets so the
 * text path still runs end-to-end.
 */
public final class OmniLLM implements AutoCloseable {

    private final LLMEngine engine;
    private final EngineConfig config;
    private final FastTokenizer tokenizer;
    private final ChatTemplate chatTemplate;
    private final MultimodalProcessor processor;

    private OmniLLM(LLMEngine engine, EngineConfig config,
                    FastTokenizer tokenizer, ChatTemplate chatTemplate,
                    MultimodalProcessor processor) {
        this.engine = Objects.requireNonNull(engine);
        this.config = config;
        this.tokenizer = tokenizer;
        this.chatTemplate = chatTemplate;
        this.processor = processor;
    }

    /** Load from HuggingFace model id via HfHub. */
    public static OmniLLM fromPretrained(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub, EngineConfig.cpuDefault());
    }

    public static OmniLLM fromPretrained(String modelId, HfHub hub, EngineConfig engConfig) throws IOException {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.fromPretrained(modelId, hub);
        return fromBundle(bundle, engConfig);
    }

    /** Load from local directory (HF layout with config.json + safetensors). */
    public static OmniLLM fromDirectory(Path dir) throws IOException {
        return fromDirectory(dir, EngineConfig.cpuDefault());
    }

    public static OmniLLM fromDirectory(Path dir, EngineConfig engConfig) throws IOException {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.fromDirectory(dir);
        return fromBundle(bundle, engConfig);
    }

    /** Tiny offline model for offline benchmarking. */
    public static OmniLLM tiny(String kind) {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.tiny(kind);
        return fromBundle(bundle, EngineConfig.cpuDefault());
    }

    private static OmniLLM fromBundle(AutoModelForCausalLM.Bundle bundle, EngineConfig engConfig) {
        Module model = bundle.model();
        PretrainedConfig cfg = bundle.config();
        FastTokenizer tok = bundle.tokenizer();

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

        ChatTemplate ct = bundle.chatTemplate();
        MultimodalProcessor proc = new TextOnlyProcessor(tok, ct);

        return new OmniLLM(engine, ec, tok, ct, proc);
    }

    /** Generate for text prompts (convenience, no multimodal). */
    public List<RequestOutput> generate(List<String> prompts, SamplingParams params) {
        int[] reqIds = new int[prompts.size()];
        for (int i = 0; i < prompts.size(); i++) {
            int[] ids = tokenizer.encode(prompts.get(i), true).ids();
            reqIds[i] = (int) engine.addRequest(ids, params, prompts.get(i), null);
        }
        List<RequestOutput> outs = engine.generateAll();
        return outs;
    }

    public String chat(List<Map<String, String>> messages, SamplingParams params) {
        String prompt = chatTemplate.apply(messages, true);
        int[] ids = tokenizer.encode(prompt, true).ids();
        engine.addRequest(ids, params, prompt, null);
        List<RequestOutput> outs = engine.generateAll();
        if (outs.isEmpty()) return "";
        RequestOutput out = outs.get(0);
        // Decode output tokens
        int[] outIds = out.outputs.isEmpty() ? new int[0] : out.outputs.get(0).tokenIds;
        return tokenizer.decode(outIds, true);
    }

    /**
     * Generate for a multimodal prompt (text + media).
     * Media parts are processed by the registered MultimodalProcessor.
     */
    public RequestOutput generate(MultimodalPrompt prompt, SamplingParams params) {
        int[] ids = processor.process(prompt, null);
        long reqId = engine.addRequest(ids, params, prompt.toString(), null);
        List<RequestOutput> outs = engine.generateAll();
        return outs.isEmpty() ? null : outs.get(0);
    }

    /** Batch text embedding via the embedded EmbeddingRunner (if set). */
    public float[][] embed(List<String> texts, EmbeddingRunner embedRunner) {
        return engine.embedTexts(texts, embedRunner);
    }

    public EngineConfig config() { return config; }
    public EngineMetrics metrics() { return engine.metrics(); }
    public LLMEngine engine() { return engine; }
    public FastTokenizer tokenizer() { return tokenizer; }
    public ChatTemplate chatTemplate() { return chatTemplate; }

    @Override
    public void close() { engine.close(); }
}
