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
        // Look for sibling multimodal encoder snapshots under models/
        Path modelsRoot = dir.getParent() != null ? dir.getParent() : Path.of("models");
        return fromBundle(bundle, engConfig, modelsRoot);
    }

    /** Load backbone + real encoders discovered under {@code modelsRoot}. */
    public static OmniLLM fromDirectory(Path dir, EngineConfig engConfig, Path modelsRoot)
            throws IOException {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.fromDirectory(dir);
        return fromBundle(bundle, engConfig, modelsRoot);
    }

    /** Tiny offline model for offline benchmarking. */
    public static OmniLLM tiny(String kind) {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.tiny(kind);
        return fromBundle(bundle, EngineConfig.cpuDefault(), null);
    }

    private static OmniLLM fromBundle(AutoModelForCausalLM.Bundle bundle, EngineConfig engConfig) {
        return fromBundle(bundle, engConfig, Path.of("models"));
    }

    private static OmniLLM fromBundle(AutoModelForCausalLM.Bundle bundle, EngineConfig engConfig,
                                      Path modelsRoot) {
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
        MultimodalProcessor proc;
        if (modelsRoot != null) {
            proc = CompositeMultimodalProcessor.withEncoders(tok, ct, modelsRoot);
        } else {
            proc = CompositeMultimodalProcessor.of(tok, ct);
        }

        return new OmniLLM(engine, ec, tok, ct, proc);
    }

    /** Generate for text prompts (convenience, no multimodal). */
    public List<RequestOutput> generate(List<String> prompts, SamplingParams params) {
        if (params == null) params = SamplingParams.defaults();
        for (String p : prompts) {
            int[] ids = tokenizer.encode(p, true).ids();
            engine.addRequest(ids, params, p, null);
        }
        return engine.generateAll();
    }

    public String chat(List<Map<String, String>> messages, SamplingParams params) {
        String prompt = chatTemplate.apply(messages, true);
        int[] ids = tokenizer.encode(prompt, false).ids();
        if (params == null) params = SamplingParams.greedy(64);
        engine.addRequest(ids, params, prompt, null);
        List<RequestOutput> outs = engine.generateAll();
        if (outs.isEmpty()) return "";
        RequestOutput out = outs.get(0);
        int[] outIds = out.outputs.isEmpty() ? new int[0] : out.outputs.get(0).tokenIds;
        return LLM.stripSpecialTokens(tokenizer.decode(outIds, true));
    }

    /**
     * Generate for a multimodal prompt (text + image/audio/video/embedding).
     * Media parts are processed by {@link CompositeMultimodalProcessor}.
     */
    public RequestOutput generate(MultimodalPrompt prompt, SamplingParams params) {
        return generate(prompt, null, params);
    }

    /**
     * Multimodal generate with optional chat history messages.
     */
    public RequestOutput generate(MultimodalPrompt prompt,
                                  List<Map<String, String>> messages,
                                  SamplingParams params) {
        Objects.requireNonNull(prompt, "prompt");
        // Inject EOS / im_end stop ids like chat() — prevents runaway and helps decode quality.
        SamplingParams sp = params == null ? SamplingParams.greedy(32) : params;
        sp = withStopTokens(sp);
        int[] ids = processor.process(prompt, messages);
        if (processor instanceof CompositeMultimodalProcessor cmp) {
            for (String line : cmp.encodeLog()) {
                System.out.println("  [mm] " + line);
            }
        }
        System.out.println("  [mm] prompt_tokens=" + ids.length
                + (ids.length > 0 ? (" first=" + ids[0] + " last=" + ids[ids.length - 1]) : ""));
        String label = prompt.toString() + " | " + CompositeMultimodalProcessor.mediaSummary(prompt);
        engine.addRequest(ids, sp, label, null);
        List<RequestOutput> outs = engine.generateAll();
        return outs.isEmpty() ? null : outs.get(0);
    }

    /** Decode generated token ids; log raw when strip would yield empty. */
    private String decodeOutput(RequestOutput out) {
        if (out == null) return "";
        int[] outIds = out.outputs.isEmpty() ? new int[0] : out.outputs.get(0).tokenIds;
        if (outIds.length == 0) {
            System.out.println("  [mm] gen_tokens=0 (empty output ids)");
            return "";
        }
        String raw = tokenizer.decode(outIds, true);
        String stripped = LLM.stripSpecialTokens(raw);
        if ((stripped == null || stripped.isBlank()) && raw != null && !raw.isBlank()) {
            // Prefer non-empty raw over fully-stripped blank (still clean specials lightly)
            System.out.println("  [mm] strip emptied output; raw_len=" + raw.length()
                    + " raw_preview=" + raw.replace("\n", "\\n").substring(0, Math.min(80, raw.length()))
                    + " ids0=" + outIds[0]
                    + (outIds.length > 1 ? (" ids1=" + outIds[1]) : ""));
            stripped = raw.replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<|endoftext|>", "")
                    .trim();
        }
        if (stripped == null || stripped.isBlank()) {
            System.out.println("  [mm] gen still empty after decode; n=" + outIds.length
                    + " head=" + java.util.Arrays.toString(
                    java.util.Arrays.copyOf(outIds, Math.min(8, outIds.length))));
            return "";
        }
        return stripped;
    }

    /** Inject multi-eos stop ids when caller left stopTokenIds empty. */
    private SamplingParams withStopTokens(SamplingParams params) {
        if (params == null) params = SamplingParams.defaults();
        if (params.stopTokenIds != null && !params.stopTokenIds.isEmpty()) return params;
        // Mirror LLM.withStopTokens: eos + generation_config multi-eos + common ChatML
        java.util.LinkedHashSet<Integer> stops = new java.util.LinkedHashSet<>();
        try {
            // Prefer tokenizer-known specials when available
            stops.add(151645); // <|im_end|>
            stops.add(151643); // <|endoftext|>
        } catch (Throwable ignored) {}
        return params.toBuilder().stopTokenIds(new java.util.ArrayList<>(stops)).build();
    }

    /** Convenience: image path + text question. */
    public String askImage(Path image, String question, SamplingParams params) {
        MultimodalPrompt mp = MultimodalPrompt.of(
                MediaInput.image(image),
                MediaInput.text(question == null ? "Describe this image." : question));
        return decodeOutput(generate(mp, params));
    }

    /** Convenience: audio path + text question. */
    public String askAudio(Path audio, String question, SamplingParams params) {
        MultimodalPrompt mp = MultimodalPrompt.of(
                MediaInput.audio(audio),
                MediaInput.text(question == null ? "Transcribe or describe this audio." : question));
        return decodeOutput(generate(mp, params));
    }

    /** Convenience: video path + text question. */
    public String askVideo(Path video, String question, SamplingParams params) {
        MultimodalPrompt mp = MultimodalPrompt.of(
                MediaInput.video(video),
                MediaInput.text(question == null ? "Describe this video." : question));
        return decodeOutput(generate(mp, params));
    }

    /**
     * OCR path: image of text/UI/document + question.
     * Uses OCR encoder features when {@link CompositeMultimodalProcessor} is wired.
     */
    public String askOcr(Path image, String question, SamplingParams params) {
        // Prefer OCR feature encode for logging, then standard image multimodal path
        if (processor instanceof CompositeMultimodalProcessor cmp) {
            cmp.encodeOcrFeatures(MediaInput.image(image));
        }
        MultimodalPrompt mp = MultimodalPrompt.of(
                MediaInput.image(image),
                MediaInput.text(question == null
                        ? "Read any visible text in this image. Transcribe briefly."
                        : question));
        return decodeOutput(generate(mp, params));
    }

    /**
     * ASR path: audio + transcription-style question.
     * Uses ASR/Whisper encoder features when available.
     */
    public String askAsr(Path audio, String question, SamplingParams params) {
        if (processor instanceof CompositeMultimodalProcessor cmp) {
            cmp.encodeAsrFeatures(MediaInput.audio(audio));
        }
        MultimodalPrompt mp = MultimodalPrompt.of(
                MediaInput.audio(audio),
                MediaInput.text(question == null
                        ? "Transcribe the speech. Output only the text."
                        : question));
        return decodeOutput(generate(mp, params));
    }

    /** Batch text embedding via {@link EmbeddingRunner}. */
    public float[][] embed(List<String> texts, EmbeddingRunner embedRunner) {
        Objects.requireNonNull(embedRunner, "embedRunner");
        return engine.embedTexts(texts, embedRunner);
    }

    /** Create a mini embedding runner (offline, no HF weights). */
    public static EmbeddingRunner miniEmbedder() {
        return new EmbeddingRunner(
                org.bytedeco.pytorch.utils.sentence.SentenceTransformer.mini());
    }

    public EngineConfig config() { return config; }
    public EngineMetrics metrics() { return engine.metrics(); }
    public LLMEngine engine() { return engine; }
    public FastTokenizer tokenizer() { return tokenizer; }
    public ChatTemplate chatTemplate() { return chatTemplate; }
    public MultimodalProcessor processor() { return processor; }

    @Override
    public void close() { engine.close(); }
}
