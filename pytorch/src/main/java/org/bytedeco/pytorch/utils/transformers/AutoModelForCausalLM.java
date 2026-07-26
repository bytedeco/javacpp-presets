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
package org.bytedeco.pytorch.utils.transformers;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.utils.bitsandbytes.BitsAndBytes;
import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.utils.transformers.generation.Generator;
import org.bytedeco.pytorch.utils.transformers.loading.SnapshotFiles;
import org.bytedeco.pytorch.utils.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.utils.transformers.mapping.ModelRegistry;
import org.bytedeco.pytorch.utils.transformers.mapping.WeightMap;
import org.bytedeco.pytorch.utils.transformers.tokenization.ChatTemplate;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code AutoModelForCausalLM.from_pretrained} entry point.
 *
 * <p>Resolves architecture via {@link ModelRegistry}, zero-copy-binds safetensors
 * via {@link WeightLoader}, and pairs with {@link FastTokenizer}.
 *
 * <pre>{@code
 * AutoModelForCausalLM.Bundle b = AutoModelForCausalLM.fromPretrained(
 *     "Qwen/Qwen2-0.5B-Instruct", hub);
 * String reply = b.chat(List.of(
 *     Map.of("role","user","content","What is 2+2?")));
 * }</pre>
 */
public final class AutoModelForCausalLM {

    private AutoModelForCausalLM() {}

    /** Loaded model + tokenizer + configs + load report (+ optional bnb quant state). */
    public static final class Bundle {
        private final Module model;
        private final FastTokenizer tokenizer;
        private final PretrainedConfig config;
        private final GenerationConfig generationConfig;
        private final Path snapshot;
        private final WeightLoader.LoadReport loadReport;
        private final ChatTemplate chatTemplate;
        private final BitsAndBytesConfig quantizationConfig;
        private final BitsAndBytes.QuantizedModel quantizedModel;

        public Bundle(Module model, FastTokenizer tokenizer, PretrainedConfig config,
                      GenerationConfig generationConfig, Path snapshot,
                      WeightLoader.LoadReport loadReport, ChatTemplate chatTemplate) {
            this(model, tokenizer, config, generationConfig, snapshot, loadReport, chatTemplate, null, null);
        }

        public Bundle(Module model, FastTokenizer tokenizer, PretrainedConfig config,
                      GenerationConfig generationConfig, Path snapshot,
                      WeightLoader.LoadReport loadReport, ChatTemplate chatTemplate,
                      BitsAndBytesConfig quantizationConfig,
                      BitsAndBytes.QuantizedModel quantizedModel) {
            this.model = Objects.requireNonNull(model);
            this.tokenizer = Objects.requireNonNull(tokenizer);
            this.config = Objects.requireNonNull(config);
            this.generationConfig = generationConfig == null ? GenerationConfig.greedy() : generationConfig;
            this.snapshot = snapshot;
            this.loadReport = loadReport;
            this.chatTemplate = chatTemplate == null ? ChatTemplate.qwen() : chatTemplate;
            this.quantizationConfig = quantizationConfig;
            this.quantizedModel = quantizedModel;
        }

        public Module model() { return model; }
        public FastTokenizer tokenizer() { return tokenizer; }
        public PretrainedConfig config() { return config; }
        public GenerationConfig generationConfig() { return generationConfig; }
        public Path snapshot() { return snapshot; }
        public WeightLoader.LoadReport loadReport() { return loadReport; }
        public ChatTemplate chatTemplate() { return chatTemplate; }
        public BitsAndBytesConfig quantizationConfig() { return quantizationConfig; }
        public BitsAndBytes.QuantizedModel quantizedModel() { return quantizedModel; }
        public boolean isQuantized() { return quantizedModel != null; }

        /** Encode raw prompt and greedy/sample generate. */
        public String generate(String prompt, GenerationConfig gen) {
            var enc = tokenizer.encode(prompt, true);
            GenerationConfig g = mergeGen(gen);
            int[] out = Generator.generate(model, enc.ids(), g, config.maxPositionEmbeddings());
            // decode only the newly generated portion when possible
            int promptLen = enc.ids().length;
            if (out.length > promptLen) {
                int[] neu = new int[out.length - promptLen];
                System.arraycopy(out, promptLen, neu, 0, neu.length);
                return tokenizer.decode(neu, true);
            }
            return tokenizer.decode(out, true);
        }

        public String generate(String prompt, int maxNewTokens) {
            return generate(prompt, generationConfig.toBuilder().maxNewTokens(maxNewTokens).build());
        }

        /** Apply chat template then generate (Instruct models). */
        public String chat(List<Map<String, String>> messages, GenerationConfig gen) {
            String prompt = chatTemplate.apply(messages, /*addGenerationPrompt=*/true);
            // Template already embeds BOS/specials — do not double-add via post-processor.
            return generateEncoded(prompt, gen, /*addSpecialTokens=*/false);
        }

        /** Encode + generate with explicit add_special_tokens control. */
        public String generateEncoded(String prompt, GenerationConfig gen, boolean addSpecialTokens) {
            var enc = tokenizer.encode(prompt, addSpecialTokens);
            GenerationConfig g = mergeGen(gen);
            int[] out = Generator.generate(model, enc.ids(), g, config.maxPositionEmbeddings());
            int promptLen = enc.ids().length;
            if (out.length > promptLen) {
                int[] neu = new int[out.length - promptLen];
                System.arraycopy(out, promptLen, neu, 0, neu.length);
                return tokenizer.decode(neu, true);
            }
            return tokenizer.decode(out, true);
        }

        public String chat(List<Map<String, String>> messages) {
            return chat(messages, generationConfig);
        }

        private GenerationConfig mergeGen(GenerationConfig gen) {
            GenerationConfig base = generationConfig;
            if (gen == null) gen = base;
            GenerationConfig.Builder b = gen.toBuilder();
            if (gen.eosTokenIds.isEmpty()) {
                b.eosTokenId(config.eosTokenId());
                for (int id : base.eosTokenIds) b.eosTokenId(id);
            }
            return b.build();
        }
    }

    public static final class LoadOptions {
        public WeightLoader.BindMode bindMode = WeightLoader.BindMode.ZERO_COPY;
        public boolean strict = true;
        public boolean zeroCopyMmap = true;
        public boolean loadWeights = true;
        /** Optional HF-style BitsAndBytes quantization (4/8-bit). */
        public BitsAndBytesConfig quantizationConfig;
        /** Freeze base weights after quant (QLoRA prepare). Default true when quant is set. */
        public boolean prepareForKbitTraining = true;

        public LoadOptions bindMode(WeightLoader.BindMode m) { this.bindMode = m; return this; }
        public LoadOptions strict(boolean v) { this.strict = v; return this; }
        public LoadOptions zeroCopyMmap(boolean v) { this.zeroCopyMmap = v; return this; }
        public LoadOptions loadWeights(boolean v) { this.loadWeights = v; return this; }
        public LoadOptions quantizationConfig(BitsAndBytesConfig cfg) {
            this.quantizationConfig = cfg;
            return this;
        }
        /** Snake alias matching Python {@code quantization_config=}. */
        public LoadOptions quantization_config(BitsAndBytesConfig cfg) {
            return quantizationConfig(cfg);
        }
        public LoadOptions prepareForKbitTraining(boolean v) {
            this.prepareForKbitTraining = v;
            return this;
        }
    }

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub, new LoadOptions());
    }

    public static Bundle fromPretrained(String modelId, HfHub hub, LoadOptions opts) throws IOException {
        Objects.requireNonNull(modelId, "modelId");
        Objects.requireNonNull(hub, "hub");
        Path snap = hub.snapshotDownload(modelId);
        return fromDirectory(snap, opts);
    }

    public static Bundle fromDirectory(Path dir) throws IOException {
        return fromDirectory(dir, new LoadOptions());
    }

    public static Bundle fromDirectory(Path dir, LoadOptions opts) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (opts == null) opts = new LoadOptions();
        if (!Files.isDirectory(dir)) {
            throw new IOException("Not a model directory: " + dir);
        }

        PretrainedConfig cfg = readConfig(dir);
        Module model = ModelRegistry.create(cfg);
        model.eval();

        // Convert model to the dtype specified in config (e.g., bfloat16 from HF safetensors)
        // Must be done BEFORE loading weights so storage sizes match
        String dtypeStr = cfg.torchDtype();
        System.out.println("[DEBUG] config torch_dtype = " + dtypeStr);
        boolean needsDtypeConversion = dtypeStr != null && !dtypeStr.isEmpty() && !"float32".equals(dtypeStr);
        if (needsDtypeConversion) {
            try {
                var scalarType = switch (dtypeStr.toLowerCase()) {
                    case "bfloat16", "bf16" -> org.bytedeco.pytorch.global.torch.ScalarType.BFloat16;
                    case "float16", "fp16", "half" -> org.bytedeco.pytorch.global.torch.ScalarType.Half;
                    case "float", "float32" -> org.bytedeco.pytorch.global.torch.ScalarType.Float;
                    case "double", "float64" -> org.bytedeco.pytorch.global.torch.ScalarType.Double;
                    case "int32" -> org.bytedeco.pytorch.global.torch.ScalarType.Int;
                    case "int64", "long" -> org.bytedeco.pytorch.global.torch.ScalarType.Long;
                    default -> null;
                };
                System.out.println("[DEBUG] converting to scalarType = " + scalarType);
                if (scalarType != null) {
                    org.bytedeco.pytorch.utils.transformers.loading.SnapshotFiles.toDtype(model, scalarType);
                }
            } catch (Throwable t) {
                System.out.println("[DEBUG] dtype conversion failed: " + t.getMessage());
                // dtype conversion failed, continue with default dtype
            }
        }

        WeightLoader.LoadReport report = null;
        if (opts.loadWeights) {
            WeightMap map = ModelRegistry.weightMap(cfg);
            // When dtype conversion was needed, we must use COPY mode since ZERO_COPY
            // cannot rebind storage between tensors of different dtypes
            var bindMode = needsDtypeConversion ? WeightLoader.BindMode.COPY : opts.bindMode;
            if (needsDtypeConversion) {
                System.out.println("[DEBUG] Using COPY mode due to dtype conversion");
            }
            // When tie_word_embeddings is true, lm_head.weight is tied to embed_tokens.weight
            // so we allow strict to pass even if lm_head.weight is "missing"
            boolean allowTiedMissing = cfg.tieWordEmbeddings();
            try {
                report = WeightLoader.loadAndBind(model, dir, map, bindMode, opts.strict && !allowTiedMissing, opts.zeroCopyMmap);
            } catch (IOException e) {
                if (opts.strict) throw e;
                // no weights — leave random init
                report = new WeightLoader.LoadReport(
                        List.of(), List.of("(no safetensors)"), List.of(), List.of(), 0, 0, bindMode);
            }
            // After COPY mode loading with tie_word_embeddings, re-apply the tie
            // because COPY mode creates new tensors breaking the constructor's set_() binding
            if (cfg.tieWordEmbeddings() && bindMode == WeightLoader.BindMode.COPY) {
                System.out.println("[DEBUG] Re-applying tie_word_embeddings after COPY load");
                try {
                    var qwen = (org.bytedeco.pytorch.utils.transformers.modeling.Qwen2ForCausalLM) model;
                    qwen.lmHead().weight().set_(qwen.model().embed_tokens.weight());
                } catch (Throwable t) {
                    System.out.println("[DEBUG] Failed to re-apply tie: " + t.getMessage());
                }
            }
        }

        FastTokenizer tok = readTokenizer(dir, cfg);
        GenerationConfig genCfg = readGenerationConfig(dir, cfg);
        ChatTemplate template = ChatTemplate.detect(dir, cfg);

        BitsAndBytes.QuantizedModel qm = applyQuantization(model, opts);
        return new Bundle(model, tok, cfg, genCfg, dir, report, template,
                opts.quantizationConfig, qm);
    }

    /** Random-init tiny model for offline unit tests (no weights). */
    public static Bundle tiny(String kind) {
        return tiny(kind, null);
    }

    /** Random-init tiny model with optional BitsAndBytes quantization (QLoRA offline path). */
    public static Bundle tiny(String kind, BitsAndBytesConfig bnb) {
        PretrainedConfig cfg = switch (kind == null ? "qwen" : kind.toLowerCase()) {
            case "llama", "mistral" -> PretrainedConfig.tinyLlama();
            case "qwen", "qwen2" -> PretrainedConfig.tinyQwen();
            default -> PretrainedConfig.tinyGpt2();
        };
        Module model = ModelRegistry.create(cfg);
        BitsAndBytes.QuantizedModel qm = null;
        if (bnb != null && bnb.isQuantized()) {
            LoadOptions opts = new LoadOptions()
                    .quantizationConfig(bnb)
                    .prepareForKbitTraining(true);
            qm = applyQuantization(model, opts);
        }
        FastTokenizer tok = FastTokenizer.whitespace()
                .modelMaxLength(cfg.maxPositionEmbeddings())
                .build();
        GenerationConfig gen = GenerationConfig.builder()
                .maxNewTokens(16)
                .eosTokenId(cfg.eosTokenId())
                .build();
        return new Bundle(model, tok, cfg, gen, null, null,
                ChatTemplate.forModelType(cfg.modelType()), bnb, qm);
    }

    /**
     * Apply BitsAndBytes 4/8-bit quantization to a loaded model when
     * {@link LoadOptions#quantizationConfig} is set.
     *
     * <p>Collects HF-named linears from {@link CausalLM},
     * {@link org.bytedeco.pytorch.utils.transformers.modeling.Qwen2ForCausalLM},
     * and {@link org.bytedeco.pytorch.utils.transformers.modeling.LlamaForCausalLM},
     * then quantize→materialize→freeze (QLoRA prepare).
     */
    public static BitsAndBytes.QuantizedModel applyQuantization(Module model, LoadOptions opts) {
        if (opts == null || opts.quantizationConfig == null || !opts.quantizationConfig.isQuantized()) {
            return null;
        }
        BitsAndBytesConfig bnb = opts.quantizationConfig;
        BitsAndBytes.QuantizedModel qm = null;
        try {
            Map<String, LinearImpl> linears = collectQuantizableLinears(model);
            if (!linears.isEmpty()) {
                qm = BitsAndBytes.prepareForQLoRA(linears, bnb);
            } else if (opts.prepareForKbitTraining) {
                BitsAndBytes.prepareModelForKbitTraining(model);
            }
        } catch (Exception e) {
            System.out.println("[bnb] quantization skipped: " + e.getMessage());
        }
        if (opts.prepareForKbitTraining && model != null) {
            try {
                BitsAndBytes.prepareModelForKbitTraining(model);
            } catch (Exception ignored) {}
        }
        return qm;
    }

    /**
     * Collect quantizable LinearImpls (excludes lm_head) from known causal LM layouts.
     */
    public static Map<String, LinearImpl> collectQuantizableLinears(Module model) {
        java.util.LinkedHashMap<String, LinearImpl> m = new java.util.LinkedHashMap<>();
        if (model == null) return m;
        if (model instanceof CausalLM clm) {
            return clm.quantizableLinears();
        }
        if (model instanceof org.bytedeco.pytorch.utils.transformers.modeling.Qwen2ForCausalLM qwen) {
            var layers = qwen.model().layers;
            for (int i = 0; i < layers.size(); i++) {
                var layer = layers.get(i);
                String p = "model.layers." + i;
                var attn = layer.self_attn;
                m.put(p + ".self_attn.q_proj", attn.q_proj);
                m.put(p + ".self_attn.k_proj", attn.k_proj);
                m.put(p + ".self_attn.v_proj", attn.v_proj);
                m.put(p + ".self_attn.o_proj", attn.o_proj);
                m.put(p + ".mlp.gate_proj", layer.mlp.gate_proj);
                m.put(p + ".mlp.up_proj", layer.mlp.up_proj);
                m.put(p + ".mlp.down_proj", layer.mlp.down_proj);
            }
            return m;
        }
        if (model instanceof org.bytedeco.pytorch.utils.transformers.modeling.LlamaForCausalLM llama) {
            // LlamaForCausalLM exposes model() similarly — use reflection-safe path via fields
            try {
                var modelMethod = llama.getClass().getMethod("model");
                Object inner = modelMethod.invoke(llama);
                @SuppressWarnings("unchecked")
                var layers = (java.util.List<?>) inner.getClass().getField("layers").get(inner);
                for (int i = 0; i < layers.size(); i++) {
                    Object layer = layers.get(i);
                    String p = "model.layers." + i;
                    Object attn = layer.getClass().getField("self_attn").get(layer);
                    Object mlp = layer.getClass().getField("mlp").get(layer);
                    m.put(p + ".self_attn.q_proj", (LinearImpl) attn.getClass().getField("q_proj").get(attn));
                    m.put(p + ".self_attn.k_proj", (LinearImpl) attn.getClass().getField("k_proj").get(attn));
                    m.put(p + ".self_attn.v_proj", (LinearImpl) attn.getClass().getField("v_proj").get(attn));
                    m.put(p + ".self_attn.o_proj", (LinearImpl) attn.getClass().getField("o_proj").get(attn));
                    m.put(p + ".mlp.gate_proj", (LinearImpl) mlp.getClass().getField("gate_proj").get(mlp));
                    m.put(p + ".mlp.up_proj", (LinearImpl) mlp.getClass().getField("up_proj").get(mlp));
                    m.put(p + ".mlp.down_proj", (LinearImpl) mlp.getClass().getField("down_proj").get(mlp));
                }
            } catch (Exception ignored) {}
            return m;
        }
        return m;
    }

    private static PretrainedConfig readConfig(Path dir) throws IOException {
        Path cfg = SnapshotFiles.configJson(dir);
        if (Files.isRegularFile(cfg)) {
            return PretrainedConfig.fromFile(cfg);
        }
        throw new IOException("Missing config.json in " + dir);
    }

    private static GenerationConfig readGenerationConfig(Path dir, PretrainedConfig cfg) {
        Path p = SnapshotFiles.generationConfigJson(dir);
        try {
            if (Files.isRegularFile(p)) {
                GenerationConfig g = GenerationConfig.fromFile(p);
                if (g.eosTokenIds.isEmpty()) {
                    return g.toBuilder().eosTokenId(cfg.eosTokenId()).build();
                }
                return g;
            }
        } catch (IOException ignored) {}
        return GenerationConfig.builder()
                .maxNewTokens(64)
                .eosTokenId(cfg.eosTokenId())
                .padTokenId(cfg.padTokenId())
                .bosTokenId(cfg.bosTokenId())
                .build();
    }

    private static FastTokenizer readTokenizer(Path dir, PretrainedConfig cfg) throws IOException {
        // Full HF tokenizer.json / vocab+merges / whitespace fallback
        FastTokenizer tok = org.bytedeco.pytorch.utils.tokenizers.DirectoryTokenizerLoader.load(dir);
        if (cfg != null && cfg.maxPositionEmbeddings() > 0
                && tok.modelMaxLength() <= 0) {
            return tok.withTruncation(FastTokenizer.Truncation.of(cfg.maxPositionEmbeddings()));
        }
        return tok;
    }
}
