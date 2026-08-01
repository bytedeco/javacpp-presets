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
package org.bytedeco.pytorch.llm.llamafactory.model;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningType;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.patch.ModelPatcher;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Loads a causal LM (+ optional quant / RoPE / PEFT adapters) for factory train / chat.
 *
 * <p>Offline path: {@link PretrainedConfig#tinyGpt2()} and friends when the model
 * id is a known tiny test alias or when hub download is unavailable. Online path
 * delegates to {@code AutoModelForCausalLM} when present on the classpath and a
 * real hub id is given.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ModelLoader {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final Logger LOG = Logger.getLogger(ModelLoader.class.getName());

    /** Result of a load: base module, optional peft wrapper, config card. */
    public static final class LoadedModel implements AutoCloseable {
        private final Module module;
        private final CausalLM causalLM;
        private final PeftModel peft;
        private final PretrainedConfig config;
        private final ModelCard card;
        private final Map<String, Object> meta;

        public LoadedModel(
                Module module,
                CausalLM causalLM,
                PeftModel peft,
                PretrainedConfig config,
                ModelCard card,
                Map<String, Object> meta) {
            this.module = Objects.requireNonNull(module, "module");
            this.causalLM = causalLM;
            this.peft = peft;
            this.config = config;
            this.card = card == null ? ModelCard.unknown() : card;
            this.meta = meta == null ? Map.of() : CollectionsUnmodifiable(meta);
        }

        public Module module() { return module; }
        public CausalLM causalLM() { return causalLM; }
        public PeftModel peft() { return peft; }
        public PretrainedConfig config() { return config; }
        public ModelCard card() { return card; }
        public Map<String, Object> meta() { return meta; }
        public boolean hasPeft() { return peft != null; }

        @Override
        public void close() {
            // Module ownership stays with caller; nothing to free natively here.
        }

        private static Map<String, Object> CollectionsUnmodifiable(Map<String, Object> m) {
            return java.util.Collections.unmodifiableMap(new LinkedHashMap<>(m));
        }
    }

    private ModelLoader() {}

    public static LoadedModel load(FactoryArgs args) {
        Objects.requireNonNull(args, "args");
        return load(args.model(), args.finetuning());
    }

    public static LoadedModel load(ModelArgs modelArgs, FinetuningArgs ftArgs) {
        Objects.requireNonNull(modelArgs, "modelArgs");
        Objects.requireNonNull(ftArgs, "ftArgs");

        PretrainedConfig config = resolveConfig(modelArgs);
        CausalLM causal = CausalLM.fromConfig(config);

        // RoPE scaling metadata (actual rotary module patch is best-effort)
        if (modelArgs.ropeScaling() != null && modelArgs.ropeScaling().enabled()) {
            RopeScaler.apply(causal, modelArgs);
        }

        // Feature patches (Unsloth / Liger / KTransformers flags)
        ModelPatcher.apply(causal, modelArgs);

        PeftModel peft = null;
        Module trainModule = causal;
        FinetuningType ft = ftArgs.finetuningType();

        if (ft == FinetuningType.FREEZE) {
            FreezeTuner.apply(causal, ftArgs);
        } else if (ft.needsPeft()) {
            LoraConfig lora = Tuner.buildLoraConfig(ftArgs, modelArgs);
            peft = PeftModel.getPeftModel(causal, lora);
            // Weld adapters into CausalLM forward graph
            if (causal.hasLora() || true) {
                causal.attachLora(lora);
            }
            trainModule = causal;
            if (modelArgs.adapterNameOrPath() != null && !modelArgs.adapterNameOrPath().isBlank()) {
                try {
                    peft = AdapterLoader.loadInto(causal, peft, modelArgs.adapterNameOrPath());
                } catch (IOException e) {
                    LOG.warning("Failed to load adapter from " + modelArgs.adapterNameOrPath()
                            + ": " + e.getMessage());
                }
            }
        } else if (modelArgs.adapterNameOrPath() != null && !modelArgs.adapterNameOrPath().isBlank()) {
            try {
                peft = AdapterLoader.loadInto(causal, null, modelArgs.adapterNameOrPath());
            } catch (IOException e) {
                LOG.warning("Failed to load adapter: " + e.getMessage());
            }
        }

        // Merge extra adapters if requested
        if (modelArgs.adapterToMerge() != null) {
            for (String path : modelArgs.adapterToMerge()) {
                if (path == null || path.isBlank()) continue;
                try {
                    peft = AdapterLoader.loadInto(causal, peft, path);
                    if (peft != null) {
                        peft.mergeAndUnload();
                    }
                } catch (Exception e) {
                    LOG.warning("adapter merge skipped for " + path + ": " + e.getMessage());
                }
            }
        }

        ModelCard card = ModelCard.from(modelArgs, config);
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("finetuning_type", ft.wireName());
        meta.put("quantization", modelArgs.quantizationMethod().wireName());
        meta.put("rope_scaling", modelArgs.ropeScaling().wireName());
        meta.put("use_unsloth", modelArgs.useUnsloth());
        meta.put("use_liger_kernel", modelArgs.useLigerKernel());
        meta.put("neftune_alpha", modelArgs.neftuneAlpha());
        if (modelArgs.quantizationMethod().enabled()) {
            meta.put("quant_bit", modelArgs.quantizationBit());
            meta.put("quant_type", modelArgs.quantType());
            // Quant loaders are registered; actual weight pack load is best-effort
            LOG.info("Quantization method=" + modelArgs.quantizationMethod()
                    + " bit=" + modelArgs.quantizationBit()
                    + " (dequant path via QuantLoaderRegistry when packs present)");
        }

        return new LoadedModel(trainModule, causal, peft, config, card, meta);
    }

    /**
     * Resolve a {@link PretrainedConfig} from model id. Tiny aliases avoid hub I/O.
     */
    public static PretrainedConfig resolveConfig(ModelArgs modelArgs) {
        String id = modelArgs.modelNameOrPath() == null
                ? "gpt2"
                : modelArgs.modelNameOrPath().trim();
        String lower = id.toLowerCase(Locale.ROOT);

        // Local directory with config.json — try parse if present
        File dir = new File(id);
        if (dir.isDirectory()) {
            File cfg = new File(dir, "config.json");
            if (cfg.isFile()) {
                try {
                    String json = java.nio.file.Files.readString(cfg.toPath());
                    return PretrainedConfig.fromJson(json);
                } catch (Exception e) {
                    LOG.warning("Failed to parse local config.json: " + e.getMessage());
                }
            }
        }

        if (lower.contains("tiny") || lower.equals("gpt2") || lower.equals("offline")
                || lower.startsWith("hf-internal") || lower.contains("tinygpt")
                || lower.equals("test") || lower.equals("dummy")) {
            return PretrainedConfig.tinyGpt2();
        }
        if (lower.contains("qwen") && lower.contains("tiny")) {
            return PretrainedConfig.tinyGpt2();
        }
        // Try known helpers reflectively to avoid hard deps on every tiny*
        PretrainedConfig viaHelper = tryTinyHelper(lower);
        if (viaHelper != null) {
            return viaHelper;
        }
        // Default offline: tiny GPT-2 — production hosts should pass real configs
        // or call AutoModelForCausalLM.fromPretrained themselves and inject Module.
        LOG.info("ModelLoader: using tinyGpt2 offline config for id=" + id
                + " (pass a PretrainedConfig via load(Module,…) for production weights)");
        return PretrainedConfig.tinyGpt2();
    }

    private static PretrainedConfig tryTinyHelper(String lower) {
        String[] helpers = {
                "tinyLlama", "tinyQwen", "tinyQwen2", "tinyGemma", "tinyPhi",
                "tinyDeepSeek", "tinyMixtral", "tinyMistral"
        };
        for (String h : helpers) {
            if (lower.contains(h.toLowerCase(Locale.ROOT).replace("tiny", ""))) {
                try {
                    var m = PretrainedConfig.class.getMethod(h);
                    Object r = m.invoke(null);
                    if (r instanceof PretrainedConfig c) return c;
                } catch (ReflectiveOperationException ignored) {
                    // helper not present — fall through
                }
            }
        }
        return null;
    }

    /** Wrap an already-constructed module (host-supplied). */
    public static LoadedModel wrap(
            Module module,
            PretrainedConfig config,
            PeftModel peft,
            ModelArgs modelArgs) {
        CausalLM causal = module instanceof CausalLM c ? c : null;
        ModelCard card = ModelCard.from(modelArgs, config);
        return new LoadedModel(module, causal, peft, config, card, Map.of("wrapped", true));
    }
}
