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
package org.bytedeco.pytorch.llm.ktransformers.adapter;

import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtModelFamily;
import org.bytedeco.pytorch.llm.ktransformers.config.KtSftConfig;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Bridges {@code kt_*} flat keys ↔ {@link KtConfig} / FactoryArgs maps.
 *
 * <p>Unknown keys are preserved so LLaMA-Factory hosts can pass mixed maps.
 * Documented keys (non-exhaustive):
 * <ul>
 *   <li>{@code kt_model_family}, {@code kt_max_steps}, {@code kt_lora_r}</li>
 *   <li>{@code kt_visual_board}, {@code kt_tensorboard}, {@code kt_stage}</li>
 *   <li>{@code kt_num_experts}, {@code kt_top_k}, {@code kt_quant_bits}</li>
 * </ul>
 */
public final class KtFactoryHparamsExtension {

    private KtFactoryHparamsExtension() {}

    /** Known kt_* key names for docs / validation UIs. */
    public static final String[] KNOWN_KEYS = {
            "kt_model_family",
            "kt_max_steps",
            "kt_lora_r",
            "kt_lora_alpha",
            "kt_lora_dropout",
            "kt_visual_board",
            "kt_tensorboard",
            "kt_wandb",
            "kt_stage",
            "kt_peft",
            "kt_learning_rate",
            "kt_batch_size",
            "kt_num_experts",
            "kt_top_k",
            "kt_quant_bits",
            "kt_enable_monitor",
            "kt_output_dir",
            "kt_hidden_size",
            "kt_num_layers",
            "kt_vocab_size"
    };

    /**
     * Merge optional kt map into a factory flat map (factory keys win on collision
     * only when both set non-kt names; kt_* always written from {@code kt}).
     */
    public static Map<String, Object> mergeIntoFactoryMap(Map<String, ?> raw) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (raw != null) {
            for (Map.Entry<String, ?> e : raw.entrySet()) {
                if (e.getKey() != null) out.put(e.getKey(), e.getValue());
            }
        }
        // Ensure stage/finetuning defaults exist for FactoryArgs.parse
        out.putIfAbsent("model_name_or_path",
                firstString(out, "kt-mini-moe", "model_name_or_path", "model"));
        out.putIfAbsent("stage", mapStage(out.get("kt_stage"), out.get("stage")));
        out.putIfAbsent("finetuning_type", mapPeft(out.get("kt_peft"), out.get("finetuning_type")));
        out.putIfAbsent("output_dir", firstString(out, "saves/kt", "kt_output_dir", "output_dir"));
        if (out.containsKey("kt_max_steps") && !out.containsKey("max_steps")) {
            out.put("max_steps", out.get("kt_max_steps"));
        }
        if (out.containsKey("kt_learning_rate") && !out.containsKey("learning_rate")) {
            out.put("learning_rate", out.get("kt_learning_rate"));
        }
        if (out.containsKey("kt_lora_r") && !out.containsKey("lora_rank")) {
            out.put("lora_rank", out.get("kt_lora_r"));
        }
        return out;
    }

    /** Emit a factory-parseable flat map from a full {@link KtConfig}. */
    public static Map<String, Object> toFactoryMap(KtConfig config) {
        Objects.requireNonNull(config, "config");
        Map<String, Object> m = new LinkedHashMap<>();
        String name = config.modelNameOrPath() != null ? config.modelNameOrPath() : "kt-model";
        m.put("model_name_or_path", name);
        m.put("kt_model_family", config.modelFamily().name());
        KtSftConfig sft = config.sft();
        m.put("stage", sft.stage().name().toLowerCase());
        m.put("kt_stage", sft.stage().name());
        m.put("finetuning_type", peftWire(sft.peftKind()));
        m.put("kt_peft", sft.peftKind().name());
        m.put("kt_max_steps", sft.maxSteps());
        m.put("max_steps", sft.maxSteps());
        m.put("kt_lora_r", sft.loraR());
        m.put("lora_rank", Math.max(1, sft.loraR()));
        m.put("kt_lora_alpha", sft.loraAlpha());
        m.put("kt_lora_dropout", sft.loraDropout());
        m.put("kt_learning_rate", sft.learningRate());
        m.put("learning_rate", sft.learningRate());
        m.put("kt_batch_size", sft.batchSize());
        m.put("kt_visual_board", sft.visualBoard());
        m.put("kt_tensorboard", sft.tensorboard());
        m.put("kt_wandb", sft.wandb());
        m.put("kt_enable_monitor", config.enableMonitor());
        m.put("kt_hidden_size", config.hiddenSize());
        m.put("kt_num_layers", config.numLayers());
        m.put("kt_vocab_size", config.vocabSize());
        if (config.moe() != null) {
            m.put("kt_num_experts", config.moe().numExperts());
            m.put("kt_top_k", config.moe().topK());
        }
        if (config.quant() != null) {
            m.put("kt_quant_bits", config.quant().effectiveBits());
        }
        if (sft.outputDir() != null) {
            m.put("output_dir", sft.outputDir().toString());
            m.put("kt_output_dir", sft.outputDir().toString());
        } else {
            m.put("output_dir", "saves/kt");
        }
        return m;
    }

    public static boolean isKtKey(String key) {
        return key != null && (key.startsWith("kt_") || key.startsWith("KT_"));
    }

    private static String mapStage(Object ktStage, Object factoryStage) {
        if (factoryStage != null) return String.valueOf(factoryStage).toLowerCase();
        if (ktStage == null) return "sft";
        String s = String.valueOf(ktStage).trim().toLowerCase();
        if (s.isEmpty()) return "sft";
        return s;
    }

    private static String mapPeft(Object ktPeft, Object factoryFt) {
        if (factoryFt != null) return String.valueOf(factoryFt).toLowerCase();
        if (ktPeft == null) return "lora";
        String s = String.valueOf(ktPeft).trim().toLowerCase();
        if (s.contains("qlora")) return "qlora";
        if (s.contains("none") || s.equals("full")) return "full";
        if (s.contains("dora")) return "lora";
        return "lora";
    }

    private static String peftWire(KtSftConfig.PeftKind k) {
        if (k == null) return "lora";
        switch (k) {
            case NONE: return "full";
            case QLORA: return "qlora";
            case DORA:
            case LORA:
            default: return "lora";
        }
    }

    private static String firstString(Map<String, Object> m, String def, String... keys) {
        for (String k : keys) {
            Object v = m.get(k);
            if (v != null) {
                String s = String.valueOf(v).trim();
                if (!s.isEmpty()) return s;
            }
        }
        return def;
    }

    /** Resolve family from mixed factory map. */
    public static KtModelFamily resolveFamily(Map<String, ?> raw) {
        if (raw == null) return KtModelFamily.GENERIC;
        Object f = raw.get("kt_model_family");
        if (f == null) f = raw.get("model_family");
        if (f == null) f = raw.get("model_name_or_path");
        return KtModelFamily.fromString(f != null ? String.valueOf(f) : null);
    }
}
