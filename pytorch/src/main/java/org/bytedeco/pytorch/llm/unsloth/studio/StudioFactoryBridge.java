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
package org.bytedeco.pytorch.llm.unsloth.studio;

import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingStartRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingType;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Bridge between LLaMA-Factory-style {@code llm.factory} args maps and Studio
 * training requests. Hosts that already speak factory hparams can open Studio
 * runs without re-encoding every field.
 *
 * <p>Does not depend on factory classes at compile time (stringly map contract)
 * so partial trees still compile.
 */
public final class StudioFactoryBridge {

    private StudioFactoryBridge() {}

    /**
     * Convert a factory-like argument map into {@link TrainingStartRequest}.
     * Recognises common keys: model_name_or_path, stage, finetuning_type,
     * dataset, output_dir, learning_rate, cutoff_len / max_seq_length, lora_rank,
     * lora_alpha, per_device_train_batch_size, gradient_accumulation_steps,
     * max_steps, num_train_epochs, quantization_bit, gradient_checkpointing.
     */
    public static TrainingStartRequest fromFactoryArgs(Map<String, Object> args) {
        if (args == null) args = Map.of();
        Map<String, Object> m = new LinkedHashMap<>(args);

        String model = firstString(m, "model_name", "model_name_or_path", "model");
        if (model == null) model = "studio/tiny-gpt2";

        String ft = firstString(m, "finetuning_type", "training_type");
        String stage = firstString(m, "stage");
        TrainingType type = TrainingType.LORA_QLORA;
        if (ft != null) {
            try { type = TrainingType.fromLabel(ft); } catch (Exception ignored) {}
        } else if (stage != null) {
            String s = stage.toLowerCase();
            if (s.contains("dpo") || s.contains("kto") || s.contains("ppo") || s.contains("grpo") || s.contains("orpo")) {
                type = TrainingType.REINFORCEMENT_LEARNING;
            } else if (s.contains("pt") || s.contains("pretrain")) {
                type = TrainingType.CONTINUED_PRETRAINING;
            } else if ("full".equalsIgnoreCase(String.valueOf(m.get("finetuning_type")))) {
                type = TrainingType.FULL_FINETUNING;
            }
        }

        TrainingStartRequest.Builder b = TrainingStartRequest.builder()
                .modelName(model)
                .trainingType(type);

        if (m.get("dataset") != null) b.dataset(String.valueOf(m.get("dataset")));
        if (m.get("dataset_dir") != null) b.datasetPath(String.valueOf(m.get("dataset_dir")));
        if (m.get("output_dir") != null) b.outputDir(String.valueOf(m.get("output_dir")));
        if (m.get("learning_rate") != null) {
            b.learningRate(org.bytedeco.pytorch.llm.unsloth.studio.util.Validate.parseLearningRate(m.get("learning_rate")));
        }
        Integer seq = firstInt(m, "max_seq_length", "cutoff_len");
        if (seq != null) b.maxSeqLength(seq);
        Integer rank = firstInt(m, "lora_r", "lora_rank");
        if (rank != null) b.loraR(rank);
        Integer alpha = firstInt(m, "lora_alpha");
        if (alpha != null) b.loraAlpha(alpha);
        Integer batch = firstInt(m, "batch_size", "per_device_train_batch_size");
        if (batch != null) b.batchSize(batch);
        Integer gas = firstInt(m, "gradient_accumulation_steps");
        if (gas != null) b.gradientAccumulationSteps(gas);
        Integer steps = firstInt(m, "max_steps");
        if (steps != null) b.maxSteps(steps);
        if (m.get("num_train_epochs") != null) {
            b.numTrainEpochs(Double.parseDouble(String.valueOf(m.get("num_train_epochs"))));
        }
        Integer bit = firstInt(m, "quantization_bit");
        if (bit != null) {
            if (bit == 4) b.loadIn4bit(true).loadIn8bit(false);
            else if (bit == 8) b.loadIn8bit(true).loadIn4bit(false);
            else b.loadIn4bit(false).loadIn8bit(false);
        }
        if (m.containsKey("gradient_checkpointing")) {
            b.gradientCheckpointing(Boolean.parseBoolean(String.valueOf(m.get("gradient_checkpointing"))));
        }
        if (type == TrainingType.REINFORCEMENT_LEARNING && stage != null) {
            b.rlAlgorithm(stage.toLowerCase());
        }
        if (m.get("project_name") != null) b.projectName(String.valueOf(m.get("project_name")));
        return b.build();
    }

    /** Reverse: Studio request → factory-ish flat map for hosts that log hparams. */
    public static Map<String, Object> toFactoryArgs(TrainingStartRequest req) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_name_or_path", req.modelName());
        m.put("finetuning_type", req.trainingType() == TrainingType.LORA_QLORA ? "lora"
                : req.trainingType() == TrainingType.FULL_FINETUNING ? "full"
                : req.trainingType() == TrainingType.CONTINUED_PRETRAINING ? "full"
                : "lora");
        if (req.trainingType() == TrainingType.REINFORCEMENT_LEARNING) {
            m.put("stage", req.rlAlgorithm().orElse("grpo"));
        } else if (req.trainingType() == TrainingType.CONTINUED_PRETRAINING) {
            m.put("stage", "pt");
        } else {
            m.put("stage", "sft");
        }
        req.dataset().ifPresent(d -> m.put("dataset", d));
        req.outputDir().ifPresent(d -> m.put("output_dir", d));
        m.put("learning_rate", req.learningRate());
        m.put("cutoff_len", req.maxSeqLength());
        m.put("lora_rank", req.loraR());
        m.put("lora_alpha", req.loraAlpha());
        m.put("per_device_train_batch_size", req.batchSize());
        m.put("gradient_accumulation_steps", req.gradientAccumulationSteps());
        m.put("max_steps", req.maxSteps());
        m.put("quantization_bit", req.loadIn4bit() ? 4 : req.loadIn8bit() ? 8 : 0);
        m.put("gradient_checkpointing", req.gradientCheckpointing());
        return m;
    }

    private static String firstString(Map<String, Object> m, String... keys) {
        for (String k : keys) {
            if (m.get(k) != null) {
                String s = String.valueOf(m.get(k));
                if (!s.isBlank() && !"null".equals(s)) return s;
            }
        }
        return null;
    }

    private static Integer firstInt(Map<String, Object> m, String... keys) {
        for (String k : keys) {
            Object v = m.get(k);
            if (v instanceof Number n) return n.intValue();
            if (v != null) {
                try { return Integer.parseInt(String.valueOf(v)); } catch (Exception ignored) {}
            }
        }
        return null;
    }
}
