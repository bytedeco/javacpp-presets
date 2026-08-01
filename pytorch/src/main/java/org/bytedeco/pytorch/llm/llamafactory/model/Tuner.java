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

import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.peft.LoraConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Builds PEFT / freeze parameter groups from {@link FinetuningArgs}.
 *
 * <p>Lives in {@code model} (not train) so chat/export can reuse the same
 * adapter construction rules without depending on trainer types.
 */
public final class Tuner {

    private Tuner() {}

    /**
     * Construct a {@link LoraConfig} from factory finetuning args.
     *
     * <p>Honours rank/alpha/dropout, rsLoRA, target module list ({@code all}
     * expands to common attention + MLP projections).
     */
    public static LoraConfig buildLoraConfig(FinetuningArgs ft, ModelArgs model) {
        int r = Math.max(1, ft.loraRank());
        double alpha = ft.effectiveLoraAlpha();
        double dropout = Math.max(0.0, ft.loraDropout());
        List<String> targets = parseTargets(ft.loraTarget());

        LoraConfig.Builder b = LoraConfig.builder()
                .r(r)
                .alpha(alpha)
                .dropout(dropout)
                .targetModules(targets)
                .freezeBase(true)
                .useRslora(ft.useRslora());

        // Additional target modules appended if present
        if (ft.additionalTarget() != null && !ft.additionalTarget().isBlank()) {
            List<String> extra = parseTargets(ft.additionalTarget());
            List<String> merged = new ArrayList<>(targets);
            for (String e : extra) {
                if (!merged.contains(e)) merged.add(e);
            }
            b.targetModules(merged);
        }
        return b.build();
    }

    /**
     * Parse comma-separated target module names. {@code all} → common set used
     * by LLaMA-Factory / PEFT for decoder-only transformers.
     */
    public static List<String> parseTargets(String raw) {
        if (raw == null || raw.isBlank() || "all".equalsIgnoreCase(raw.trim())) {
            return new ArrayList<>(Arrays.asList(
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "c_attn", "c_proj", "c_fc", "linear"));
        }
        String[] parts = raw.split("[,;\\s]+");
        List<String> out = new ArrayList<>();
        for (String p : parts) {
            if (p == null) continue;
            String t = p.trim();
            if (!t.isEmpty()) out.add(t);
        }
        return out.isEmpty()
                ? new ArrayList<>(List.of("q_proj", "v_proj", "linear"))
                : out;
    }

    /** Human-readable summary for logs / board. */
    public static String describe(FinetuningArgs ft) {
        return "finetuning_type=" + ft.finetuningType().wireName()
                + " stage=" + ft.stage().wireName()
                + " lora_rank=" + ft.loraRank()
                + " lora_alpha=" + ft.effectiveLoraAlpha()
                + " use_rslora=" + ft.useRslora()
                + " use_dora=" + ft.useDora()
                + " use_oft=" + ft.useOft()
                + " use_galore=" + ft.useGalore()
                + " use_badam=" + ft.useBadam()
                + " use_muon=" + ft.useMuon();
    }
}
