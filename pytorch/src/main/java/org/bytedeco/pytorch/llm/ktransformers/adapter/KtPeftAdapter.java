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
import org.bytedeco.pytorch.llm.ktransformers.config.KtSftConfig;
import org.bytedeco.pytorch.llm.ktransformers.inject.LayerInjectPlan;
import org.bytedeco.pytorch.llm.ktransformers.inject.ModelInjector;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.QLoRAConfig;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Resolves PEFT / QLoRA target modules from KT inject plans + SFT config.
 *
 * <p>Does not fork peft — only builds {@link LoraConfig} / {@link QLoRAConfig}
 * compatible with {@code llm.peft}.
 */
public final class KtPeftAdapter {

    private final KtConfig config;
    private final ModelInjector injector;

    public KtPeftAdapter(KtConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        this.injector = ModelInjector.forConfig(config);
    }

    public static KtPeftAdapter of(KtConfig config) {
        return new KtPeftAdapter(config);
    }

    public KtConfig config() { return config; }
    public ModelInjector injector() { return injector; }

    /** Target module leaf names for LoRA (q_proj, v_proj, …). */
    public List<String> targetModules() {
        Set<String> leaves = new LinkedHashSet<>();
        LayerInjectPlan plan = injector.plan();
        for (String glob : plan.quantLinearGlobs()) {
            String leaf = leafOf(glob);
            if (leaf != null && isLoraFriendly(leaf)) {
                leaves.add(leaf);
            }
        }
        // Always include common attention projections for SFT demos.
        leaves.add("q_proj");
        leaves.add("v_proj");
        leaves.add("o_proj");
        if (config.sft().peftKind() != KtSftConfig.PeftKind.NONE) {
            leaves.add("lm_head");
        }
        return new ArrayList<>(leaves);
    }

    public LoraConfig buildLoraConfig() {
        KtSftConfig sft = config.sft();
        List<String> targets = targetModules();
        return LoraConfig.builder()
                .r(Math.max(1, sft.loraR() > 0 ? sft.loraR() : 8))
                .alpha(sft.loraAlpha() > 0 ? sft.loraAlpha() : 16.0)
                .dropout(Math.max(0.0, sft.loraDropout()))
                .targetModules(targets.toArray(new String[0]))
                .freezeBase(true)
                .build();
    }

    /**
     * QLoRA config when peft kind is QLORA; otherwise same as LoRA with a note
     * that base quant is handled by {@link ModelInjector}.
     */
    public PeftConfig buildPeftConfig() {
        KtSftConfig sft = config.sft();
        if (sft.peftKind() == KtSftConfig.PeftKind.NONE) {
            return null;
        }
        LoraConfig lora = buildLoraConfig();
        if (sft.peftKind() == KtSftConfig.PeftKind.QLORA) {
            try {
                return QLoRAConfig.builder()
                        .r(lora.r())
                        .alpha(lora.alpha())
                        .dropout(lora.dropout())
                        .targetModules(lora.targetModules().toArray(new String[0]))
                        .freezeBase(true)
                        .build();
            } catch (Throwable t) {
                // Older peft surface may not expose QLoRA builder the same way.
                return lora;
            }
        }
        return lora;
    }

    public boolean isPeftEnabled() {
        return config.sft().peftKind() != KtSftConfig.PeftKind.NONE && config.sft().loraR() > 0;
    }

    private static String leafOf(String glob) {
        if (glob == null || glob.isBlank()) return null;
        String g = glob.trim();
        int dot = g.lastIndexOf('.');
        String leaf = dot >= 0 ? g.substring(dot + 1) : g;
        if (leaf.startsWith("*")) leaf = leaf.substring(1);
        return leaf.isEmpty() ? null : leaf;
    }

    private static boolean isLoraFriendly(String leaf) {
        String n = leaf.toLowerCase();
        if (n.contains("expert") && n.contains("gate") && n.length() < 6) return false;
        return n.contains("proj") || n.contains("dense") || n.equals("w1") || n.equals("w2")
                || n.equals("w3") || n.equals("lm_head") || n.equals("linear");
    }
}
