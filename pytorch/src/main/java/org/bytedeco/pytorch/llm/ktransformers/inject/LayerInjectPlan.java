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
package org.bytedeco.pytorch.llm.ktransformers.inject;

import org.bytedeco.pytorch.llm.ktransformers.config.KtModelFamily;
import org.bytedeco.pytorch.llm.ktransformers.config.KtQuantConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Declarative per-model injection plan (upstream "injection" idea).
 *
 * <p>Describes which linear modules become quant linear, which FFN blocks are
 * MoE, whether MLA is preferred, and recommended quant defaults. Does not
 * invent unpublished weight layouts — host loaders still supply tensors.
 */
public final class LayerInjectPlan {

    public enum LinearKind {
        DENSE,
        QUANT,
        EXPERT_FFN,
        SHARED_EXPERT
    }

    public enum AttentionKind {
        STANDARD,
        MLA,
        PAGED
    }

    /** One named linear replacement target. */
    public static final class LinearTarget {
        private final String modulePath;
        private final LinearKind kind;
        private final int bits;
        private final int groupSize;

        public LinearTarget(String modulePath, LinearKind kind, int bits, int groupSize) {
            this.modulePath = Objects.requireNonNull(modulePath, "modulePath");
            this.kind = Objects.requireNonNull(kind, "kind");
            this.bits = bits;
            this.groupSize = Math.max(1, groupSize);
        }

        public String modulePath() { return modulePath; }
        public LinearKind kind() { return kind; }
        public int bits() { return bits; }
        public int groupSize() { return groupSize; }
    }

    private final KtModelFamily family;
    private final AttentionKind attentionKind;
    private final boolean moe;
    private final boolean sharedExpert;
    private final List<String> quantLinearGlobs;
    private final List<String> moeFfnGlobs;
    private final List<LinearTarget> explicitTargets;
    private final KtQuantConfig recommendedQuant;
    private final Map<String, String> notes;

    private LayerInjectPlan(Builder b) {
        this.family = Objects.requireNonNull(b.family, "family");
        this.attentionKind = Objects.requireNonNull(b.attentionKind, "attentionKind");
        this.moe = b.moe;
        this.sharedExpert = b.sharedExpert;
        this.quantLinearGlobs = Collections.unmodifiableList(new ArrayList<>(b.quantLinearGlobs));
        this.moeFfnGlobs = Collections.unmodifiableList(new ArrayList<>(b.moeFfnGlobs));
        this.explicitTargets = Collections.unmodifiableList(new ArrayList<>(b.explicitTargets));
        this.recommendedQuant = b.recommendedQuant != null ? b.recommendedQuant : KtQuantConfig.bf16();
        this.notes = Collections.unmodifiableMap(new LinkedHashMap<>(b.notes));
    }

    public KtModelFamily family() { return family; }
    public AttentionKind attentionKind() { return attentionKind; }
    public boolean moe() { return moe; }
    public boolean sharedExpert() { return sharedExpert; }
    public List<String> quantLinearGlobs() { return quantLinearGlobs; }
    public List<String> moeFfnGlobs() { return moeFfnGlobs; }
    public List<LinearTarget> explicitTargets() { return explicitTargets; }
    public KtQuantConfig recommendedQuant() { return recommendedQuant; }
    public Map<String, String> notes() { return notes; }

    public static Builder builder(KtModelFamily family) {
        return new Builder(family);
    }

    /** Safe default when only a manual layer map is known. */
    public static LayerInjectPlan generic() {
        return builder(KtModelFamily.GENERIC)
                .attentionKind(AttentionKind.STANDARD)
                .moe(false)
                .quantLinearGlobs("*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj", "*.gate_proj",
                        "*.up_proj", "*.down_proj")
                .recommendedQuant(KtQuantConfig.int8AmxLike())
                .note("layout", "user-supplied layer map / generic HF names")
                .build();
    }

    public static final class Builder {
        private final KtModelFamily family;
        private AttentionKind attentionKind = AttentionKind.STANDARD;
        private boolean moe;
        private boolean sharedExpert;
        private final List<String> quantLinearGlobs = new ArrayList<>();
        private final List<String> moeFfnGlobs = new ArrayList<>();
        private final List<LinearTarget> explicitTargets = new ArrayList<>();
        private KtQuantConfig recommendedQuant;
        private final Map<String, String> notes = new LinkedHashMap<>();

        public Builder(KtModelFamily family) {
            this.family = family;
        }

        public Builder attentionKind(AttentionKind v) { this.attentionKind = v; return this; }
        public Builder moe(boolean v) { this.moe = v; return this; }
        public Builder sharedExpert(boolean v) { this.sharedExpert = v; return this; }
        public Builder quantLinearGlobs(String... globs) {
            if (globs != null) {
                for (String g : globs) {
                    if (g != null && !g.isBlank()) quantLinearGlobs.add(g.trim());
                }
            }
            return this;
        }
        public Builder moeFfnGlobs(String... globs) {
            if (globs != null) {
                for (String g : globs) {
                    if (g != null && !g.isBlank()) moeFfnGlobs.add(g.trim());
                }
            }
            return this;
        }
        public Builder target(String path, LinearKind kind, int bits, int groupSize) {
            explicitTargets.add(new LinearTarget(path, kind, bits, groupSize));
            return this;
        }
        public Builder recommendedQuant(KtQuantConfig v) { this.recommendedQuant = v; return this; }
        public Builder note(String k, String v) {
            if (k != null) notes.put(k, v);
            return this;
        }

        public LayerInjectPlan build() {
            return new LayerInjectPlan(this);
        }
    }
}
