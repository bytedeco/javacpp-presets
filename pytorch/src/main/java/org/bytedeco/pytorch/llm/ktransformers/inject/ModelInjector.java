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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtModelFamily;
import org.bytedeco.pytorch.llm.ktransformers.config.KtQuantConfig;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.DeepSeekFamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.GenericFamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.GlmMoeFamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.KimiFamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.Llama4FamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.MiniMaxFamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.MixtralFamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.inject.families.QwenMoEFamilyInject;
import org.bytedeco.pytorch.llm.ktransformers.kernel.KernelRegistry;
import org.bytedeco.pytorch.llm.ktransformers.kernel.KtKernelBackend;
import org.bytedeco.pytorch.llm.ktransformers.kernel.QuantLinearOp;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Applies a {@link LayerInjectPlan} to host models / weight maps.
 *
 * <p>Two concrete surfaces:
 * <ul>
 *   <li>{@link #plan()} — family declarative plan (no weights needed)</li>
 *   <li>{@link #replaceLinear(String, LinearImpl)} / {@link #packWeight(String, Tensor)} —
 *       build {@link QuantLinearOp} for matching module paths</li>
 * </ul>
 *
 * <p>Does not invent unpublished HF layouts; hosts still load tensors.
 */
public final class ModelInjector {

    private final KtModelFamily family;
    private final LayerInjectPlan plan;
    private final KtKernelBackend backend;
    private final LinearReplacer replacer;
    private final Map<String, QuantLinearOp> replaced = new LinkedHashMap<>();

    public ModelInjector(KtModelFamily family, LayerInjectPlan plan, KtKernelBackend backend) {
        this.family = Objects.requireNonNull(family, "family");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.backend = backend != null ? backend : KernelRegistry.defaultBackend();
        this.replacer = new LinearReplacer(this.plan, this.backend);
    }

    public static ModelInjector forFamily(KtModelFamily family, KtKernelBackend backend) {
        return new ModelInjector(family, planFor(family), backend);
    }

    public static ModelInjector forFamily(KtModelFamily family) {
        return forFamily(family, KernelRegistry.defaultBackend());
    }

    public static ModelInjector forConfig(KtConfig config) {
        Objects.requireNonNull(config, "config");
        LayerInjectPlan p = planFor(config.modelFamily());
        // Prefer quant settings from config when integer weights are requested.
        if (config.quant() != null && config.quant().isIntegerWeights()) {
            p = LayerInjectPlan.builder(config.modelFamily())
                    .attentionKind(p.attentionKind())
                    .moe(p.moe())
                    .sharedExpert(p.sharedExpert())
                    .quantLinearGlobs(p.quantLinearGlobs().toArray(new String[0]))
                    .moeFfnGlobs(p.moeFfnGlobs().toArray(new String[0]))
                    .recommendedQuant(config.quant())
                    .note("source", "KtConfig quant override")
                    .build();
        }
        return new ModelInjector(config.modelFamily(), p, KernelRegistry.defaultBackend());
    }

    public static ModelInjector generic() {
        return forFamily(KtModelFamily.GENERIC);
    }

    public static LayerInjectPlan planFor(KtModelFamily family) {
        Objects.requireNonNull(family, "family");
        switch (family) {
            case DEEPSEEK_V2:
            case DEEPSEEK_V3:
            case DEEPSEEK_R1:
            case DEEPSEEK_V4_FLASH:
                return DeepSeekFamilyInject.plan(family);
            case KIMI_K2:
            case KIMI_K2_THINKING:
            case KIMI_K2_5:
                return KimiFamilyInject.plan(family);
            case MINIMAX_M2:
            case MINIMAX_M2_1:
            case MINIMAX_M2_5:
            case MINIMAX_M3:
                return MiniMaxFamilyInject.plan(family);
            case GLM4_MOE:
            case GLM5:
            case GLM5_2:
                return GlmMoeFamilyInject.plan(family);
            case QWEN3_MOE:
            case QWEN3_NEXT:
                return QwenMoEFamilyInject.plan(family);
            case MIXTRAL:
            case SMALLTHINKER:
                return MixtralFamilyInject.plan(family);
            case LLAMA4:
                return Llama4FamilyInject.plan(family);
            case GENERIC:
            default:
                return GenericFamilyInject.plan();
        }
    }

    public KtModelFamily family() { return family; }
    public LayerInjectPlan plan() { return plan; }
    public KtKernelBackend backend() { return backend; }
    public LinearReplacer replacer() { return replacer; }

    public Map<String, QuantLinearOp> replaced() {
        return Collections.unmodifiableMap(replaced);
    }

    /** Whether {@code modulePath} should become a quant linear under this plan. */
    public boolean shouldQuantize(String modulePath) {
        return replacer.matchesQuant(modulePath);
    }

    public boolean shouldMoE(String modulePath) {
        return replacer.matchesMoE(modulePath);
    }

    /**
     * Build a packed {@link QuantLinearOp} from a float Linear module if the path matches.
     * Returns {@code null} when the path is not a quant target.
     */
    public QuantLinearOp replaceLinear(String modulePath, LinearImpl linear) {
        Objects.requireNonNull(modulePath, "modulePath");
        Objects.requireNonNull(linear, "linear");
        if (!shouldQuantize(modulePath)) {
            return null;
        }
        QuantLinearOp op = replacer.fromLinear(modulePath, linear);
        if (op != null) {
            replaced.put(modulePath, op);
        }
        return op;
    }

    /**
     * Pack a free weight matrix {@code [out, in]} for a matching module path.
     */
    public QuantLinearOp packWeight(String modulePath, Tensor weightFp) {
        Objects.requireNonNull(modulePath, "modulePath");
        Objects.requireNonNull(weightFp, "weightFp");
        if (!shouldQuantize(modulePath)) {
            return null;
        }
        QuantLinearOp op = replacer.fromWeight(modulePath, weightFp);
        if (op != null) {
            replaced.put(modulePath, op);
        }
        return op;
    }

    /**
     * Walk a flat name→Linear map (host loaders often produce this) and replace matches.
     *
     * @return map of path → QuantLinearOp for successfully packed layers
     */
    public Map<String, QuantLinearOp> injectLinears(Map<String, LinearImpl> namedLinears) {
        Map<String, QuantLinearOp> out = new LinkedHashMap<>();
        if (namedLinears == null) return out;
        for (Map.Entry<String, LinearImpl> e : namedLinears.entrySet()) {
            QuantLinearOp op = replaceLinear(e.getKey(), e.getValue());
            if (op != null) out.put(e.getKey(), op);
        }
        return out;
    }

    /** Summary for demos / board. */
    public Map<String, Double> stats() {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("kt/inject/family_ordinal", (double) family.ordinal());
        m.put("kt/inject/replaced", (double) replaced.size());
        m.put("kt/inject/quant_globs", (double) plan.quantLinearGlobs().size());
        m.put("kt/inject/moe", plan.moe() ? 1.0 : 0.0);
        m.put("kt/inject/mla",
                plan.attentionKind() == LayerInjectPlan.AttentionKind.MLA ? 1.0 : 0.0);
        return m;
    }

    public void closeReplaced() {
        List<String> keys = new ArrayList<>(replaced.keySet());
        for (String k : keys) {
            QuantLinearOp op = replaced.remove(k);
            if (op != null) {
                try {
                    op.close();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    /** Touch root module graph so hosts can discover named children (no rewire). */
    public void touchGraph(Module root) {
        if (root != null) {
            root.named_modules();
        }
    }
}
