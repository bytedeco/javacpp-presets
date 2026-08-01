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

import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.RopeScalingType;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.nn.Module;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * RoPE scaling metadata + best-effort patch hooks.
 *
 * <p>Full rotary re-parameterization depends on the concrete modeling code
 * (LLaMA / Qwen / GPT-NeoX). Here we record the requested scaling on the
 * module via a side-channel map and attempt reflective hooks when present
 * ({@code setRopeScaling}, {@code ropeScalingFactor}).
 *
 * <p>Supported types mirror LLaMA-Factory: linear / dynamic / yarn / llama3.
 */
public final class RopeScaler {

    private static final Logger LOG = Logger.getLogger(RopeScaler.class.getName());

    /** Side-channel attributes keyed by identity hash of the module. */
    private static final Map<Integer, Map<String, Object>> ATTRS = new java.util.concurrent.ConcurrentHashMap<>();

    private RopeScaler() {}

    public static void apply(Module model, ModelArgs args) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(args, "args");
        RopeScalingType type = args.ropeScaling();
        if (type == null || !type.enabled()) {
            return;
        }
        double factor = args.ropeScalingFactor() <= 0.0 ? 1.0 : args.ropeScalingFactor();
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("type", type.wireName());
        meta.put("factor", factor);
        ATTRS.put(System.identityHashCode(model), meta);

        // Reflective apply if modeling exposes hooks
        boolean hooked = false;
        for (String method : new String[]{"setRopeScaling", "applyRopeScaling", "set_rope_scaling"}) {
            try {
                var m = model.getClass().getMethod(method, String.class, double.class);
                m.invoke(model, type.wireName(), factor);
                hooked = true;
                break;
            } catch (ReflectiveOperationException ignored) {
            }
        }
        if (!hooked) {
            try {
                var m = model.getClass().getMethod("ropeScalingFactor", double.class);
                m.invoke(model, factor);
                hooked = true;
            } catch (ReflectiveOperationException ignored) {
            }
        }
        if (model instanceof CausalLM) {
            // CausalLM tiny path uses learned pos emb by default; scaling is recorded
            // for export/config parity and for modeling classes that read ATTRS.
            LOG.info("RopeScaler: recorded " + type.wireName() + " factor=" + factor
                    + " on CausalLM (pos-emb models keep absolute positions)");
        } else if (!hooked) {
            LOG.info("RopeScaler: recorded meta only for " + model.getClass().getSimpleName()
                    + " type=" + type.wireName() + " factor=" + factor);
        }
    }

    public static Map<String, Object> get(Module model) {
        if (model == null) return Map.of();
        Map<String, Object> m = ATTRS.get(System.identityHashCode(model));
        return m == null ? Map.of() : Map.copyOf(m);
    }

    /** Effective context length after linear scaling (approx). */
    public static int scaledMaxLen(int baseMaxLen, ModelArgs args) {
        if (args == null || args.ropeScaling() == null || !args.ropeScaling().enabled()) {
            return baseMaxLen;
        }
        double f = args.ropeScalingFactor() <= 0 ? 1.0 : args.ropeScalingFactor();
        return Math.max(1, (int) Math.round(baseMaxLen * f));
    }
}
