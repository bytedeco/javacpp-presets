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
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningArgs;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Freeze-tuning: keep base weights frozen and unfreeze last N layers and/or
 * name-pattern matches (LLaMA-Factory {@code freeze_trainable_layers} /
 * {@code freeze_trainable_modules}).
 *
 * <p>Operates by setting {@code requires_grad} on parameter tensors. When the
 * underlying Module API does not expose named parameters uniformly, we fall
 * back to unfreezing the LM head + last N transformer blocks on
 * {@link CausalLM}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class FreezeTuner {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final Logger LOG = Logger.getLogger(FreezeTuner.class.getName());

    private FreezeTuner() {}

    /**
     * Apply freeze policy. Returns count of tensors left trainable (best-effort).
     */
    public static int apply(Module model, FinetuningArgs ft) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(ft, "ft");

        // 1) Freeze everything first
        freezeAll(model);

        int lastN = ft.freezeTrainableLayers();
        String modules = ft.freezeTrainableModules();
        String extra = ft.freezeExtraModules();

        int unfrozen = 0;
        if (model instanceof CausalLM causal) {
            unfrozen += unfreezeCausal(causal, lastN, modules, extra);
        } else {
            // Generic: unfreeze all if lastN<=0 and modules=all (full-ish freeze path
            // with no structure — host should prefer CausalLM).
            if (lastN <= 0 && isAll(modules)) {
                unfrozen += unfreezeAll(model);
            } else {
                // Best-effort: unfreeze LM-head-like last linear via parameters tail
                unfrozen += unfreezeTailParams(model, Math.max(1, lastN) * 4);
            }
        }
        LOG.info("FreezeTuner: trainable tensors≈" + unfrozen
                + " lastN=" + lastN + " modules=" + modules);
        return unfrozen;
    }

    private static void freezeAll(Module model) {
        TensorVector params = model.parameters();
        if (params == null) return;
        for (long i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p != null) {
                try {
                    p.set_requires_grad(false);
                } catch (Throwable t) {
                    // older bindings may use requires_grad_(false)
                    try {
                        p.requires_grad_(false);
                    } catch (Throwable ignored) {
                        // leave as-is
                    }
                }
            }
        }
    }

    private static int unfreezeAll(Module model) {
        int n = 0;
        TensorVector params = model.parameters();
        if (params == null) return 0;
        for (long i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p != null) {
                try {
                    p.set_requires_grad(true);
                    n++;
                } catch (Throwable t) {
                    try {
                        p.requires_grad_(true);
                        n++;
                    } catch (Throwable ignored) {
                    }
                }
            }
        }
        return n;
    }

    private static int unfreezeTailParams(Module model, int count) {
        TensorVector params = model.parameters();
        if (params == null || params.size() == 0) return 0;
        int n = 0;
        long size = params.size();
        long start = Math.max(0, size - count);
        for (long i = start; i < size; i++) {
            Tensor p = params.get(i);
            if (p == null) continue;
            try {
                p.set_requires_grad(true);
                n++;
            } catch (Throwable t) {
                try {
                    p.requires_grad_(true);
                    n++;
                } catch (Throwable ignored) {
                }
            }
        }
        return n;
    }

    private static int unfreezeCausal(CausalLM causal, int lastN, String modules, String extra) {
        int n = 0;
        List<CausalLM.Block> blocks = causal.blocks();
        int total = blocks == null ? 0 : blocks.size();
        int from = lastN <= 0 ? 0 : Math.max(0, total - lastN);

        // Module name filters
        List<Pattern> allow = compilePatterns(modules);
        List<Pattern> extraPat = compilePatterns(extra);

        // Always allow lm_head when modules=all or explicitly listed
        boolean tuneHead = isAll(modules) || nameMatches("lm_head", allow)
                || nameMatches("lm_head", extraPat);
        if (tuneHead) {
            n += unfreezeModuleParams(causal.lmHead());
        }

        if (blocks != null) {
            for (int i = from; i < total; i++) {
                CausalLM.Block b = blocks.get(i);
                if (b == null) continue;
                if (isAll(modules) || lastN > 0) {
                    n += unfreezeModuleParams(b);
                } else {
                    // selective by submodule name patterns
                    n += unfreezeNamedLinears(causal, allow, extraPat, i);
                }
            }
        }

        // Extra modules anywhere
        if (!extraPat.isEmpty()) {
            n += unfreezeNamedLinears(causal, extraPat, List.of(), -1);
        }
        return n;
    }

    private static int unfreezeNamedLinears(
            CausalLM causal, List<Pattern> allow, List<Pattern> extra, int blockIdx) {
        int n = 0;
        Map<String, LinearImpl> linears = causal.namedLinears();
        if (linears == null) return 0;
        for (Map.Entry<String, LinearImpl> e : linears.entrySet()) {
            String name = e.getKey();
            if (blockIdx >= 0 && !name.contains("." + blockIdx + ".")
                    && !name.contains("blocks." + blockIdx)
                    && !name.contains("h." + blockIdx)
                    && !name.contains("layers." + blockIdx)) {
                // best-effort block filter; if name has no index, still allow match
                if (name.matches(".*\\d+.*") && !name.contains(String.valueOf(blockIdx))) {
                    continue;
                }
            }
            if (nameMatches(name, allow) || nameMatches(name, extra)) {
                n += unfreezeModuleParams(e.getValue());
            }
        }
        return n;
    }

    private static int unfreezeModuleParams(Module m) {
        if (m == null) return 0;
        return unfreezeAll(m);
    }

    private static boolean isAll(String modules) {
        return modules == null || modules.isBlank()
                || "all".equalsIgnoreCase(modules.trim());
    }

    private static List<Pattern> compilePatterns(String raw) {
        List<Pattern> out = new ArrayList<>();
        if (raw == null || raw.isBlank() || isAll(raw)) {
            return out;
        }
        for (String p : raw.split("[,;\\s]+")) {
            if (p == null || p.isBlank()) continue;
            String s = p.trim();
            // treat as substring regex (literal-ish)
            out.add(Pattern.compile(Pattern.quote(s), Pattern.CASE_INSENSITIVE));
        }
        return out;
    }

    private static boolean nameMatches(String name, List<Pattern> pats) {
        if (pats == null || pats.isEmpty()) return false;
        String n = name == null ? "" : name;
        for (Pattern p : pats) {
            if (p.matcher(n).find()) return true;
        }
        // also bare endswith
        String lower = n.toLowerCase(Locale.ROOT);
        for (Pattern p : pats) {
            if (lower.contains(p.pattern().toLowerCase(Locale.ROOT)
                    .replace("\\q", "").replace("\\E", "").replace("\\Q", ""))) {
                return true;
            }
        }
        return false;
    }

    /** Snapshot of which freeze knobs were applied (for board / checkpoint meta). */
    public static Map<String, Object> describe(FinetuningArgs ft) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("freeze_trainable_layers", ft.freezeTrainableLayers());
        m.put("freeze_trainable_modules", ft.freezeTrainableModules());
        m.put("freeze_extra_modules", ft.freezeExtraModules());
        return m;
    }
}
