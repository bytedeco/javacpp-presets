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
import org.bytedeco.pytorch.llm.ktransformers.config.KtQuantConfig;
import org.bytedeco.pytorch.llm.ktransformers.kernel.KtKernelBackend;
import org.bytedeco.pytorch.llm.ktransformers.kernel.QuantLinearOp;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Path-glob matcher + float Linear → {@link QuantLinearOp} packing.
 *
 * <p>Globs use simple {@code *} wildcards (converted to regex). Explicit
 * {@link LayerInjectPlan.LinearTarget} entries override glob bit/group settings.
 */
public final class LinearReplacer {

    private final LayerInjectPlan plan;
    private final KtKernelBackend backend;
    private final List<CompiledGlob> quantGlobs;
    private final List<CompiledGlob> moeGlobs;

    public LinearReplacer(LayerInjectPlan plan, KtKernelBackend backend) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.backend = Objects.requireNonNull(backend, "backend");
        this.quantGlobs = compile(plan.quantLinearGlobs());
        this.moeGlobs = compile(plan.moeFfnGlobs());
    }

    public LayerInjectPlan plan() { return plan; }
    public KtKernelBackend backend() { return backend; }

    public boolean matchesQuant(String modulePath) {
        if (modulePath == null || modulePath.isBlank()) return false;
        if (explicit(modulePath) != null) return true;
        return anyMatch(quantGlobs, modulePath);
    }

    public boolean matchesMoE(String modulePath) {
        if (modulePath == null || modulePath.isBlank()) return false;
        return anyMatch(moeGlobs, modulePath);
    }

    public QuantLinearOp fromLinear(String modulePath, LinearImpl linear) {
        Objects.requireNonNull(linear, "linear");
        Tensor w = linear.weight();
        if (w == null) {
            throw new IllegalStateException("LinearImpl has null weight: " + modulePath);
        }
        Tensor bias = null;
        try {
            bias = linear.bias();
        } catch (Throwable ignored) {
            bias = null;
        }
        return pack(modulePath, w, bias);
    }

    public QuantLinearOp fromWeight(String modulePath, Tensor weightFp) {
        return pack(modulePath, weightFp, null);
    }

    private QuantLinearOp pack(String modulePath, Tensor weightFp, Tensor biasFp) {
        BitsGroup bg = resolveBits(modulePath);
        if (bg.bits < 4) {
            // non-integer path: still pack as INT8 ref for inject demos when plan says quant
            bg = new BitsGroup(8, Math.max(1, plan.recommendedQuant().groupSize()));
        }
        long out = weightFp.size(0);
        long in = weightFp.size(1);
        QuantLinearOp op = new QuantLinearOp(in, out, bg.bits, bg.groupSize, backend, biasFp != null);
        op.packFromFloat(weightFp, biasFp);
        return op;
    }

    private BitsGroup resolveBits(String modulePath) {
        LayerInjectPlan.LinearTarget t = explicit(modulePath);
        if (t != null) {
            int bits = t.bits() > 0 ? t.bits() : effectiveBits(plan.recommendedQuant());
            return new BitsGroup(bits, t.groupSize());
        }
        KtQuantConfig q = plan.recommendedQuant();
        return new BitsGroup(effectiveBits(q), q.groupSize());
    }

    private static int effectiveBits(KtQuantConfig q) {
        if (q == null) return 8;
        int b = q.effectiveBits();
        if (b == 4 || b == 8) return b;
        return 8; // ref path only supports 4/8 packing
    }

    private LayerInjectPlan.LinearTarget explicit(String modulePath) {
        for (LayerInjectPlan.LinearTarget t : plan.explicitTargets()) {
            if (modulePath.equals(t.modulePath()) || matchGlob(t.modulePath(), modulePath)) {
                return t;
            }
        }
        return null;
    }

    private static boolean anyMatch(List<CompiledGlob> globs, String path) {
        for (CompiledGlob g : globs) {
            if (g.pattern.matcher(path).matches() || path.endsWith(g.rawSuffix)) {
                return true;
            }
        }
        return false;
    }

    private static boolean matchGlob(String glob, String path) {
        if (glob == null) return false;
        if (glob.equals(path)) return true;
        return Pattern.compile(globToRegex(glob)).matcher(path).matches();
    }

    private static List<CompiledGlob> compile(List<String> globs) {
        java.util.ArrayList<CompiledGlob> out = new java.util.ArrayList<>();
        if (globs == null) return out;
        for (String g : globs) {
            if (g == null || g.isBlank()) continue;
            String raw = g.trim();
            String suffix = raw.contains(".") ? raw.substring(raw.lastIndexOf('.') + 1) : raw;
            if (suffix.startsWith("*")) suffix = suffix.substring(1);
            out.add(new CompiledGlob(raw, Pattern.compile(globToRegex(raw)), suffix));
        }
        return out;
    }

    /** {@code *.q_proj} → {@code .*\.q_proj} ; {@code model.layers.*.mlp.*} supported. */
    static String globToRegex(String glob) {
        String g = glob.trim();
        StringBuilder sb = new StringBuilder("^");
        for (int i = 0; i < g.length(); i++) {
            char c = g.charAt(i);
            if (c == '*') {
                sb.append(".*");
            } else if (".+?^$()[]{}|\\".indexOf(c) >= 0) {
                sb.append('\\').append(c);
            } else {
                sb.append(c);
            }
        }
        sb.append('$');
        return sb.toString();
    }

    /** Leaf name match helper for hosts that only have short names. */
    public boolean matchesLeaf(String leafName) {
        if (leafName == null) return false;
        String n = leafName.toLowerCase(Locale.ROOT);
        for (CompiledGlob g : quantGlobs) {
            if (n.equals(g.rawSuffix.toLowerCase(Locale.ROOT)) || n.endsWith(g.rawSuffix.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private static final class CompiledGlob {
        final String raw;
        final Pattern pattern;
        final String rawSuffix;

        CompiledGlob(String raw, Pattern pattern, String rawSuffix) {
            this.raw = raw;
            this.pattern = pattern;
            this.rawSuffix = rawSuffix != null ? rawSuffix : raw;
        }
    }

    private static final class BitsGroup {
        final int bits;
        final int groupSize;

        BitsGroup(int bits, int groupSize) {
            this.bits = bits;
            this.groupSize = Math.max(1, groupSize);
        }
    }
}
