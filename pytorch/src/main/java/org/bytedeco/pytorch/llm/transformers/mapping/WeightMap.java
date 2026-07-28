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
package org.bytedeco.pytorch.llm.transformers.mapping;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Declarative HF checkpoint key → Module parameter key rewrite.
 *
 * <p>Rules (applied in order per source key):
 * <ol>
 *   <li>exact rename {@code hf → module}</li>
 *   <li>prefix strip (e.g. drop leading {@code model.} if already nested)</li>
 *   <li>regex rewrite with capture groups</li>
 *   <li>optional transform: {@code none|transpose}</li>
 * </ol>
 *
 * <p>Prefer HF-identical module names so maps stay empty / identity.
 */
public final class WeightMap {

    public enum Transform { NONE, TRANSPOSE }

    public static final class Rule {
        public final String hf;       // exact or regex (when regex=true)
        public final String module;   // replacement; may use $1 style for regex
        public final boolean regex;
        public final Transform transform;

        public Rule(String hf, String module, boolean regex, Transform transform) {
            this.hf = Objects.requireNonNull(hf);
            this.module = Objects.requireNonNull(module);
            this.regex = regex;
            this.transform = transform == null ? Transform.NONE : transform;
        }

        public static Rule exact(String hf, String module) {
            return new Rule(hf, module, false, Transform.NONE);
        }

        public static Rule exact(String hf, String module, Transform t) {
            return new Rule(hf, module, false, t);
        }

        public static Rule regex(String pattern, String module) {
            return new Rule(pattern, module, true, Transform.NONE);
        }
    }

    private final List<Rule> rules;
    private final List<String> stripPrefixes;
    private final boolean identity;

    private WeightMap(List<Rule> rules, List<String> stripPrefixes, boolean identity) {
        this.rules = Collections.unmodifiableList(new ArrayList<>(rules));
        this.stripPrefixes = Collections.unmodifiableList(new ArrayList<>(stripPrefixes));
        this.identity = identity;
    }

    public static WeightMap identity() {
        return new WeightMap(List.of(), List.of(), true);
    }

    public static WeightMap of(List<Rule> rules) {
        return new WeightMap(rules, List.of(), false);
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Apply mapping to a raw HF weight dict → module-keyed dict.
     *
     * <p>For Qwen2/Llama style models, HF checkpoint keys use dots for module hierarchy
     * while Java modules use slashes for layer indices.
     * Conversion: {@code model.layers.0.xxx} → {@code model.layers/0.xxx}
     */
    public Map<String, Tensor> apply(Map<String, Tensor> hfWeights) {
        Objects.requireNonNull(hfWeights, "hfWeights");
        if (identity && stripPrefixes.isEmpty() && rules.isEmpty()) {
            // Identity with slash conversion for Qwen2/Llama HF compatibility
            // Convert only dots before numbers (layer indices) to slashes
            Map<String, Tensor> out = new LinkedHashMap<>(hfWeights.size());
            for (Map.Entry<String, Tensor> e : hfWeights.entrySet()) {
                String key = dotBeforeDigitToSlash(e.getKey());
                out.put(key, e.getValue());
            }
            return out;
        }
        Map<String, Tensor> out = new LinkedHashMap<>(hfWeights.size());
        for (Map.Entry<String, Tensor> e : hfWeights.entrySet()) {
            String src = e.getKey();
            Tensor t = e.getValue();
            Mapped m = mapOne(src);
            Tensor val = t;
            if (m.transform == Transform.TRANSPOSE && t != null && t.defined() && t.dim() == 2) {
                val = t.transpose(0, 1).contiguous();
            }
            out.put(m.key, val);
        }
        return out;
    }

    /**
     * Convert dot-separated HF checkpoint keys to slash-separated module keys.
     * HF:  {@code model.layers.0.self_attn.q_proj.weight}
     * JVM:  {@code model.layers/0.self_attn.q_proj.weight}
     * Only converts dots immediately before digits (layer indices).
     */
    static String dotBeforeDigitToSlash(String hfKey) {
        if (hfKey == null) return null;
        // Replace dots before digits with slash (e.g., "layers.0" → "layers/0")
        return hfKey.replaceAll("\\.(\\d)", "/$1");
    }

    /**
     * Legacy: convert ALL dots to slashes.
     */
    static String dotToSlash(String hfKey) {
        if (hfKey == null) return null;
        return hfKey.replace('.', '/');
    }

    public Mapped mapOne(String hfKey) {
        String key = hfKey;
        for (String p : stripPrefixes) {
            if (key.startsWith(p)) {
                key = key.substring(p.length());
                break;
            }
        }
        for (Rule r : rules) {
            if (!r.regex) {
                if (r.hf.equals(hfKey) || r.hf.equals(key)) {
                    // Always normalize layers.0 → layers/0 for Java module names
                    return new Mapped(dotBeforeDigitToSlash(r.module), r.transform);
                }
            } else {
                Pattern pat = Pattern.compile(r.hf);
                Matcher m = pat.matcher(key);
                if (m.matches()) {
                    String repl = r.module;
                    for (int g = 1; g <= m.groupCount(); g++) {
                        repl = repl.replace("$" + g, m.group(g));
                    }
                    // Critical for Qwen3-VL: rule rewrites
                    // model.language_model.layers.0.* → model.layers.0.*
                    // but Module registers layers/0 — convert before bind.
                    return new Mapped(dotBeforeDigitToSlash(repl), r.transform);
                }
            }
        }
        // No rule matched: apply HF→JVM key conversion
        return new Mapped(dotBeforeDigitToSlash(key), Transform.NONE);
    }

    public static final class Mapped {
        public final String key;
        public final Transform transform;
        public Mapped(String key, Transform transform) {
            this.key = key;
            this.transform = transform;
        }
    }

    public static final class Builder {
        private final List<Rule> rules = new ArrayList<>();
        private final List<String> stripPrefixes = new ArrayList<>();

        public Builder rule(Rule r) { rules.add(r); return this; }
        public Builder exact(String hf, String module) { rules.add(Rule.exact(hf, module)); return this; }
        public Builder exact(String hf, String module, Transform t) { rules.add(Rule.exact(hf, module, t)); return this; }
        public Builder regex(String pattern, String module) { rules.add(Rule.regex(pattern, module)); return this; }
        public Builder stripPrefix(String prefix) { stripPrefixes.add(prefix); return this; }

        public WeightMap build() {
            return new WeightMap(rules, stripPrefixes, rules.isEmpty() && stripPrefixes.isEmpty());
        }
    }

    /** Load rules from a JSON resource / string: {@code { "strip_prefixes":[], "rules":[{hf,module,regex,transform}] }}. */
    @SuppressWarnings("unchecked")
    public static WeightMap fromJson(String json) throws IOException {
        if (json == null || json.isBlank()) return identity();
        Map<String, Object> root = Json.decodeObject(json);
        Builder b = builder();
        Object sp = root.get("strip_prefixes");
        if (sp instanceof List<?> list) {
            for (Object o : list) b.stripPrefix(String.valueOf(o));
        }
        Object rs = root.get("rules");
        if (rs instanceof List<?> list) {
            for (Object o : list) {
                if (!(o instanceof Map<?, ?> m)) continue;
                String hf = String.valueOf(m.get("hf"));
                String module = String.valueOf(m.get("module"));
                boolean regex = Boolean.TRUE.equals(m.get("regex"))
                        || "true".equalsIgnoreCase(String.valueOf(m.get("regex")));
                Transform t = Transform.NONE;
                Object tt = m.get("transform");
                if (tt != null && "transpose".equalsIgnoreCase(String.valueOf(tt))) {
                    t = Transform.TRANSPOSE;
                }
                b.rule(new Rule(hf, module, regex, t));
            }
        }
        return b.build();
    }

    public static WeightMap fromResource(String resourcePath) throws IOException {
        try (InputStream in = WeightMap.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (in == null) return identity();
            String json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return fromJson(json);
        }
    }
}
