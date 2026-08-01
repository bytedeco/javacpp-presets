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

package org.bytedeco.pytorch.llm.unsloth.studio.inference.tools;

import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Self-healing tool calling: repair malformed JSON arguments and missing required keys.
 */
public final class SelfHealingToolCaller {

    public static final class HealResult {
        public final ToolCallParser.ToolCall original;
        public final ToolCallParser.ToolCall repaired;
        public final List<String> repairs;
        public final boolean changed;

        public HealResult(ToolCallParser.ToolCall original, ToolCallParser.ToolCall repaired,
                          List<String> repairs, boolean changed) {
            this.original = original;
            this.repaired = repaired;
            this.repairs = List.copyOf(repairs);
            this.changed = changed;
        }
    }

    public HealResult heal(ToolCallParser.ToolCall call, ToolSpec spec) {
        List<String> repairs = new ArrayList<>();
        Map<String, Object> args = new LinkedHashMap<>(call.arguments);
        boolean changed = false;

        if (!call.wellFormed) {
            repairs.add("marked_well_formed_after_parse_salvage");
            changed = true;
        }
        if (spec != null) {
            if (call.name == null || "unknown".equals(call.name)) {
                repairs.add("filled_name_from_spec:" + spec.name());
                changed = true;
            }
            // required property names from JSON schema-like map
            Object req = spec.parameters().get("required");
            if (req instanceof List<?> reqList) {
                for (Object r : reqList) {
                    String key = String.valueOf(r);
                    if (!args.containsKey(key) || args.get(key) == null) {
                        args.put(key, defaultFor(spec, key));
                        repairs.add("filled_missing_required:" + key);
                        changed = true;
                    }
                }
            }
            Object props = spec.parameters().get("properties");
            if (props instanceof Map<?, ?> pm) {
                for (Object k : pm.keySet()) {
                    String key = String.valueOf(k);
                    if (args.containsKey(key) && args.get(key) instanceof String s) {
                        String t = s.trim();
                        if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
                            try {
                                args.put(key, JsonMaps.parse(t));
                                repairs.add("parsed_nested_json:" + key);
                                changed = true;
                            } catch (Exception ignored) {}
                        }
                    }
                }
            }
        }
        // strip internal error keys if we recovered a name
        if (args.containsKey("_error") && call.name != null && !"unknown".equals(call.name)) {
            args.remove("_error");
            repairs.add("removed_parse_error_marker");
            changed = true;
        }
        String name = (call.name == null || "unknown".equals(call.name)) && spec != null
                ? spec.name() : call.name;
        ToolCallParser.ToolCall repaired = new ToolCallParser.ToolCall(
                call.id, name, args, true, call.raw);
        return new HealResult(call, repaired, repairs, changed);
    }

    public List<HealResult> healAll(List<ToolCallParser.ToolCall> calls, List<ToolSpec> specs) {
        Map<String, ToolSpec> byName = new LinkedHashMap<>();
        if (specs != null) for (ToolSpec s : specs) byName.put(s.name(), s);
        List<HealResult> out = new ArrayList<>();
        for (ToolCallParser.ToolCall c : calls) {
            ToolSpec spec = byName.get(c.name);
            if (spec == null && specs != null && specs.size() == 1) spec = specs.get(0);
            out.add(heal(c, spec));
        }
        return out;
    }

    private Object defaultFor(ToolSpec spec, String key) {
        Object props = spec.parameters().get("properties");
        if (props instanceof Map<?, ?> pm && pm.get(key) instanceof Map<?, ?> sch) {
            Object t = sch.get("type");
            if ("integer".equals(t) || "number".equals(t)) return 0;
            if ("boolean".equals(t)) return false;
            if ("array".equals(t)) return List.of();
            if ("object".equals(t)) return Map.of();
        }
        return "";
    }
}
