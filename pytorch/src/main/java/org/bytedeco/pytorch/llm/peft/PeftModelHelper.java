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
package org.bytedeco.pytorch.llm.peft;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Target-module name matching for PEFT injection
 * (mirrors Hugging Face PEFT target module resolution).
 *
 * <p>A module name matches when its final path segment equals a target
 * (case-insensitive), or when the full dotted path ends with {@code .target}.
 */
public final class PeftModelHelper {
    private PeftModelHelper() {}

    /** True if {@code moduleName} should receive a LoRA adapter. */
    public static boolean matchesTarget(String moduleName, LoraConfig config) {
        if (moduleName == null || config == null) {
            return false;
        }
        return matchesTarget(moduleName, config.targetModules());
    }

    public static boolean matchesTarget(String moduleName, List<String> targets) {
        if (moduleName == null || targets == null || targets.isEmpty()) {
            return false;
        }
        String name = moduleName.toLowerCase(Locale.ROOT);
        String leaf = leafName(name);
        for (String t : targets) {
            if (t == null || t.isEmpty()) {
                continue;
            }
            String target = t.toLowerCase(Locale.ROOT).trim();
            if (target.isEmpty()) {
                continue;
            }
            if (target.startsWith("re:")) {
                // Optional regex form: re:.*q_proj$
                if (Pattern.compile(target.substring(3)).matcher(moduleName).find()) {
                    return true;
                }
                continue;
            }
            if (leaf.equals(target) || name.equals(target) || name.endsWith("." + target)) {
                return true;
            }
        }
        return false;
    }

    /** Last dotted segment of a module path. */
    public static String leafName(String moduleName) {
        if (moduleName == null || moduleName.isEmpty()) {
            return "";
        }
        int dot = moduleName.lastIndexOf('.');
        return dot >= 0 ? moduleName.substring(dot + 1) : moduleName;
    }

    /** Filter names that match the config targets. */
    public static List<String> filterTargets(Iterable<String> names, LoraConfig config) {
        List<String> out = new ArrayList<>();
        if (names == null || config == null) {
            return out;
        }
        for (String n : names) {
            if (matchesTarget(n, config)) {
                out.add(n);
            }
        }
        return out;
    }

    /** Build a safetensors-style adapter key: {@code base_model.model.<name>.lora_A.weight}. */
    public static String adapterKey(String moduleName, String which) {
        String base = moduleName == null || moduleName.isEmpty() ? "default" : moduleName;
        return "base_model.model." + base + ".lora_" + which + ".weight";
    }
}
