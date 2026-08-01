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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public final class ToolSpec {
    private final String name;
    private final String description;
    private final Map<String, Object> parameters; // JSON-schema-like
    private final boolean required;

    public ToolSpec(String name, String description, Map<String, Object> parameters, boolean required) {
        this.name = Objects.requireNonNull(name);
        this.description = description != null ? description : "";
        this.parameters = parameters != null ? Map.copyOf(parameters) : Map.of();
        this.required = required;
    }

    public String name() { return name; }
    public String description() { return description; }
    public Map<String, Object> parameters() { return parameters; }
    public boolean required() { return required; }

    public Map<String, Object> toOpenAiTool() {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", name);
        fn.put("description", description);
        fn.put("parameters", parameters);
        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", "function");
        tool.put("function", fn);
        return tool;
    }
}
