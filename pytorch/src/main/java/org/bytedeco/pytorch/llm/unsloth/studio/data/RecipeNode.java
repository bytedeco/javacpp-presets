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

package org.bytedeco.pytorch.llm.unsloth.studio.data;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public final class RecipeNode {
    private final String id;
    private final RecipeNodeType type;
    private final Map<String, Object> params;
    private final java.util.List<String> inputs;

    public RecipeNode(String id, RecipeNodeType type, Map<String, Object> params, java.util.List<String> inputs) {
        this.id = Objects.requireNonNull(id);
        this.type = Objects.requireNonNull(type);
        this.params = params != null ? Map.copyOf(params) : Map.of();
        this.inputs = inputs != null ? java.util.List.copyOf(inputs) : java.util.List.of();
    }

    public String id() { return id; }
    public RecipeNodeType type() { return type; }
    public Map<String, Object> params() { return params; }
    public java.util.List<String> inputs() { return inputs; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("type", type.name());
        m.put("params", params);
        m.put("inputs", inputs);
        return m;
    }
}
