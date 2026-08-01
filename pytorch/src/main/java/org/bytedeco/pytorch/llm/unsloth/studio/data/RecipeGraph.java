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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Visual-node data recipe graph. */
public final class RecipeGraph {
    private final String name;
    private final List<RecipeNode> nodes;

    public RecipeGraph(String name, List<RecipeNode> nodes) {
        this.name = name != null ? name : "recipe";
        this.nodes = List.copyOf(nodes);
    }

    public String name() { return name; }
    public List<RecipeNode> nodes() { return nodes; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", name);
        List<Map<String, Object>> ns = new ArrayList<>();
        for (RecipeNode n : nodes) ns.add(n.toMap());
        m.put("nodes", ns);
        return m;
    }

    public static RecipeGraph csvToAlpaca(String csvPath, String outJsonl) {
        return new RecipeGraph("csv_to_alpaca", List.of(
                new RecipeNode("n1", RecipeNodeType.LOAD_CSV, Map.of("path", csvPath), List.of()),
                new RecipeNode("n2", RecipeNodeType.MAP, Map.of("template", "alpaca"), List.of("n1")),
                new RecipeNode("n3", RecipeNodeType.PREVIEW, Map.of("limit", 5), List.of("n2")),
                new RecipeNode("n4", RecipeNodeType.EXPORT_ALPACA, Map.of("path", outJsonl), List.of("n2"))
        ));
    }
}
