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

import org.bytedeco.pytorch.llm.unsloth.studio.util.IdGen;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Executes data recipe graphs (Load → Map → Preview → Export). */
public final class RecipeService {

    private final DocumentIngest ingest = new DocumentIngest();
    private final Path recipesDir;

    public RecipeService(Path recipesDir) {
        this.recipesDir = recipesDir;
    }

    public Map<String, Object> run(RecipeGraph graph) throws Exception {
        String jobId = IdGen.recipeJobId();
        Map<String, List<String>> outputs = new LinkedHashMap<>();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("job_id", jobId);
        result.put("recipe", graph.name());
        List<Map<String, Object>> nodeResults = new ArrayList<>();

        for (RecipeNode node : graph.nodes()) {
            List<String> inputRows = new ArrayList<>();
            for (String in : node.inputs()) {
                if (outputs.containsKey(in)) inputRows.addAll(outputs.get(in));
            }
            List<String> outRows = new ArrayList<>();
            Map<String, Object> nr = new LinkedHashMap<>();
            nr.put("node_id", node.id());
            nr.put("type", node.type().name());
            switch (node.type()) {
                case LOAD_CSV, LOAD_JSONL, LOAD_PDF, LOAD_DOCX, LOAD_TEXT -> {
                    Object pathObj = node.params().get("path");
                    if (pathObj == null) throw new IllegalArgumentException("path required for " + node.id());
                    Path p = Path.of(String.valueOf(pathObj));
                    outRows.addAll(ingest.ingest(p));
                }
                case MAP -> {
                    String template = String.valueOf(node.params().getOrDefault("template", "alpaca"));
                    for (String row : inputRows) {
                        outRows.add(mapRow(template, row));
                    }
                }
                case FILTER -> {
                    String contains = String.valueOf(node.params().getOrDefault("contains", ""));
                    for (String row : inputRows) {
                        if (contains.isEmpty() || row.contains(contains)) outRows.add(row);
                    }
                }
                case SAMPLE -> {
                    int limit = node.params().get("limit") instanceof Number n ? n.intValue() : 10;
                    outRows.addAll(inputRows.subList(0, Math.min(limit, inputRows.size())));
                }
                case PREVIEW -> {
                    int limit = node.params().get("limit") instanceof Number n ? n.intValue() : 5;
                    outRows.addAll(inputRows.subList(0, Math.min(limit, inputRows.size())));
                    nr.put("preview", outRows);
                }
                case EXPORT_JSONL, EXPORT_ALPACA -> {
                    Object pathObj = node.params().get("path");
                    Path out = pathObj != null ? Path.of(String.valueOf(pathObj))
                            : recipesDir.resolve(jobId + ".jsonl");
                    if (out.getParent() != null) StudioPaths.mkdirs(out.getParent());
                    Files.write(out, inputRows, StandardCharsets.UTF_8);
                    outRows.addAll(inputRows);
                    nr.put("exported_path", out.toString());
                    nr.put("rows", inputRows.size());
                }
                case PACK -> outRows.addAll(inputRows);
                default -> outRows.addAll(inputRows);
            }
            outputs.put(node.id(), outRows);
            nr.put("row_count", outRows.size());
            nodeResults.add(nr);
        }
        result.put("nodes", nodeResults);
        result.put("status", "completed");
        // persist recipe run
        if (recipesDir != null) {
            StudioPaths.mkdirs(recipesDir);
            Files.writeString(recipesDir.resolve(jobId + "_result.json"),
                    JsonMaps.stringify(result), StandardCharsets.UTF_8);
        }
        return result;
    }

    private String mapRow(String template, String row) {
        if ("alpaca".equalsIgnoreCase(template)) {
            // CSV-ish: instruction,input,output OR single text
            String[] parts = row.split(",", 3);
            if (parts.length >= 3) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("instruction", parts[0].trim());
                m.put("input", parts[1].trim());
                m.put("output", parts[2].trim());
                return JsonMaps.stringify(m);
            }
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("instruction", "Continue the text");
            m.put("input", "");
            m.put("output", row);
            return JsonMaps.stringify(m);
        }
        return row;
    }
}
