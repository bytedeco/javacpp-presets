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

package org.bytedeco.pytorch.llm.unsloth.studio.mcp;

import org.bytedeco.pytorch.llm.unsloth.studio.export.ExportOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.hardware.DeviceProbe;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioModelRegistry;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ExportRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ModelCard;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingStartRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.train.StudioTrainingOrchestrator;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Registers Studio control tools on an MCP registry (models/train/export/hardware). */
public final class McpStudioTools {

    private McpStudioTools() {}

    public static void registerAll(McpToolRegistry registry,
                                   StudioModelRegistry models,
                                   StudioTrainingOrchestrator training,
                                   ExportOrchestrator export) {
        registry.register(new McpToolRegistry.Tool(
                "list_models",
                "List or search Studio model catalog",
                Map.of("type", "object", "properties", Map.of("query", Map.of("type", "string"))),
                args -> {
                    String q = args.get("query") != null ? String.valueOf(args.get("query")) : "";
                    List<Map<String, Object>> list = new ArrayList<>();
                    for (ModelCard c : models.search(q)) list.add(c.toMap());
                    return Map.of("models", list);
                }));

        registry.register(new McpToolRegistry.Tool(
                "start_train",
                "Start a LoRA/QLoRA or full fine-tune run",
                Map.of("type", "object", "properties", Map.of(
                        "model_name", Map.of("type", "string"),
                        "max_steps", Map.of("type", "integer"),
                        "lora_r", Map.of("type", "integer"))),
                args -> {
                    TrainingStartRequest req = TrainingStartRequest.fromMap(args);
                    String runId = training.start(req);
                    return Map.of("run_id", runId);
                }));

        registry.register(new McpToolRegistry.Tool(
                "stop_train",
                "Request cooperative stop of a training run",
                Map.of("type", "object", "properties", Map.of("run_id", Map.of("type", "string")),
                        "required", List.of("run_id")),
                args -> {
                    String runId = String.valueOf(args.get("run_id"));
                    training.stop(runId);
                    return Map.of("run_id", runId, "status", "stop_requested");
                }));

        registry.register(new McpToolRegistry.Tool(
                "list_runs",
                "List training runs",
                Map.of("type", "object", "properties", Map.of()),
                args -> {
                    List<Map<String, Object>> runs = new ArrayList<>();
                    training.list().forEach(r -> runs.add(r.toMap()));
                    return Map.of("runs", runs);
                }));

        registry.register(new McpToolRegistry.Tool(
                "export_model",
                "Export a checkpoint to safetensors / LoRA / GGUF plan",
                Map.of("type", "object", "properties", Map.of(
                        "checkpoint_path", Map.of("type", "string"),
                        "format", Map.of("type", "string"),
                        "save_directory", Map.of("type", "string")),
                        "required", List.of("checkpoint_path", "save_directory")),
                args -> {
                    try {
                        ExportRequest.Builder b = ExportRequest.builder();
                        b.checkpointPath(String.valueOf(args.get("checkpoint_path")));
                        if (args.get("format") != null) b.format(String.valueOf(args.get("format")));
                        b.saveDirectory(String.valueOf(args.get("save_directory")));
                        var path = export.export(b.build());
                        return Map.of("success", true, "output", path.toString());
                    } catch (Exception e) {
                        return Map.of("success", false, "error", String.valueOf(e.getMessage()));
                    }
                }));

        registry.register(new McpToolRegistry.Tool(
                "hardware_probe",
                "Probe local CPU/GPU hardware",
                Map.of("type", "object", "properties", Map.of()),
                args -> DeviceProbe.probe().toMap()));

        registry.register(new McpToolRegistry.Tool(
                "list_recipes",
                "List built-in data recipe templates",
                Map.of("type", "object", "properties", Map.of()),
                args -> Map.of("recipes", List.of("csv_to_alpaca", "pdf_to_text", "docx_to_text"))));
    }
}
