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

package org.bytedeco.pytorch.llm.unsloth.studio.export;

import org.bytedeco.pytorch.llm.unsloth.studio.model.ExportFormat;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ExportRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.util.IdGen;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Export orchestration: safetensors / peft-merge / GGUF plan.
 * Writes manifests always; weight conversion is best-effort via peft helpers.
 */
public final class ExportOrchestrator {

    private final AtomicInteger opSeq = new AtomicInteger();
    private volatile String currentCheckpoint;
    private volatile boolean exportActive;
    private volatile String lastOpKind;
    private volatile String lastOpStatus;
    private volatile String lastOpOutputPath;
    private volatile String lastOpError;

    public Path export(ExportRequest request) throws Exception {
        exportActive = true;
        lastOpKind = "export_" + request.format().name();
        lastOpError = null;
        try {
            Path checkpoint = Path.of(request.checkpointPath());
            currentCheckpoint = checkpoint.toString();
            Path saveDir = Path.of(request.saveDirectory());
            StudioPaths.mkdirs(saveDir);

            Map<String, Object> manifest = new LinkedHashMap<>();
            manifest.put("export_id", IdGen.exportId());
            manifest.put("checkpoint_path", request.checkpointPath());
            manifest.put("format", request.format().name());
            manifest.put("format_label", request.format().label());
            manifest.put("load_in_4bit", request.loadIn4bit());
            manifest.put("max_seq_length", request.maxSeqLength());
            manifest.put("created_at_ms", System.currentTimeMillis());

            Path out;
            if (request.format().isGguf()) {
                out = new GgufExportPlanner().plan(request, saveDir, manifest);
            } else if (request.format() == ExportFormat.LORA_ADAPTER) {
                out = new PeftMergeExporter().exportAdapterOnly(request, saveDir, manifest);
            } else if (request.format() == ExportFormat.MERGED_16BIT) {
                out = new PeftMergeExporter().mergeAndExport(request, saveDir, manifest);
            } else {
                out = new SafetensorsExporter().export(request, saveDir, manifest);
            }

            Path manifestPath = out.resolve("export_manifest.json");
            Files.writeString(manifestPath, JsonMaps.stringify(manifest), StandardCharsets.UTF_8);
            lastOpStatus = "success";
            lastOpOutputPath = out.toString();
            opSeq.incrementAndGet();
            return out;
        } catch (Exception e) {
            lastOpStatus = "error";
            lastOpError = e.getMessage();
            opSeq.incrementAndGet();
            throw e;
        } finally {
            exportActive = false;
        }
    }

    public Map<String, Object> status() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("current_checkpoint", currentCheckpoint);
        m.put("is_export_active", exportActive);
        m.put("last_op_seq", opSeq.get());
        m.put("last_op_kind", lastOpKind);
        m.put("last_op_status", lastOpStatus);
        m.put("last_op_output_path", lastOpOutputPath);
        m.put("last_op_error", lastOpError);
        return m;
    }
}
