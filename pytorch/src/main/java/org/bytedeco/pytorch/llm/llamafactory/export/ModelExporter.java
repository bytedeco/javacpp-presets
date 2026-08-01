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
package org.bytedeco.pytorch.llm.llamafactory.export;

import org.bytedeco.pytorch.llm.llamafactory.hparams.ExportArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.AdapterLoader;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelCard;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.peft.MergedModelExporter;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Export / merge adapters to a directory (safetensors + config markers).
 *
 * <p>Delegates PEFT merge to {@link MergedModelExporter}; full-weight dumps write
 * a marker JSON for host savers when no PEFT is attached.
 */
public final class ModelExporter {

    private static final Logger LOG = Logger.getLogger(ModelExporter.class.getName());

    private ModelExporter() {}

    public static Path export(FactoryArgs args, LoadedModel loaded, ExportArgs exportArgs)
            throws IOException {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(loaded, "loaded");
        ExportArgs ex = exportArgs == null ? ExportArgs.defaults() : exportArgs;
        Path dir = Path.of(ex.exportDir() == null || ex.exportDir().isBlank()
                ? "export" : ex.exportDir());
        Files.createDirectories(dir);

        PeftModel peft = loaded.peft();
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("export_dir", dir.toAbsolutePath().toString());
        report.put("merge_adapters", ex.mergeAdapters());
        report.put("export_dtype", ex.exportDtype());
        report.put("model_card", loaded.card() == null ? Map.of() : loaded.card().toMap());
        report.put("finetuning", loaded.meta());

        if (peft != null) {
            if (ex.mergeAdapters()) {
                MergedModelExporter.Options opts = MergedModelExporter.Options.builder()
                        .mergeBeforeSave(true)
                        .torchDtype(normalizeDtype(ex.exportDtype()))
                        .build();
                MergedModelExporter.Result r = MergedModelExporter.export(peft, dir, opts);
                report.put("merged", r.merged);
                report.put("tensors_written", r.tensorsWritten);
                report.put("weights_file", r.weightsFile == null ? null : r.weightsFile.toString());
                report.put("trainable_before_merge", r.trainableBeforeMerge);
                LOG.info("Exported merged PEFT model to " + dir.toAbsolutePath()
                        + " tensors=" + r.tensorsWritten);
            } else {
                AdapterLoader.save(peft, dir);
                report.put("merged", false);
                report.put("adapter_only", true);
                LOG.info("Exported adapter-only to " + dir.toAbsolutePath());
            }
        } else {
            // Full / freeze — marker for host weight dump
            Map<String, Object> marker = new LinkedHashMap<>();
            marker.put("full_weights", true);
            marker.put("note", "Attach host saver or use safetensors dump of CausalLM state");
            ModelCard card = loaded.card();
            if (card != null) {
                marker.put("model_card", card.toMap());
            }
            writeJson(dir.resolve("pytorch_model_marker.json"), marker);
            report.put("merged", false);
            report.put("full_weights_marker", true);
            LOG.info("Wrote full-weight export marker to " + dir.toAbsolutePath());
        }

        writeJson(dir.resolve("export_report.json"), report);
        // factory args snapshot for reproducibility
        try {
            writeJson(dir.resolve("factory_args.json"), args.toMap());
        } catch (Throwable ignored) {
        }
        return dir;
    }

    private static String normalizeDtype(String raw) {
        if (raw == null || raw.isBlank()) return "float16";
        String s = raw.trim().toLowerCase(Locale.ROOT);
        return switch (s) {
            case "fp16", "float16", "half" -> "float16";
            case "bf16", "bfloat16" -> "bfloat16";
            case "fp32", "float32", "float" -> "float32";
            default -> s;
        };
    }

    private static void writeJson(Path path, Map<String, Object> map) throws IOException {
        String json;
        try {
            json = Json.encode(map);
        } catch (Throwable t) {
            json = String.valueOf(map);
        }
        Files.writeString(path, json, StandardCharsets.UTF_8);
    }
}
