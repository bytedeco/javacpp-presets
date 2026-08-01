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
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Plans GGUF export (quant variant, files, runner args). Actual quantization
 * requires an external GGUF converter SPI.
 */
public final class GgufExportPlanner {

    public Path plan(ExportRequest request, Path saveDir, Map<String, Object> manifest) throws Exception {
        Path out = saveDir.resolve("gguf");
        StudioPaths.mkdirs(out);
        String quant = request.ggufQuant().orElse(quantFromFormat(request.format()));
        Map<String, Object> plan = new LinkedHashMap<>();
        plan.put("quant", quant);
        plan.put("source_checkpoint", request.checkpointPath());
        plan.put("steps", java.util.List.of(
                "1. Load merged fp16/bf16 weights",
                "2. Convert to GGUF via llama.cpp convert script or GgufRuntime SPI",
                "3. Optional quantize-to " + quant));
        plan.put("output_filename", "model-" + quant + ".gguf");
        Files.writeString(out.resolve("gguf_plan.json"),
                org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps.stringify(plan),
                StandardCharsets.UTF_8);
        Files.writeString(out.resolve("CONVERT.md"),
                "# GGUF export plan\n\nQuant: " + quant + "\n\n"
                        + "Host must provide GgufRuntime or llama.cpp converter.\n",
                StandardCharsets.UTF_8);
        manifest.put("gguf_quant", quant);
        manifest.put("gguf_plan", plan);
        manifest.put("output_dir", out.toString());
        return out;
    }

    private static String quantFromFormat(ExportFormat format) {
        return switch (format) {
            case GGUF_Q4_K_M -> "Q4_K_M";
            case GGUF_Q5_K_M -> "Q5_K_M";
            case GGUF_Q8_0 -> "Q8_0";
            default -> "F16";
        };
    }
}
