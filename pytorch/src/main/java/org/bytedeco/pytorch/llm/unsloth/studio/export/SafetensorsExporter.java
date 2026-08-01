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

import org.bytedeco.pytorch.llm.unsloth.studio.model.ExportRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public final class SafetensorsExporter {

    public Path export(ExportRequest request, Path saveDir, Map<String, Object> manifest) throws Exception {
        Path out = saveDir.resolve("safetensors_" + request.format().name().toLowerCase());
        StudioPaths.mkdirs(out);
        // Placeholder weight file + config for offline completeness; real tensors via peft/CausalLM save when wired.
        Path cfg = out.resolve("config.json");
        String configJson = "{\n  \"architectures\": [\"StudioExport\"],\n  \"torch_dtype\": \""
                + dtypeFor(request) + "\",\n  \"model_type\": \"studio\"\n}\n";
        Files.writeString(cfg, configJson, StandardCharsets.UTF_8);
        Path weights = out.resolve("model.safetensors.index.json");
        Files.writeString(weights, "{\n  \"metadata\": {\"total_size\": 0}, \"weight_map\": {}\n}\n",
                StandardCharsets.UTF_8);
        // Copy checkpoint manifest if present
        Path ckptMeta = Path.of(request.checkpointPath()).resolve("studio_checkpoint.json");
        if (Files.exists(ckptMeta)) {
            Files.copy(ckptMeta, out.resolve("studio_checkpoint.json"));
        }
        manifest.put("dtype", dtypeFor(request));
        manifest.put("files", java.util.List.of("config.json", "model.safetensors.index.json"));
        manifest.put("output_dir", out.toString());
        return out;
    }

    private static String dtypeFor(ExportRequest request) {
        return switch (request.format()) {
            case SAFETENSORS_BF16 -> "bfloat16";
            case SAFETENSORS_FP32 -> "float32";
            default -> "float16";
        };
    }
}
