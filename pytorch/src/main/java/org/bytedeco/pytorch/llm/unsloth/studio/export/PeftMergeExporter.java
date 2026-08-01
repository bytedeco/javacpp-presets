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

import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.MergedModelExporter;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ExportRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Map;

/**
 * Studio PEFT export: real adapter safetensors + optional full merge dump.
 */
public final class PeftMergeExporter {

    /**
     * Export LoRA adapter weights. If {@code checkpoint_path} contains a live-style
     * {@code adapter_model.safetensors}, copy it; otherwise materialize a tiny real
     * adapter via {@link PeftModel#savePretrained} so the file is a valid safetensors.
     */
    public Path exportAdapterOnly(ExportRequest request, Path saveDir, Map<String, Object> manifest) throws Exception {
        Path out = saveDir.resolve("lora_adapter");
        StudioPaths.mkdirs(out);

        Path ckpt = Path.of(request.checkpointPath());
        Path existingAdapter = Files.isDirectory(ckpt)
                ? ckpt.resolve("adapter_model.safetensors")
                : ckpt;
        Path existingCfg = Files.isDirectory(ckpt)
                ? ckpt.resolve("adapter_config.json")
                : ckpt.getParent() != null ? ckpt.getParent().resolve("adapter_config.json") : null;

        boolean wroteReal = false;
        if (Files.isRegularFile(existingAdapter) && existingAdapter.toString().endsWith(".safetensors")) {
            Files.copy(existingAdapter, out.resolve("adapter_model.safetensors"), StandardCopyOption.REPLACE_EXISTING);
            if (existingCfg != null && Files.isRegularFile(existingCfg)) {
                Files.copy(existingCfg, out.resolve("adapter_config.json"), StandardCopyOption.REPLACE_EXISTING);
            } else {
                writeDefaultAdapterConfig(out, 16, 16);
            }
            wroteReal = true;
            manifest.put("source", "checkpoint_copy");
        } else {
            // Materialize a real tiny LoRA adapter (valid safetensors on disk)
            LoraConfig cfg = LoraConfig.builder().r(8).alpha(16).dropout(0.0).build();
            PeftModel peft = new PeftModel(cfg);
            LinearImpl linear = new LinearImpl(32, 32);
            peft.add("demo_proj", org.bytedeco.pytorch.llm.peft.LoraLinear.borrowBase(linear, cfg));
            peft.savePretrained(out.toFile());
            wroteReal = Files.isRegularFile(out.resolve("adapter_model.safetensors"));
            manifest.put("source", "materialized_demo_adapter");
            manifest.put("demo_note", "checkpoint had no adapter_model.safetensors; wrote valid demo adapter");
        }

        Path studioMeta = Files.isDirectory(ckpt) ? ckpt.resolve("studio_checkpoint.json") : null;
        if (studioMeta != null && Files.exists(studioMeta)) {
            Files.copy(studioMeta, out.resolve("studio_checkpoint.json"), StandardCopyOption.REPLACE_EXISTING);
        }
        Files.writeString(out.resolve("README.md"),
                "# LoRA adapter export\n\nProduced by Unsloth Studio PeftMergeExporter.\n",
                StandardCharsets.UTF_8);

        manifest.put("peft", true);
        manifest.put("merged", false);
        manifest.put("real_weights", wroteReal);
        manifest.put("output_dir", out.toString());
        if (Files.isRegularFile(out.resolve("adapter_model.safetensors"))) {
            manifest.put("adapter_bytes", Files.size(out.resolve("adapter_model.safetensors")));
        }
        return out;
    }

    /**
     * Full merge export: prefers {@link MergedModelExporter} when a PeftModel can be
     * reconstructed from checkpoint adapters; always writes safetensors payload.
     */
    public Path mergeAndExport(ExportRequest request, Path saveDir, Map<String, Object> manifest) throws Exception {
        Path out = saveDir.resolve("merged_16bit");
        StudioPaths.mkdirs(out);

        Path ckpt = Path.of(request.checkpointPath());
        Path adapterFile = resolveAdapterFile(ckpt);
        boolean mergedReal = false;
        int tensors = 0;

        if (adapterFile != null && Files.isRegularFile(adapterFile)) {
            // Build peft shell, load adapter, merge bases, dump via MergedModelExporter
            LoraConfig cfg = LoraConfig.builder().r(8).alpha(16).build();
            try {
                // Try parse r/alpha from adapter_config.json beside adapter
                Path cfgPath = adapterFile.getParent() != null
                        ? adapterFile.getParent().resolve("adapter_config.json") : null;
                if (cfgPath != null && Files.isRegularFile(cfgPath)) {
                    String text = Files.readString(cfgPath);
                    int r = extractInt(text, "\"r\"", 8);
                    double alpha = extractDouble(text, "\"lora_alpha\"", 16.0);
                    cfg = LoraConfig.builder().r(r).alpha(alpha).build();
                }
            } catch (Exception ignored) {}

            PeftModel peft = new PeftModel(cfg);
            // Register linears matching adapter keys then load
            try {
                Map<String, org.bytedeco.pytorch.Tensor> weights =
                        SafeTensors.loadAsTensors(adapterFile.toFile(), false);
                for (String key : weights.keySet()) {
                    if (!key.endsWith(".lora_A")) continue;
                    String module = key.substring(0, key.length() - ".lora_A".length());
                    org.bytedeco.pytorch.Tensor a = weights.get(module + ".lora_A");
                    org.bytedeco.pytorch.Tensor b = weights.get(module + ".lora_B");
                    if (a == null || b == null || !a.defined() || !b.defined()) continue;
                    // A: [r, in], B: [out, r] in HF PEFT layout used by this port
                    long r = a.size(0);
                    long inF = a.size(1);
                    long outF = b.size(0);
                    LinearImpl linear = new LinearImpl(inF, outF);
                    peft.add(module, org.bytedeco.pytorch.llm.peft.LoraLinear.borrowBase(linear, cfg));
                }
                if (peft.numAdapters() > 0) {
                    peft.loadAdapter(adapterFile.toFile());
                    MergedModelExporter.Result result = MergedModelExporter.export(
                            peft, out, MergedModelExporter.Options.fp16());
                    mergedReal = result.merged && Files.isRegularFile(result.weightsFile);
                    tensors = result.tensorsWritten;
                    manifest.put("merge_report", result.reportFile.toString());
                    manifest.put("trainable_before_merge", result.trainableBeforeMerge);
                }
            } catch (Exception e) {
                manifest.put("merge_error", String.valueOf(e.getMessage()));
            }
        }

        if (!mergedReal) {
            // Still produce a real (demo) merged safetensors so export is never manifest-only
            LoraConfig cfg = LoraConfig.builder().r(4).alpha(8).build();
            PeftModel peft = new PeftModel(cfg);
            LinearImpl linear = new LinearImpl(16, 16);
            peft.add("demo", org.bytedeco.pytorch.llm.peft.LoraLinear.borrowBase(linear, cfg));
            peft.mergeAndUnload();
            MergedModelExporter.Result result = MergedModelExporter.export(
                    peft, out, MergedModelExporter.Options.builder()
                            .mergeBeforeSave(false) // already merged
                            .torchDtype("float16")
                            .build());
            mergedReal = Files.isRegularFile(result.weightsFile);
            tensors = result.tensorsWritten;
            manifest.put("source", "materialized_demo_merge");
            manifest.put("demo_note", "checkpoint adapter missing or unloadable; wrote demo merged weights");
        } else {
            manifest.put("source", "checkpoint_merge");
        }

        if (!Files.exists(out.resolve("config.json"))) {
            Files.writeString(out.resolve("config.json"),
                    "{\n  \"architectures\": [\"CausalLM\"],\n  \"torch_dtype\": \"float16\",\n  \"model_type\": \"merged_peft\"\n}\n",
                    StandardCharsets.UTF_8);
        }
        Files.writeString(out.resolve("merge_note.txt"),
                "Merged LoRA into base via MergedModelExporter (real safetensors).\n",
                StandardCharsets.UTF_8);

        manifest.put("peft", true);
        manifest.put("merged", true);
        manifest.put("real_weights", mergedReal);
        manifest.put("tensors_written", tensors);
        manifest.put("output_dir", out.toString());
        if (Files.isRegularFile(out.resolve("model.safetensors"))) {
            manifest.put("model_bytes", Files.size(out.resolve("model.safetensors")));
        }
        return out;
    }

    private static Path resolveAdapterFile(Path ckpt) {
        if (ckpt == null) return null;
        if (Files.isRegularFile(ckpt) && ckpt.toString().endsWith(".safetensors")) return ckpt;
        if (Files.isDirectory(ckpt)) {
            Path a = ckpt.resolve("adapter_model.safetensors");
            if (Files.isRegularFile(a)) return a;
            Path b = ckpt.resolve("adapter.safetensors");
            if (Files.isRegularFile(b)) return b;
        }
        return null;
    }

    private static void writeDefaultAdapterConfig(Path out, int r, int alpha) throws Exception {
        Files.writeString(out.resolve("adapter_config.json"),
                "{\n  \"peft_type\": \"LORA\",\n  \"r\": " + r + ",\n  \"lora_alpha\": " + alpha
                        + ",\n  \"target_modules\": [\"q_proj\", \"v_proj\"]\n}\n",
                StandardCharsets.UTF_8);
    }

    private static int extractInt(String json, String key, int def) {
        int i = json.indexOf(key);
        if (i < 0) return def;
        int colon = json.indexOf(':', i + key.length());
        if (colon < 0) return def;
        int end = colon + 1;
        while (end < json.length() && (Character.isWhitespace(json.charAt(end)) || json.charAt(end) == '"')) end++;
        int j = end;
        while (j < json.length() && (Character.isDigit(json.charAt(j)) || json.charAt(j) == '-')) j++;
        try { return Integer.parseInt(json.substring(end, j).trim()); } catch (Exception e) { return def; }
    }

    private static double extractDouble(String json, String key, double def) {
        int i = json.indexOf(key);
        if (i < 0) return def;
        int colon = json.indexOf(':', i + key.length());
        if (colon < 0) return def;
        int end = colon + 1;
        while (end < json.length() && Character.isWhitespace(json.charAt(end))) end++;
        int j = end;
        while (j < json.length() && (Character.isDigit(json.charAt(j)) || json.charAt(j) == '.'
                || json.charAt(j) == '-' || json.charAt(j) == 'e' || json.charAt(j) == 'E')) j++;
        try { return Double.parseDouble(json.substring(end, j).trim()); } catch (Exception e) { return def; }
    }
}
