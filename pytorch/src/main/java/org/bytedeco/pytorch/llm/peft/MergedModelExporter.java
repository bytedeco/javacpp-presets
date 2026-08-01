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

package org.bytedeco.pytorch.llm.peft;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Full weight dump after LoRA merge — enterprise export path for Studio / factory.
 *
 * <p>Writes:
 * <ul>
 *   <li>{@code model.safetensors} — merged parameters via {@link SafeTensors#saveModule}</li>
 *   <li>{@code config.json} — optional caller-supplied or minimal stub</li>
 *   <li>{@code merge_report.json} — adapter counts, param totals, paths</li>
 * </ul>
 */
public final class MergedModelExporter {

    public static final class Options {
        private final boolean mergeBeforeSave;
        private final boolean unloadAdaptersNote;
        private final String torchDtype;
        private final String configJson;

        private Options(Builder b) {
            this.mergeBeforeSave = b.mergeBeforeSave;
            this.unloadAdaptersNote = b.unloadAdaptersNote;
            this.torchDtype = b.torchDtype != null ? b.torchDtype : "float16";
            this.configJson = b.configJson;
        }

        public static Builder builder() { return new Builder(); }
        public static Options fp16() { return builder().torchDtype("float16").build(); }
        public static Options bf16() { return builder().torchDtype("bfloat16").build(); }
        public static Options fp32() { return builder().torchDtype("float32").build(); }

        public boolean mergeBeforeSave() { return mergeBeforeSave; }
        public String torchDtype() { return torchDtype; }
        public String configJson() { return configJson; }

        public static final class Builder {
            private boolean mergeBeforeSave = true;
            private boolean unloadAdaptersNote = true;
            private String torchDtype = "float16";
            private String configJson;

            public Builder mergeBeforeSave(boolean v) { this.mergeBeforeSave = v; return this; }
            public Builder unloadAdaptersNote(boolean v) { this.unloadAdaptersNote = v; return this; }
            public Builder torchDtype(String v) { this.torchDtype = v; return this; }
            public Builder configJson(String v) { this.configJson = v; return this; }
            public Options build() { return new Options(this); }
        }
    }

    public static final class Result {
        public final Path outputDir;
        public final Path weightsFile;
        public final Path configFile;
        public final Path reportFile;
        public final int tensorsWritten;
        public final long trainableBeforeMerge;
        public final boolean merged;

        public Result(Path outputDir, Path weightsFile, Path configFile, Path reportFile,
                      int tensorsWritten, long trainableBeforeMerge, boolean merged) {
            this.outputDir = outputDir;
            this.weightsFile = weightsFile;
            this.configFile = configFile;
            this.reportFile = reportFile;
            this.tensorsWritten = tensorsWritten;
            this.trainableBeforeMerge = trainableBeforeMerge;
            this.merged = merged;
        }
    }

    private MergedModelExporter() {}

    /**
     * Merge adapters into base (if requested) and write full safetensors under {@code outDir}.
     *
     * @param peft live PeftModel with registered adapters (and optional root Module)
     * @param outDir destination directory
     * @param options export options
     */
    public static Result export(PeftModel peft, Path outDir, Options options) throws IOException {
        Objects.requireNonNull(peft, "peft");
        Objects.requireNonNull(outDir, "outDir");
        Options opt = options != null ? options : Options.fp16();
        Files.createDirectories(outDir);

        long trainable = peft.trainableParameterCount();
        boolean didMerge = false;
        if (opt.mergeBeforeSave && !peft.isMerged()) {
            peft.mergeAndUnload();
            didMerge = true;
        } else if (opt.mergeBeforeSave && peft.isMerged()) {
            didMerge = true;
        }

        Module root = peft.root();
        Path weights = outDir.resolve("model.safetensors");
        int n;
        if (root != null) {
            Map<String, String> meta = new LinkedHashMap<>();
            meta.put("format", "pt");
            meta.put("producer", "jnitorch-MergedModelExporter");
            meta.put("merged", String.valueOf(didMerge));
            n = SafeTensors.saveModule(root, weights.toFile(), meta);
        } else {
            // No root module — dump adapter state (already merged into bases held by LoraLinear)
            Map<String, Tensor> state = new LinkedHashMap<>();
            for (Map.Entry<String, LoraLinear> e : peft.adapters().entrySet()) {
                LoraLinear ll = e.getValue();
                if (ll.base() != null && ll.base().weight() != null && ll.base().weight().defined()) {
                    state.put(e.getKey() + ".weight", ll.base().weight());
                    try {
                        if (ll.base().bias() != null && ll.base().bias().defined()) {
                            state.put(e.getKey() + ".bias", ll.base().bias());
                        }
                    } catch (Throwable ignored) {}
                }
            }
            if (state.isEmpty()) {
                // still write adapter tensors as fallback documentation
                state.putAll(peft.adapterStateDict());
            }
            SafeTensors.save(state, weights.toFile(), Map.of(
                    "producer", "jnitorch-MergedModelExporter",
                    "merged", String.valueOf(didMerge)));
            n = state.size();
        }

        Path cfg = outDir.resolve("config.json");
        String configJson = opt.configJson;
        if (configJson == null || configJson.isBlank()) {
            configJson = "{\n"
                    + "  \"architectures\": [\"CausalLM\"],\n"
                    + "  \"torch_dtype\": \"" + opt.torchDtype + "\",\n"
                    + "  \"model_type\": \"merged_peft\",\n"
                    + "  \"producers\": [\"org.bytedeco.pytorch.llm.peft.MergedModelExporter\"]\n"
                    + "}\n";
        }
        Files.writeString(cfg, configJson, StandardCharsets.UTF_8);

        Path report = outDir.resolve("merge_report.json");
        String reportJson = "{\n"
                + "  \"merged\": " + didMerge + ",\n"
                + "  \"tensors_written\": " + n + ",\n"
                + "  \"trainable_params_before_merge\": " + trainable + ",\n"
                + "  \"num_adapters\": " + peft.numAdapters() + ",\n"
                + "  \"weights\": \"" + weights.toString().replace("\\", "\\\\") + "\",\n"
                + "  \"dtype\": \"" + opt.torchDtype + "\"\n"
                + "}\n";
        Files.writeString(report, reportJson, StandardCharsets.UTF_8);

        Files.writeString(outDir.resolve("README.md"),
                "# Merged model export\n\nProduced by MergedModelExporter after LoRA merge.\n",
                StandardCharsets.UTF_8);

        return new Result(outDir, weights, cfg, report, n, trainable, didMerge);
    }

    public static Result export(PeftModel peft, Path outDir) throws IOException {
        return export(peft, outDir, Options.fp16());
    }

    /** Offline merge of base + adapter safetensors files into one output file. */
    public static Path mergeStateDictFiles(File baseSafetensors, File adapterSafetensors,
                                           double scaling, File outSafetensors) throws IOException {
        Map<String, Tensor> base = SafeTensors.loadAsTensors(baseSafetensors, false);
        Map<String, Tensor> adapter = SafeTensors.loadAsTensors(adapterSafetensors, false);
        PeftModel.applyLoraToStateDict(base, adapter, scaling);
        SafeTensors.save(base, outSafetensors, Map.of("merged", "true"));
        return outSafetensors.toPath();
    }
}
