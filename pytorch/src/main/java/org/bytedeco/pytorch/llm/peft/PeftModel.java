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
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * PEFT model helper: wrap linears with LoRA, merge/unmerge, save/load adapters.
 *
 * <p>HuggingFace-style entry points mirror Python PEFT:
 * <pre>{@code
 * // Python: model = get_peft_model(model, LoraConfig(...)); model.print_trainable_parameters()
 * PeftModel peft = PeftModel.getPeftModel(causalLm, LoraConfig.builder().r(16).build());
 * peft.printTrainableParameters();
 * peft.savePretrained(new File("./lora_adapter"));
 * PeftModel loaded = PeftModel.fromPretrained(base, new File("./lora_adapter"));
 * Module merged = peft.mergeAndUnload();
 * }</pre>
 *
 * <p>For {@link org.bytedeco.pytorch.llm.transformers.CausalLM}, adapters are welded into
 * the forward graph via {@code attachLora}. For generic modules, use explicit
 * {@link #wrapLinear} / {@link #add}.
 *
 * <p>Also supports offline state-dict merge via {@link #applyLoraToStateDict}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PeftModel {
    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LoraConfig config;
    private final Map<String, LoraLinear> adapters = new LinkedHashMap<>();
    private Module root; // optional outer module when user registers adapters under it
    private boolean merged;
    private long totalBaseParams = -1L;

    public PeftModel(LoraConfig config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    public PeftModel(QLoRAConfig qconfig) {
        this(Objects.requireNonNull(qconfig, "qconfig").lora());
    }

    public LoraConfig config() {
        return config;
    }

    public Map<String, LoraLinear> adapters() {
        return Collections.unmodifiableMap(adapters);
    }

    public boolean isMerged() {
        return merged;
    }

    /** Optional root module the user trains as a whole. */
    public PeftModel root(Module root) {
        this.root = root;
        return this;
    }

    public Module root() {
        return root;
    }

    // ------------------------------------------------------------------ HF-style entry points

    /**
     * HuggingFace {@code get_peft_model(model, peft_config)}.
     *
     * <p>When {@code model} is a {@link org.bytedeco.pytorch.llm.transformers.CausalLM},
     * LoRA is attached into the LM forward graph ({@code attachLora}). Otherwise
     * returns a shell PeftModel with the given config (caller must {@link #add} adapters).
     */
    public static PeftModel getPeftModel(Module model, LoraConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        PeftModel peft = new PeftModel(config).root(model);
        peft.totalBaseParams = countParams(model);
        if (model instanceof org.bytedeco.pytorch.llm.transformers.CausalLM clm) {
            if (config.freezeBase()) {
                freezeBase(clm);
            }
            clm.attachLora(config);
            peft.adapters.putAll(clm.loraAdapters());
        }
        return peft;
    }

    /** Snake alias matching Python {@code get_peft_model}. */
    public static PeftModel get_peft_model(Module model, LoraConfig config) {
        return getPeftModel(model, config);
    }

    /**
     * HuggingFace {@code PeftModel.from_pretrained(base_model, adapter_path)}.
     * Attaches LoRA to a CausalLM (if applicable) then loads adapter safetensors.
     */
    public static PeftModel fromPretrained(Module baseModel, File adapterDir) throws IOException {
        Objects.requireNonNull(baseModel, "baseModel");
        Objects.requireNonNull(adapterDir, "adapterDir");
        LoraConfig cfg = loadConfigOrDefault(adapterDir);
        PeftModel peft = getPeftModel(baseModel, cfg);
        File weights = resolveAdapterFile(adapterDir);
        if (weights != null && weights.isFile()) {
            peft.loadAdapter(weights);
        }
        return peft;
    }

    /** Snake alias matching Python {@code PeftModel.from_pretrained}. */
    public static PeftModel from_pretrained(Module baseModel, File adapterDir) throws IOException {
        return fromPretrained(baseModel, adapterDir);
    }

    public static PeftModel fromPretrained(Module baseModel, String adapterPath) throws IOException {
        return fromPretrained(baseModel, new File(adapterPath));
    }

    public static PeftModel from_pretrained(Module baseModel, String adapterPath) throws IOException {
        return fromPretrained(baseModel, adapterPath);
    }

    /**
     * HuggingFace {@code model.print_trainable_parameters()}.
     * Prints {@code trainable params: X || all params: Y || trainable%: Z}.
     */
    public void printTrainableParameters() {
        long trainable = trainableParameterCount();
        long total = totalParameterCount();
        double pct = total == 0 ? 0.0 : 100.0 * trainable / (double) total;
        System.out.printf(java.util.Locale.US,
                "trainable params: %,d || all params: %,d || trainable%%: %.4f%n",
                trainable, total, pct);
    }

    /** Snake alias matching Python {@code print_trainable_parameters}. */
    public void print_trainable_parameters() {
        printTrainableParameters();
    }

    /** Number of LoRA A/B elements (trainable). */
    public long trainableParameterCount() {
        long n = 0;
        for (LoraLinear layer : adapters.values()) {
            try {
                n += layer.loraA().numel() + layer.loraB().numel();
            } catch (Exception ignored) {}
        }
        return n;
    }

    /**
     * Base + adapter parameter count.
     * {@code totalBaseParams} is snapshotted <em>before</em> attach, so adapters are added on top.
     */
    public long totalParameterCount() {
        long base = totalBaseParams >= 0 ? totalBaseParams
                : (root != null ? countParams(root) : 0L);
        long train = trainableParameterCount();
        // Avoid double-count when root.parameters() already includes adapters
        if (totalBaseParams < 0 && root != null) {
            return Math.max(base, train);
        }
        return base + train;
    }

    /**
     * HuggingFace {@code merge_and_unload()}: merge LoRA into base weights and
     * return the root module (adapters remain registered but merged flag is set).
     */
    public Module mergeAndUnload() {
        mergeAll();
        return root != null ? root : null;
    }

    /** Snake alias matching Python {@code merge_and_unload}. */
    public Module merge_and_unload() {
        return mergeAndUnload();
    }

    /**
     * HuggingFace {@code save_pretrained(path)} — writes adapter safetensors +
     * a minimal {@code adapter_config.json} under the directory.
     */
    public void savePretrained(File dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Cannot create adapter dir: " + dir);
        }
        File weights = new File(dir, "adapter_model.safetensors");
        saveAdapter(weights);
        // Also write HF-ish adapter_config.json (best-effort, no external JSON lib required)
        File cfgFile = new File(dir, "adapter_config.json");
        String json = "{"
                + "\"peft_type\":\"LORA\","
                + "\"r\":" + config.r() + ","
                + "\"lora_alpha\":" + config.alpha() + ","
                + "\"lora_dropout\":" + config.dropout() + ","
                + "\"bias\":\"" + config.bias() + "\","
                + "\"task_type\":\"" + config.taskType() + "\","
                + "\"target_modules\":" + toJsonArray(config.targetModules())
                + "}\n";
        java.nio.file.Files.writeString(cfgFile.toPath(), json);
    }

    /** Snake alias matching Python {@code save_pretrained}. */
    public void save_pretrained(File dir) throws IOException {
        savePretrained(dir);
    }

    public void save_pretrained(String path) throws IOException {
        savePretrained(new File(path));
    }

    public void savePretrained(String path) throws IOException {
        savePretrained(new File(path));
    }

    private static void freezeBase(org.bytedeco.pytorch.llm.transformers.CausalLM model) {
        TensorVector pv = model.parameters();
        for (long i = 0, n = pv.size(); i < n; i++) {
            Tensor p = pv.get(i);
            if (p != null && !p.isNull() && p.defined()) {
                try { p.requires_grad_(false); } catch (Exception ignored) {}
            }
        }
    }

    private static long countParams(Module model) {
        long n = 0;
        try {
            TensorVector pv = model.parameters();
            for (long i = 0, m = pv.size(); i < m; i++) {
                Tensor p = pv.get(i);
                if (p != null && !p.isNull() && p.defined()) {
                    n += p.numel();
                }
            }
        } catch (Exception ignored) {}
        return n;
    }

    private static LoraConfig loadConfigOrDefault(File dir) {
        File cfg = new File(dir, "adapter_config.json");
        if (!cfg.isFile()) {
            return LoraConfig.builder().r(8).alpha(16).build();
        }
        try {
            String text = java.nio.file.Files.readString(cfg.toPath());
            int r = extractInt(text, "\"r\"", 8);
            double alpha = extractDouble(text, "\"lora_alpha\"", 16.0);
            double dropout = extractDouble(text, "\"lora_dropout\"", 0.0);
            return LoraConfig.builder().r(r).alpha(alpha).dropout(dropout).build();
        } catch (Exception e) {
            return LoraConfig.builder().r(8).alpha(16).build();
        }
    }

    private static File resolveAdapterFile(File dirOrFile) {
        if (dirOrFile.isFile()) return dirOrFile;
        File a = new File(dirOrFile, "adapter_model.safetensors");
        if (a.isFile()) return a;
        File b = new File(dirOrFile, "adapter.safetensors");
        if (b.isFile()) return b;
        return null;
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
        while (j < json.length() && (Character.isDigit(json.charAt(j)) || json.charAt(j) == '.' || json.charAt(j) == '-' || json.charAt(j) == 'e' || json.charAt(j) == 'E')) j++;
        try { return Double.parseDouble(json.substring(end, j).trim()); } catch (Exception e) { return def; }
    }

    private static String toJsonArray(List<String> items) {
        if (items == null || items.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append('"').append(items.get(i).replace("\"", "\\\"")).append('"');
        }
        return sb.append(']').toString();
    }

    /**
     * Wrap a {@link LinearImpl} as {@link LoraLinear} and register it under {@code name}.
     */
    public static LoraLinear wrapLinear(String name, LinearImpl linear, LoraConfig config) {
        Objects.requireNonNull(linear, "linear");
        Objects.requireNonNull(config, "config");
        return new LoraLinear(linear, config);
    }

    /** Convenience: create a new linear then wrap. */
    public static LoraLinear wrapLinear(String name, long inFeatures, long outFeatures, LoraConfig config) {
        return wrapLinear(name, new LinearImpl(inFeatures, outFeatures), config);
    }

    /** Register an already-built {@link LoraLinear}. */
    public PeftModel add(String name, LoraLinear layer) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(layer, "layer");
        adapters.put(name, layer);
        return this;
    }

    /**
     * Wrap and register in one step when the name matches {@link LoraConfig#targetModules()}.
     * Non-matching names return the original linear unchanged (not registered).
     */
    public LinearImpl maybeWrap(String name, LinearImpl linear) {
        if (!PeftModelHelper.matchesTarget(name, config)) {
            return linear;
        }
        LoraLinear lora = wrapLinear(name, linear, config);
        adapters.put(name, lora);
        return null; // signal replaced — caller should use lora
    }

    /** All LoRA A/B parameters across registered adapters. */
    public TensorVector trainableParameters() {
        TensorVector all = new TensorVector();
        for (LoraLinear layer : adapters.values()) {
            TensorVector p = layer.loraParameters();
            for (long i = 0; i < p.size(); i++) {
                all.push_back(p.get((int) i));
            }
        }
        return all;
    }

    /** Forward through a single named adapter (for small nets / tests). */
    public Tensor forward(String name, Tensor input) {
        LoraLinear layer = adapters.get(name);
        if (layer == null) {
            throw new IllegalArgumentException("No adapter registered as '" + name + "'");
        }
        return layer.forward(input);
    }

    /** Merge all adapters into their base weights. */
    public void mergeAll() {
        for (LoraLinear layer : adapters.values()) {
            layer.merge();
        }
        merged = true;
    }

    /** Unmerge all adapters. */
    public void unmergeAll() {
        for (LoraLinear layer : adapters.values()) {
            layer.unmerge();
        }
        merged = false;
    }

    /** Adapter-only state dict: keys like {@code <name>.lora_A}, {@code <name>.lora_B}. */
    public Map<String, Tensor> adapterStateDict() {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, LoraLinear> e : adapters.entrySet()) {
            String n = e.getKey();
            LoraLinear layer = e.getValue();
            out.put(n + ".lora_A", layer.loraA());
            out.put(n + ".lora_B", layer.loraB());
        }
        return out;
    }

    /** Load adapter tensors into registered layers (must already exist). */
    public void loadAdapterStateDict(Map<String, Tensor> state) {
        Objects.requireNonNull(state, "state");
        // copy_ into leaves that require_grad needs the flag cleared (libtorch check_inplace).
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            for (Map.Entry<String, LoraLinear> e : adapters.entrySet()) {
                String n = e.getKey();
                LoraLinear layer = e.getValue();
                Tensor a = state.get(n + ".lora_A");
                Tensor b = state.get(n + ".lora_B");
                if (a != null && a.defined()) {
                    safeCopy_(layer.loraA(), a);
                }
                if (b != null && b.defined()) {
                    safeCopy_(layer.loraB(), b);
                }
            }
        }
    }

    private static void safeCopy_(Tensor dst, Tensor src) {
        if (dst == null || !dst.defined() || src == null || !src.defined()) return;
        boolean rg = false;
        try { rg = dst.requires_grad(); } catch (Exception ignored) {}
        if (rg) {
            try { dst.requires_grad_(false); } catch (Exception ignored) {}
        }
        dst.copy_(src);
        if (rg) {
            try { dst.requires_grad_(true); } catch (Exception ignored) {}
        }
    }

    /** Save adapters via {@link SafeTensors}. */
    public void saveAdapter(File file) throws IOException {
        SafeTensors.save(adapterStateDict(), file);
    }

    /** Load adapters from safetensors into already-registered layers. */
    public void loadAdapter(File file) throws IOException {
        Map<String, Tensor> weights = SafeTensors.loadAsTensors(file, false);
        loadAdapterStateDict(weights);
    }

    /**
     * Offline merge: {@code W' = W + B @ A * scaling} for matching keys in a
     * base state dict. Does not require live modules.
     *
     * @param baseWeights mutable map of base parameter name → tensor
     * @param adapterWeights map with keys {@code <module>.lora_A} / {@code .lora_B}
     * @param scaling {@code alpha/r} (or rsLoRA scaling)
     * @return list of base keys that were updated
     */
    public static List<String> applyLoraToStateDict(
            Map<String, Tensor> baseWeights,
            Map<String, Tensor> adapterWeights,
            double scaling) {
        Objects.requireNonNull(baseWeights, "baseWeights");
        Objects.requireNonNull(adapterWeights, "adapterWeights");
        List<String> updated = new ArrayList<>();
        for (String key : new ArrayList<>(adapterWeights.keySet())) {
            if (!key.endsWith(".lora_A")) {
                continue;
            }
            String module = key.substring(0, key.length() - ".lora_A".length());
            Tensor a = adapterWeights.get(module + ".lora_A");
            Tensor b = adapterWeights.get(module + ".lora_B");
            if (a == null || b == null || !a.defined() || !b.defined()) {
                continue;
            }
            // Try common weight key patterns
            String[] candidates = {
                    module + ".weight",
                    module + ".base.weight",
                    "base_model.model." + module + ".weight",
                    module
            };
            for (String wkey : candidates) {
                Tensor w = baseWeights.get(wkey);
                if (w != null && w.defined()) {
                    // ΔW = B @ A * scaling ; shapes [out,r] @ [r,in] -> [out,in]
                    Tensor delta = org.bytedeco.pytorch.global.torch.mm(b, a)
                            .mul(new org.bytedeco.pytorch.Scalar(scaling));
                    w.add_(delta);
                    updated.add(wkey);
                    break;
                }
            }
        }
        return updated;
    }

    public int numAdapters() {
        return adapters.size();
    }

    @Override
    public String toString() {
        return "PeftModel{adapters=" + adapters.keySet() + ", r=" + config.r()
                + ", alpha=" + config.alpha() + ", merged=" + merged + "}";
    }
}
