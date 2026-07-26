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
 * <p>Full automatic tree rewriting of arbitrary libtorch module graphs is awkward
 * from Java, so the MVP exposes explicit injection:
 * <pre>{@code
 * LoraConfig cfg = LoraConfig.builder().r(8).alpha(16).build();
 * LoraLinear q = PeftModel.wrapLinear("q_proj", baseQ, cfg);
 * PeftModel peft = new PeftModel(cfg).add("q_proj", q).add("v_proj", v);
 * optimizer = SGD(peft.trainableParameters(), ...);
 * }</pre>
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
        for (Map.Entry<String, LoraLinear> e : adapters.entrySet()) {
            String n = e.getKey();
            LoraLinear layer = e.getValue();
            Tensor a = state.get(n + ".lora_A");
            Tensor b = state.get(n + ".lora_B");
            if (a != null && a.defined()) {
                layer.loraA().copy_(a);
            }
            if (b != null && b.defined()) {
                layer.loraB().copy_(b);
            }
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
