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
package org.bytedeco.pytorch.utils.bitsandbytes;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.LoraLinear;
import org.bytedeco.pytorch.llm.peft.QLoRAConfig;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.utils.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.utils.transformers.CausalLM;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * QLoRA fine-tune facade composing BitsAndBytes 4-bit quant + LoRA adapters +
 * transformers {@link CausalLM} / {@link AutoModelForCausalLM}.
 *
 * <p>Mirrors the common Python pattern:
 * <pre>{@code
 * model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
 * model = prepare_model_for_kbit_training(model)
 * model = get_peft_model(model, LoraConfig(...))
 * }</pre>
 *
 * <p>LoRA ΔW is welded into the CausalLM forward graph via
 * {@link CausalLM#attachLora(LoraConfig)} — {@code trainStep} optimizes only
 * A/B through the real LM cross-entropy (no auxiliary loss).
 *
 * <pre>{@code
 * QLoRA.Session s = QLoRA.fromCausalLM(PretrainedConfig.tinyGpt2(),
 *     BitsAndBytesConfig.qloraDefaults(),
 *     LoraConfig.builder().r(8).alpha(16).targetModules(QLoRA.GPT2_TARGETS).build());
 * double loss = s.trainStep(inputIds);  // only LoRA trains
 * s.saveAdapter(new File("adapter.safetensors"));
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class QLoRA {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private QLoRA() {}

    public static final String[] GPT2_TARGETS = {
            "c_attn", "c_proj", "fc_in", "fc_out"
    };

    public static final String[] LLAMA_TARGETS = {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    };

    /**
     * Lightweight plain-Tensor adapter kept for callers that want A/B without a
     * Module (e.g. offline bookkeeping). Prefer {@link LoraLinear} via
     * {@link CausalLM#attachLora} for end-to-end LM training.
     */
    public static final class Adapter {
        public final String name;
        public final LinearImpl base;
        public final Tensor loraA;
        public final Tensor loraB;
        public final double scaling;
        public final long inFeatures;
        public final long outFeatures;
        public final int r;

        public Adapter(String name, LinearImpl base, int r, double alpha) {
            this.name = name;
            this.base = base;
            this.r = r;
            this.inFeatures = base.weight().size(1);
            this.outFeatures = base.weight().size(0);
            this.scaling = alpha / (double) r;
            Tensor aInit = org.bytedeco.pytorch.global.torch.randn(r, inFeatures)
                    .div(new org.bytedeco.pytorch.Scalar(Math.sqrt(r)))
                    .contiguous().clone();
            Tensor bInit = org.bytedeco.pytorch.global.torch.zeros(outFeatures, r)
                    .contiguous().clone();
            aInit.requires_grad_(true);
            bInit.requires_grad_(true);
            this.loraA = aInit;
            this.loraB = bInit;
        }

        public Tensor forward(Tensor input) {
            Tensor result = base.forward(input);
            Tensor aT = loraA.t();
            Tensor bT = loraB.t();
            if (input.dim() == 2) {
                Tensor mid = org.bytedeco.pytorch.global.torch.mm(input, aT);
                return result.add(org.bytedeco.pytorch.global.torch.mm(mid, bT)
                        .mul(new org.bytedeco.pytorch.Scalar(scaling)));
            }
            long[] shape = input.shape();
            long in = shape[shape.length - 1];
            long rest = 1;
            for (int i = 0; i < shape.length - 1; i++) rest *= shape[i];
            Tensor flat = input.reshape(rest, in);
            Tensor mid = org.bytedeco.pytorch.global.torch.mm(flat, aT);
            Tensor out2d = org.bytedeco.pytorch.global.torch.mm(mid, bT)
                    .mul(new org.bytedeco.pytorch.Scalar(scaling));
            long[] outShape = new long[shape.length];
            System.arraycopy(shape, 0, outShape, 0, shape.length - 1);
            outShape[outShape.length - 1] = outFeatures;
            return result.add(out2d.reshape(outShape));
        }

        public long paramCount() {
            return loraA.numel() + loraB.numel();
        }
    }

    public static final class Session implements AutoCloseable {
        private final CausalLM model;
        private final BitsAndBytesConfig bnbConfig;
        private final LoraConfig loraConfig;
        private final BitsAndBytes.QuantizedModel quantized;
        private final List<LoraLinear> adapters;
        private final Map<String, LinearImpl> baseLinears;
        private Optimizer optimizer;
        private long step;
        private final boolean prepared;

        Session(CausalLM model, BitsAndBytesConfig bnbConfig, LoraConfig loraConfig,
                BitsAndBytes.QuantizedModel quantized,
                List<LoraLinear> adapters, Map<String, LinearImpl> baseLinears) {
            this.model = model;
            this.bnbConfig = bnbConfig;
            this.loraConfig = loraConfig;
            this.quantized = quantized;
            this.adapters = Collections.unmodifiableList(new ArrayList<>(adapters));
            this.baseLinears = baseLinears;
            this.prepared = true;
        }

        public CausalLM model() { return model; }
        public BitsAndBytesConfig bnbConfig() { return bnbConfig; }
        public LoraConfig loraConfig() { return loraConfig; }
        public BitsAndBytes.QuantizedModel quantized() { return quantized; }
        public List<LoraLinear> adapters() { return adapters; }
        public long step() { return step; }
        public boolean isPrepared() { return prepared; }

        public Optimizer optimizer() {
            if (optimizer == null) {
                TensorVector params = collectLoraParams();
                optimizer = new Adam(params, new AdamOptions(1e-4));
            }
            return optimizer;
        }

        public Session withOptimizer(Optimizer opt) {
            this.optimizer = opt;
            return this;
        }

        private TensorVector collectLoraParams() {
            TensorVector v = new TensorVector();
            for (LoraLinear ll : adapters) {
                v.push_back(ll.loraA());
                v.push_back(ll.loraB());
            }
            return v;
        }

        public long trainableParameters() {
            long n = 0;
            for (LoraLinear ll : adapters) {
                try {
                    n += ll.loraA().numel() + ll.loraB().numel();
                } catch (Exception ignored) {}
            }
            return n;
        }

        public long totalParameters() {
            long n = 0;
            for (LinearImpl lin : baseLinears.values()) {
                try {
                    if (lin != null && lin.weight() != null && lin.weight().defined()) {
                        n += lin.weight().numel();
                    }
                } catch (Exception ignored) {}
            }
            try {
                if (model.lmHead() != null && model.lmHead().weight() != null) {
                    n += model.lmHead().weight().numel();
                }
            } catch (Exception ignored) {}
            return n + trainableParameters();
        }

        /**
         * One QLoRA training step — real LM cross-entropy with ΔW in the forward
         * graph. Only LoRA A/B receive gradients (base is frozen after quant).
         */
        public double trainStep(Tensor inputIds) {
            model.train(true);
            Optimizer opt = optimizer();
            opt.zero_grad();

            Tensor loss = model.loss(inputIds);
            loss.backward();
            opt.step();
            step++;
            try {
                return loss.item_double();
            } catch (Exception e) {
                try { return loss.item_float(); } catch (Exception e2) { return Double.NaN; }
            }
        }

        public Tensor forward(Tensor inputIds) {
            return model.forward(inputIds);
        }

        public int[] generate(int[] prompt, int maxNew) {
            model.eval();
            return model.generate(prompt, maxNew);
        }

        public void mergeAndUnload() {
            for (LoraLinear ll : adapters) {
                try { ll.merge(); } catch (Exception ignored) {}
            }
        }

        public void saveAdapter(java.io.File file) throws java.io.IOException {
            Map<String, Tensor> state = new LinkedHashMap<>();
            for (Map.Entry<String, LoraLinear> e : model.loraAdapters().entrySet()) {
                state.put(e.getKey() + ".lora_A", e.getValue().loraA());
                state.put(e.getKey() + ".lora_B", e.getValue().loraB());
            }
            // Fallback to list order if map empty
            if (state.isEmpty()) {
                int i = 0;
                for (LoraLinear ll : adapters) {
                    state.put("adapter" + i + ".lora_A", ll.loraA());
                    state.put("adapter" + i + ".lora_B", ll.loraB());
                    i++;
                }
            }
            org.bytedeco.pytorch.data.safetensors.SafeTensors.save(state, file);
        }

        public void loadAdapter(java.io.File file) throws java.io.IOException {
            Map<String, Tensor> state =
                    org.bytedeco.pytorch.data.safetensors.SafeTensors.loadAsTensors(file, false);
            try (NoGradGuard g = new NoGradGuard()) {
                for (Map.Entry<String, LoraLinear> e : model.loraAdapters().entrySet()) {
                    Tensor aa = state.get(e.getKey() + ".lora_A");
                    Tensor bb = state.get(e.getKey() + ".lora_B");
                    if (aa != null && aa.defined()) e.getValue().loraA().copy_(aa);
                    if (bb != null && bb.defined()) e.getValue().loraB().copy_(bb);
                }
            }
        }

        public Map<String, Object> stats() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("step", step);
            m.put("trainable_params", trainableParameters());
            m.put("total_params", totalParameters());
            long total = totalParameters();
            long train = trainableParameters();
            m.put("trainable_ratio", total == 0 ? 0.0 : (double) train / (double) total);
            m.put("adapters", adapters.size());
            m.put("quantized_layers", quantized == null ? 0 : quantized.size());
            m.put("load_in_4bit", bnbConfig != null && bnbConfig.isLoadIn4Bit());
            m.put("load_in_8bit", bnbConfig != null && bnbConfig.isLoadIn8Bit());
            m.put("r", loraConfig.r());
            m.put("alpha", loraConfig.alpha());
            m.put("lora_in_forward", model.hasLora());
            if (quantized != null) {
                m.put("quant_memory_bytes", quantized.quantMemoryBytes());
                m.put("compression_ratio", quantized.stats().get("compression_ratio"));
            }
            return m;
        }

        @Override
        public void close() {
            if (quantized != null) {
                try { quantized.close(); } catch (Exception ignored) {}
            }
        }
    }

    public static Session fromCausalLM(PretrainedConfig cfg, BitsAndBytesConfig bnb, LoraConfig lora) {
        Objects.requireNonNull(cfg, "cfg");
        CausalLM model = CausalLM.fromConfig(cfg);
        return wrap(model, bnb, lora);
    }

    public static Session fromCausalLM(PretrainedConfig cfg) {
        return fromCausalLM(cfg,
                BitsAndBytesConfig.qloraDefaults(),
                LoraConfig.builder().r(8).alpha(16).targetModules(GPT2_TARGETS).build());
    }

    public static Session fromQLoRAConfig(PretrainedConfig cfg, QLoRAConfig qcfg) {
        Objects.requireNonNull(qcfg, "qcfg");
        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(qcfg.loadIn4bit())
                .bnb4BitQuantType(qcfg.bnb4bitQuantType())
                .bnb4BitUseDoubleQuant(qcfg.bnb4bitUseDoubleQuant())
                .bnb4BitComputeDtype(qcfg.bnb4bitComputeDtype())
                .build();
        return fromCausalLM(cfg, bnb, qcfg.lora());
    }

    /**
     * Quantize CausalLM linears in-place (NF4/INT8) then attach LoRA into the
     * forward graph via {@link CausalLM#attachLora(LoraConfig)}.
     */
    public static Session wrap(CausalLM model, BitsAndBytesConfig bnb, LoraConfig lora) {
        Objects.requireNonNull(model, "model");
        BitsAndBytesConfig b = bnb == null ? BitsAndBytesConfig.qloraDefaults() : bnb;
        LoraConfig lc = lora == null
                ? LoraConfig.builder().r(8).alpha(16).targetModules(GPT2_TARGETS).build()
                : lora;

        Map<String, LinearImpl> linears = model.quantizableLinears();
        BitsAndBytes.QuantizedModel qm = null;
        if (b.isQuantized()) {
            qm = BitsAndBytes.prepareForQLoRA(linears, b);
        }

        // Weld LoRA into CausalLM forward (ΔW on every matching linear).
        model.attachLora(lc);
        List<LoraLinear> adapters = new ArrayList<>(model.loraAdapters().values());
        return new Session(model, b, lc, qm, adapters, linears);
    }

    public static Session fromAutoTiny(String kind, BitsAndBytesConfig bnb, LoraConfig lora) {
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.tiny(kind, bnb);
        if (bundle.model() instanceof CausalLM clm) {
            LoraConfig lc = lora == null
                    ? LoraConfig.builder().r(8).alpha(16).targetModules(GPT2_TARGETS).build()
                    : lora;
            BitsAndBytesConfig b = bnb == null ? BitsAndBytesConfig.qloraDefaults() : bnb;
            // tiny(..., bnb) already quantized; just attach LoRA
            clm.attachLora(lc);
            List<LoraLinear> adapters = new ArrayList<>(clm.loraAdapters().values());
            return new Session(clm, b, lc, bundle.quantizedModel(), adapters, clm.quantizableLinears());
        }
        return fromCausalLM(PretrainedConfig.tinyGpt2(), bnb, lora);
    }
}
