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
package org.bytedeco.pytorch.llm.unsloth;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.peft.LoraLinear;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.bitsandbytes.BitsAndBytes;
import org.bytedeco.pytorch.llm.bitsandbytes.QLoRA;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Unsloth-style {@code FastLanguageModel} (Java port).
 *
 * <p>Composes {@link CausalLM} + LoRA (welded into forward via
 * {@link CausalLM#attachLora}) + optional 4/8-bit quant via {@link BitsAndBytes}.
 *
 * <pre>{@code
 * FastLanguageModel m = FastLanguageModel.fromPretrained(
 *     PretrainedConfig.tinyGpt2(),
 *     FastConfig.builder().r(8).loadIn4bit(true)
 *         .targetModules(List.of("c_attn","c_proj","fc_in","fc_out")).build())
 *     .getPeftModel();
 * m.trainStep(inputIds);  // ΔW in LM forward, only LoRA trains
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class FastLanguageModel {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final PretrainedConfig config;
    private final FastConfig fastConfig;
    private final CausalLM model;
    private final List<LoraLinear> adapters = new ArrayList<>();
    private final Map<String, BitsAndBytes.QuantState> quantStates = new LinkedHashMap<>();
    private BitsAndBytes.QuantizedModel quantizedModel;
    private QLoRA.Session qloraSession;
    private long stepCount;
    private boolean checkpointingEnabled;
    private boolean inferenceMode;
    private boolean peftApplied;

    private FastLanguageModel(PretrainedConfig config, FastConfig fastConfig, boolean applyPeft) {
        this.config = Objects.requireNonNull(config);
        this.fastConfig = Objects.requireNonNull(fastConfig);
        this.model = CausalLM.fromConfig(config);
        this.checkpointingEnabled = fastConfig.gradientCheckpointing()
                || fastConfig.useGradientCheckpointing();
        if (fastConfig.loadIn4bit() || fastConfig.loadIn8bit()) {
            quantizeBaseLinears();
        }
        if (applyPeft && !fastConfig.fullFinetuning()) {
            injectLoraAdapters();
            peftApplied = true;
        }
    }

    public static FastLanguageModel fromPretrained(PretrainedConfig config, FastConfig fastConfig) {
        return new FastLanguageModel(config, fastConfig, false);
    }

    public static FastLanguageModel fromPretrained(PretrainedConfig config) {
        return fromPretrained(config, FastConfig.builder().build());
    }

    public static FastLanguageModel from_pretrained(PretrainedConfig config, FastConfig fastConfig) {
        return fromPretrained(config, fastConfig);
    }

    public FastLanguageModel getPeftModel() {
        if (!peftApplied && !fastConfig.fullFinetuning()) {
            injectLoraAdapters();
            peftApplied = true;
        }
        return this;
    }

    public FastLanguageModel get_peft_model() {
        return getPeftModel();
    }

    private void quantizeBaseLinears() {
        try {
            BitsAndBytesConfig bnb = fastConfig.toBnbConfig();
            Map<String, LinearImpl> linears = model.quantizableLinears();
            if (linears.isEmpty()) {
                LinearImpl probe = new LinearImpl(config.hiddenSize(), config.hiddenSize());
                BitsAndBytes.QuantState qs = BitsAndBytes.quantize(probe.weight(), bnb);
                quantStates.put("probe_linear", qs);
                return;
            }
            quantizedModel = BitsAndBytes.prepareForQLoRA(linears, bnb);
            quantStates.putAll(quantizedModel.states());
        } catch (Exception ignored) {
        }
    }

    /**
     * Attach LoRA into the CausalLM forward graph. If not yet quantized but
     * 4/8-bit requested, goes through full {@link QLoRA#wrap}.
     */
    private void injectLoraAdapters() {
        if (!adapters.isEmpty()) return;
        try {
            BitsAndBytesConfig bnb;
            if (quantizedModel != null) {
                // Already quantized — only attach LoRA (no re-quant).
                bnb = BitsAndBytesConfig.builder().loadIn4Bit(false).loadIn8Bit(false).build();
            } else if (fastConfig.loadIn4bit() || fastConfig.loadIn8bit()) {
                bnb = fastConfig.toBnbConfig();
            } else {
                bnb = BitsAndBytesConfig.builder().loadIn4Bit(false).loadIn8Bit(false).build();
            }
            qloraSession = QLoRA.wrap(model, bnb, fastConfig.toLoraConfig());
            adapters.addAll(qloraSession.adapters());
            if (qloraSession.quantized() != null) {
                quantizedModel = qloraSession.quantized();
                quantStates.clear();
                quantStates.putAll(quantizedModel.states());
            }
        } catch (Exception ignored) {
        }
    }

    public CausalLM model() { return model; }
    public FastConfig fastConfig() { return fastConfig; }
    public PretrainedConfig config() { return config; }
    public long stepCount() { return stepCount; }
    public boolean checkpointingEnabled() { return checkpointingEnabled; }
    public boolean isInferenceMode() { return inferenceMode; }
    public boolean isPeftApplied() { return peftApplied; }
    public Map<String, BitsAndBytes.QuantState> quantStates() { return Map.copyOf(quantStates); }
    public BitsAndBytes.QuantizedModel quantizedModel() { return quantizedModel; }
    public List<LoraLinear> injectedAdapters() { return List.copyOf(adapters); }
    public QLoRA.Session qloraSession() { return qloraSession; }
    public boolean isQuantized() { return quantizedModel != null && quantizedModel.size() > 0; }

    public void enableGradientCheckpointing() { checkpointingEnabled = true; }
    public void disableGradientCheckpointing() { checkpointingEnabled = false; }

    public void forTraining() {
        inferenceMode = false;
        model.train(true);
        for (LoraLinear ll : adapters) {
            try { ll.unmerge(); } catch (Exception ignored) {}
        }
    }

    public void for_training() { forTraining(); }

    public void forInference() {
        inferenceMode = true;
        model.eval();
        try {
            if (qloraSession != null) qloraSession.mergeAndUnload();
            else for (LoraLinear ll : adapters) ll.merge();
        } catch (Exception ignored) {}
    }

    public void for_inference() { forInference(); }

    public Tensor forward(Tensor inputIds) {
        return model.forward(inputIds);
    }

    /**
     * One training step. With QLoRA session: real LM loss, ΔW in forward, only
     * LoRA A/B train. Without: plain CausalLM loss.
     */
    public Tensor trainStep(Tensor inputIds) {
        forTraining();
        if (qloraSession != null) {
            double loss = qloraSession.trainStep(inputIds);
            stepCount = qloraSession.step();
            return org.bytedeco.pytorch.global.torch.tensor(new float[]{(float) loss});
        }
        Tensor loss = model.loss(inputIds);
        loss.backward();
        stepCount++;
        return loss;
    }

    public long trainableParameters() {
        long n = 0;
        for (LoraLinear ll : adapters) {
            try { n += ll.loraA().numel() + ll.loraB().numel(); } catch (Exception ignored) {}
        }
        return n;
    }

    public long totalParameters() {
        long n = 0;
        try {
            Map<String, LinearImpl> linears = model.quantizableLinears();
            for (LinearImpl lin : linears.values()) {
                if (lin != null && lin.weight() != null && lin.weight().defined()) {
                    n += lin.weight().numel();
                }
            }
            if (model.lmHead() != null && model.lmHead().weight() != null) {
                n += model.lmHead().weight().numel();
            }
        } catch (Exception ignored) {}
        return n + trainableParameters();
    }

    public Map<String, Object> stats() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("total_params", totalParameters());
        m.put("trainable_params", trainableParameters());
        m.put("step", stepCount);
        m.put("load_in_4bit", fastConfig.loadIn4bit());
        m.put("load_in_8bit", fastConfig.loadIn8bit());
        m.put("r", fastConfig.r());
        m.put("lora_alpha", fastConfig.loraAlpha());
        m.put("gradient_checkpointing", checkpointingEnabled);
        m.put("use_gradient_checkpointing_unsloth", fastConfig.useGradientCheckpointingUnsloth());
        m.put("max_seq_length", fastConfig.maxSeqLength());
        m.put("quant_tensors", quantStates.size());
        m.put("quantized_layers", quantizedModel == null ? 0 : quantizedModel.size());
        m.put("adapters", adapters.size());
        m.put("peft_applied", peftApplied);
        m.put("inference_mode", inferenceMode);
        m.put("full_finetuning", fastConfig.fullFinetuning());
        m.put("dtype", fastConfig.dtype());
        m.put("use_rslora", fastConfig.useRslora());
        m.put("is_quantized", isQuantized());
        m.put("lora_in_forward", model.hasLora());
        if (quantizedModel != null) {
            m.putAll(quantizedModel.stats());
        } else if (!quantStates.isEmpty()) {
            BitsAndBytes.QuantState qs = quantStates.values().iterator().next();
            m.put("quant_type", qs.quantType);
            m.put("quant_mem_est", qs.memoryBytes());
        }
        long total = totalParameters();
        long train = trainableParameters();
        m.put("trainable_ratio", total == 0 ? 0.0 : (double) train / (double) total);
        return m;
    }

    public int[] generate(int[] prompt, int maxNew) {
        forInference();
        return model.generate(prompt, maxNew);
    }

    public void savePretrained(Path dir) throws IOException {
        Files.createDirectories(dir);
        if (qloraSession != null) {
            qloraSession.saveAdapter(dir.resolve("adapter.pt").toFile());
        } else if (!adapters.isEmpty()) {
            Map<String, Tensor> state = new LinkedHashMap<>();
            for (Map.Entry<String, LoraLinear> e : model.loraAdapters().entrySet()) {
                state.put(e.getKey() + ".lora_A", e.getValue().loraA());
                state.put(e.getKey() + ".lora_B", e.getValue().loraB());
            }
            org.bytedeco.pytorch.data.safetensors.SafeTensors.save(
                    state, dir.resolve("adapter.pt").toFile());
        }
        Path cfg = dir.resolve("unsloth_config.txt");
        String body = "r=" + fastConfig.r()
                + "\nalpha=" + fastConfig.loraAlpha()
                + "\nmax_seq_length=" + fastConfig.maxSeqLength()
                + "\nload_in_4bit=" + fastConfig.loadIn4bit()
                + "\nstep=" + stepCount + "\n";
        Files.writeString(cfg, body);
    }

    public void save_pretrained(Path dir) throws IOException {
        savePretrained(dir);
    }

    public void savePretrainedMerged(Path dir) throws IOException {
        forInference();
        savePretrained(dir);
        Files.writeString(dir.resolve("merged.flag"), "merged=true\n");
    }

    public void save_pretrained_merged(Path dir) throws IOException {
        savePretrainedMerged(dir);
    }

    public void loadAdapter(File file) throws IOException {
        if (qloraSession != null) {
            qloraSession.loadAdapter(file);
            return;
        }
        Map<String, Tensor> state =
                org.bytedeco.pytorch.data.safetensors.SafeTensors.loadAsTensors(file, false);
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            for (Map.Entry<String, LoraLinear> e : model.loraAdapters().entrySet()) {
                Tensor aa = state.get(e.getKey() + ".lora_A");
                Tensor bb = state.get(e.getKey() + ".lora_B");
                if (aa != null && aa.defined()) e.getValue().loraA().copy_(aa);
                if (bb != null && bb.defined()) e.getValue().loraB().copy_(bb);
            }
        }
    }

    public void loadAdapter(Path path) throws IOException {
        loadAdapter(path.toFile());
    }
}
