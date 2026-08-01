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
package org.bytedeco.pytorch.llm.ktransformers.model;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtMoEConfig;
import org.bytedeco.pytorch.llm.ktransformers.moe.RoutedMoE;
import org.bytedeco.pytorch.llm.ktransformers.util.DeviceBudget;
import org.bytedeco.pytorch.llm.modules.Attention;
import org.bytedeco.pytorch.llm.modules.Embedding;
import org.bytedeco.pytorch.llm.modules.RMSNorm;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Tiny MoE causal LM for KT CI / demos — random weights, end-to-end generate + SFT step.
 *
 * <p>Architecture: Embedding → N × (RMSNorm → Attention → residual → RMSNorm →
 * {@link RoutedMoE} → residual) → RMSNorm → lm_head.
 *
 * <p>Not weight-compatible with HF checkpoints; use {@code inject.families} for real models.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class KtMiniMoECausalLM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public static final class Layer extends Module {
        public final RMSNorm inputNorm;
        public final Attention attn;
        public final RMSNorm postNorm;
        public final RoutedMoE moe;

        public Layer(long hidden, int nHeads, int nKvHeads, long intermediate,
                     KtMoEConfig moeCfg, DeviceBudget budget, int layerIdx) {
            super("KtMiniMoELayer");
            this.inputNorm = register_module("input_norm", new RMSNorm(hidden));
            this.attn = register_module("attn",
                    Attention.llama(hidden, nHeads, nKvHeads, 10000.0));
            this.postNorm = register_module("post_norm", new RMSNorm(hidden));
            this.moe = register_module("moe", new RoutedMoE(hidden, intermediate, moeCfg, budget));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = x.add(attn.forward(inputNorm.forward(x)));
            return h.add(moe.forward(postNorm.forward(h)));
        }
    }

    private final KtConfig config;
    public final Embedding embed;
    public final List<Layer> layers = new ArrayList<>();
    public final RMSNorm norm;
    public final LinearImpl lmHead;
    private final long hiddenSize;
    private final int vocabSize;

    public KtMiniMoECausalLM(KtConfig config) {
        super("KtMiniMoECausalLM");
        this.config = Objects.requireNonNull(config, "config");
        this.hiddenSize = config.hiddenSize();
        this.vocabSize = config.vocabSize();
        int nHeads = Math.max(1, (int) (hiddenSize / 16));
        if (hiddenSize % nHeads != 0) {
            nHeads = 4;
            while (nHeads > 1 && hiddenSize % nHeads != 0) nHeads--;
        }
        int nKvHeads = Math.max(1, nHeads / 2);
        DeviceBudget budget = DeviceBudget.mini();

        this.embed = register_module("embed",
                new Embedding(vocabSize, hiddenSize, config.inference().maxSeqLen(), false, 0.0));
        for (int i = 0; i < config.numLayers(); i++) {
            layers.add(register_module("layers/" + i,
                    new Layer(hiddenSize, nHeads, nKvHeads, config.intermediateSize(),
                            config.moe(), budget, i)));
        }
        this.norm = register_module("norm", new RMSNorm(hiddenSize));
        this.lmHead = register_module("lm_head",
                new LinearImpl(new LinearOptions(hiddenSize, vocabSize).bias(false)));
    }

    public static KtMiniMoECausalLM miniDemo() {
        return new KtMiniMoECausalLM(KtConfig.miniDemo());
    }

    public KtConfig config() { return config; }
    public int numLayers() { return layers.size(); }
    public long hiddenSize() { return hiddenSize; }
    public int vocabSize() { return vocabSize; }

    /** Aggregate MoE metrics across layers. */
    public java.util.Map<String, Double> moeMetrics() {
        java.util.Map<String, Double> all = new java.util.LinkedHashMap<>();
        for (int i = 0; i < layers.size(); i++) {
            for (var e : layers.get(i).moe.metrics().toMetricMap().entrySet()) {
                all.put("layer" + i + "/" + e.getKey(), e.getValue());
            }
        }
        return all;
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        Tensor x = embed.forward(ids);
        for (Layer layer : layers) {
            x = layer.forward(x);
        }
        x = norm.forward(x);
        return lmHead.forward(x);
    }

    /** Shifted next-token CE loss for SFT single-step tests. */
    public Tensor loss(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        Tensor logits = forward(ids);
        Tensor shiftLogits = logits.slice(1, new LongOptional(0),
                new LongOptional(logits.size(1) - 1), 1).contiguous();
        Tensor shiftLabels = ids.slice(1, new LongOptional(1),
                new LongOptional(ids.size(1)), 1).contiguous();
        long V = logits.size(2);
        return cross_entropy(shiftLogits.reshape(-1, V), shiftLabels.reshape(-1));
    }

    /**
     * Greedy decode for {@code maxNew} tokens. Returns full sequence including prompt.
     */
    public int[] generateGreedy(int[] prompt, int maxNew) {
        Objects.requireNonNull(prompt, "prompt");
        if (prompt.length == 0) throw new IllegalArgumentException("empty prompt");
        List<Integer> seq = new ArrayList<>(prompt.length + maxNew);
        for (int t : prompt) seq.add(t);
        for (int step = 0; step < maxNew; step++) {
            long[] ids = new long[seq.size()];
            for (int i = 0; i < seq.size(); i++) {
                ids[i] = Math.floorMod(seq.get(i), vocabSize);
            }
            Tensor input = org.bytedeco.pytorch.global.torch.tensor(ids).unsqueeze(0);
            Tensor logits = forward(input); // [1, T, V]
            Tensor last = logits.slice(1, new LongOptional(logits.size(1) - 1),
                    new LongOptional(logits.size(1)), 1).squeeze(0).squeeze(0); // [V]
            long next = last.argmax().item_long();
            seq.add((int) next);
            input.close();
            logits.close();
            last.close();
        }
        int[] out = new int[seq.size()];
        for (int i = 0; i < seq.size(); i++) out[i] = seq.get(i);
        return out;
    }
}
