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
package org.bytedeco.pytorch.llm.modules;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Tiny composable causal LM built purely from {@code llm.modules} layers.
 *
 * <p>Use the factories to get architecture-shaped mini models for tests,
 * distillation, and structure demos:
 * <ul>
 *   <li>{@link #llama} — pre-norm RMS + GQA + SwiGLU + RoPE</li>
 *   <li>{@link #qwen2} — like Llama but qkv bias</li>
 *   <li>{@link #qwen3} — QK-Norm</li>
 *   <li>{@link #gpt2} — post-norm LayerNorm + GELU + absolute pos</li>
 *   <li>{@link #glm} — fused SwiGLU</li>
 *   <li>{@link #deepseekMoe} — dense layers + periodic MoE layers</li>
 *   <li>{@link #mixtral} — every layer MoE</li>
 * </ul>
 *
 * <p>Not a weight-compatible HF loader — for that use
 * {@code org.bytedeco.pytorch.llm.transformers.modeling.*}. This class is the
 * reusable building kit.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MiniCausalLM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public enum Arch {
        LLAMA, QWEN2, QWEN3, GPT2, GLM, DEEPSEEK_MOE, MIXTRAL, GEMMA
    }

    /** Immutable config for mini models. */
    public static final class Config {
        public final Arch arch;
        public final long vocabSize;
        public final long hiddenSize;
        public final int numLayers;
        public final int nHeads;
        public final int nKvHeads;
        public final int headDim;
        public final long intermediateSize;
        public final long maxPositions;
        public final double ropeTheta;
        public final double rmsNormEps;
        public final double residDropout;
        public final boolean tieWordEmbeddings;
        public final int moeNumExperts;
        public final int moeTopK;
        /** Insert MoE every N layers (DeepSeek); 1 = all MoE. */
        public final int moeEveryN;
        public final boolean useAbsolutePos;

        private Config(Builder b) {
            this.arch = b.arch;
            this.vocabSize = b.vocabSize;
            this.hiddenSize = b.hiddenSize;
            this.numLayers = b.numLayers;
            this.nHeads = b.nHeads;
            this.nKvHeads = b.nKvHeads;
            this.headDim = b.headDim > 0 ? b.headDim : (int) (b.hiddenSize / b.nHeads);
            this.intermediateSize = b.intermediateSize > 0 ? b.intermediateSize : 4L * b.hiddenSize;
            this.maxPositions = b.maxPositions;
            this.ropeTheta = b.ropeTheta;
            this.rmsNormEps = b.rmsNormEps;
            this.residDropout = b.residDropout;
            this.tieWordEmbeddings = b.tieWordEmbeddings;
            this.moeNumExperts = b.moeNumExperts;
            this.moeTopK = b.moeTopK;
            this.moeEveryN = Math.max(1, b.moeEveryN);
            this.useAbsolutePos = b.useAbsolutePos;
        }

        public static Builder builder(Arch arch) {
            return new Builder(arch);
        }

        public static final class Builder {
            private final Arch arch;
            private long vocabSize = 1024;
            private long hiddenSize = 128;
            private int numLayers = 2;
            private int nHeads = 4;
            private int nKvHeads = 4;
            private int headDim;
            private long intermediateSize;
            private long maxPositions = 512;
            private double ropeTheta = 10000.0;
            private double rmsNormEps = 1e-6;
            private double residDropout;
            private boolean tieWordEmbeddings = true;
            private int moeNumExperts = 4;
            private int moeTopK = 2;
            private int moeEveryN = 2;
            private boolean useAbsolutePos;

            public Builder(Arch arch) {
                this.arch = Objects.requireNonNull(arch);
                switch (arch) {
                    case GPT2 -> {
                        useAbsolutePos = true;
                        residDropout = 0.1;
                        rmsNormEps = 1e-5;
                    }
                    case QWEN2 -> ropeTheta = 1_000_000.0;
                    case QWEN3 -> {
                        ropeTheta = 1_000_000.0;
                        nKvHeads = 2;
                    }
                    case DEEPSEEK_MOE, MIXTRAL -> {
                        moeNumExperts = 4;
                        moeTopK = 2;
                        nKvHeads = 2;
                    }
                    case GEMMA -> {
                        // GeGLU + RMS, smaller intermediate often
                        intermediateSize = 0; // default 4H unless set
                    }
                    default -> { }
                }
            }

            public Builder vocabSize(long v) { this.vocabSize = v; return this; }
            public Builder hiddenSize(long v) { this.hiddenSize = v; return this; }
            public Builder numLayers(int v) { this.numLayers = v; return this; }
            public Builder nHeads(int v) { this.nHeads = v; return this; }
            public Builder nKvHeads(int v) { this.nKvHeads = v; return this; }
            public Builder headDim(int v) { this.headDim = v; return this; }
            public Builder intermediateSize(long v) { this.intermediateSize = v; return this; }
            public Builder maxPositions(long v) { this.maxPositions = v; return this; }
            public Builder ropeTheta(double v) { this.ropeTheta = v; return this; }
            public Builder rmsNormEps(double v) { this.rmsNormEps = v; return this; }
            public Builder residDropout(double v) { this.residDropout = v; return this; }
            public Builder tieWordEmbeddings(boolean v) { this.tieWordEmbeddings = v; return this; }
            public Builder moe(int experts, int topK, int everyN) {
                this.moeNumExperts = experts;
                this.moeTopK = topK;
                this.moeEveryN = everyN;
                return this;
            }

            public Config build() { return new Config(this); }
        }
    }

    private final Config config;
    public final Embedding embed;
    public final List<DecoderLayer> layers = new ArrayList<>();
    public final Module norm;
    public final LinearImpl lm_head;

    public MiniCausalLM(Config config) {
        super("MiniCausalLM");
        this.config = Objects.requireNonNull(config);
        this.embed = register_module("embed",
                new Embedding(config.vocabSize, config.hiddenSize, config.maxPositions,
                        config.useAbsolutePos, config.residDropout));

        for (int i = 0; i < config.numLayers; i++) {
            layers.add(register_module("layers/" + i, buildLayer(config, i)));
        }

        if (config.arch == Arch.GPT2) {
            this.norm = register_module("norm", new LayerNorm(config.hiddenSize, config.rmsNormEps));
        } else {
            this.norm = register_module("norm", new RMSNorm(config.hiddenSize, config.rmsNormEps));
        }

        this.lm_head = register_module("lm_head",
                new LinearImpl(new LinearOptions(config.hiddenSize, config.vocabSize).bias(false)));
        if (config.tieWordEmbeddings) {
            try {
                lm_head.weight().set_(embed.weight());
            } catch (Throwable ignored) {
                // tying may fail if shapes differ; keep independent head
            }
        }
    }

    public static MiniCausalLM llama(long vocab, long hidden, int layers, int heads, int kvHeads) {
        return new MiniCausalLM(Config.builder(Arch.LLAMA)
                .vocabSize(vocab).hiddenSize(hidden).numLayers(layers)
                .nHeads(heads).nKvHeads(kvHeads).build());
    }

    public static MiniCausalLM qwen2(long vocab, long hidden, int layers, int heads, int kvHeads) {
        return new MiniCausalLM(Config.builder(Arch.QWEN2)
                .vocabSize(vocab).hiddenSize(hidden).numLayers(layers)
                .nHeads(heads).nKvHeads(kvHeads).build());
    }

    public static MiniCausalLM qwen3(long vocab, long hidden, int layers, int heads, int kvHeads, int headDim) {
        return new MiniCausalLM(Config.builder(Arch.QWEN3)
                .vocabSize(vocab).hiddenSize(hidden).numLayers(layers)
                .nHeads(heads).nKvHeads(kvHeads).headDim(headDim).build());
    }

    public static MiniCausalLM gpt2(long vocab, long hidden, int layers, int heads) {
        return new MiniCausalLM(Config.builder(Arch.GPT2)
                .vocabSize(vocab).hiddenSize(hidden).numLayers(layers)
                .nHeads(heads).nKvHeads(heads).build());
    }

    public static MiniCausalLM glm(long vocab, long hidden, int layers, int heads, int kvHeads) {
        return new MiniCausalLM(Config.builder(Arch.GLM)
                .vocabSize(vocab).hiddenSize(hidden).numLayers(layers)
                .nHeads(heads).nKvHeads(kvHeads).build());
    }

    public static MiniCausalLM deepseekMoe(long vocab, long hidden, int layers, int heads, int kvHeads,
                                           int experts, int topK) {
        return new MiniCausalLM(Config.builder(Arch.DEEPSEEK_MOE)
                .vocabSize(vocab).hiddenSize(hidden).numLayers(layers)
                .nHeads(heads).nKvHeads(kvHeads)
                .moe(experts, topK, 2).build());
    }

    public static MiniCausalLM mixtral(long vocab, long hidden, int layers, int heads, int kvHeads,
                                       int experts, int topK) {
        return new MiniCausalLM(Config.builder(Arch.MIXTRAL)
                .vocabSize(vocab).hiddenSize(hidden).numLayers(layers)
                .nHeads(heads).nKvHeads(kvHeads)
                .moe(experts, topK, 1).build());
    }

    /** Ultra-tiny default for unit tests: vocab=128, H=64, L=2, heads=4. */
    public static MiniCausalLM tiny() {
        return llama(128, 64, 2, 4, 2);
    }

    public Config config() { return config; }

    @Override
    public Tensor forward(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        long T = ids.size(1);
        if (T > config.maxPositions) {
            throw new IllegalArgumentException("T=" + T + " > maxPositions=" + config.maxPositions);
        }
        Tensor x = embed.forward(ids);
        for (DecoderLayer layer : layers) {
            x = layer.forward(x);
        }
        x = norm.forward(x);
        return lm_head.forward(x);
    }

    /** Shifted next-token cross-entropy loss. */
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
     * Cache-aware forward for incremental decode.
     *
     * @return {logits [B,T,V], newKs[L], newVs[L]}
     */
    public Tensor[] forwardCached(Tensor inputIds, long positionOffset,
                                  Tensor[] pastKs, Tensor[] pastVs) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        Tensor x = config.useAbsolutePos
                ? embed.forward(ids, positionOffset)
                : embed.forward(ids);
        Tensor[] newKs = new Tensor[layers.size()];
        Tensor[] newVs = new Tensor[layers.size()];
        for (int i = 0; i < layers.size(); i++) {
            Tensor[] out = layers.get(i).forwardCached(x, positionOffset,
                    pastKs != null ? pastKs[i] : null,
                    pastVs != null ? pastVs[i] : null);
            x = out[0];
            newKs[i] = out[1];
            newVs[i] = out[2];
        }
        x = norm.forward(x);
        Tensor logits = lm_head.forward(x);
        // pack: logits + interleaved? keep simple array: [logits] separate return via holder
        Tensor[] result = new Tensor[1 + newKs.length + newVs.length];
        result[0] = logits;
        System.arraycopy(newKs, 0, result, 1, newKs.length);
        System.arraycopy(newVs, 0, result, 1 + newKs.length, newVs.length);
        return result;
    }

    public int numLayers() { return layers.size(); }

    private static DecoderLayer buildLayer(Config c, int idx) {
        boolean useMoe = switch (c.arch) {
            case MIXTRAL -> true;
            case DEEPSEEK_MOE -> (idx % c.moeEveryN) == (c.moeEveryN - 1);
            default -> false;
        };

        return switch (c.arch) {
            case GPT2 -> DecoderLayer.gpt2(c.hiddenSize, c.nHeads, c.intermediateSize,
                    c.residDropout, idx);
            case QWEN2 -> new DecoderLayer.DecoderLayerBuilder(c.hiddenSize, c.nHeads)
                    .nKvHeads(c.nKvHeads).headDim(c.headDim)
                    .intermediateSize(c.intermediateSize).ropeTheta(c.ropeTheta)
                    .qkvBias(true).rmsNormEps(c.rmsNormEps).layerIdx(idx).build();
            case QWEN3 -> new DecoderLayer.DecoderLayerBuilder(c.hiddenSize, c.nHeads)
                    .nKvHeads(c.nKvHeads).headDim(c.headDim)
                    .intermediateSize(c.intermediateSize).ropeTheta(c.ropeTheta)
                    .useQkNorm(true).rmsNormEps(c.rmsNormEps).layerIdx(idx).build();
            case GLM -> DecoderLayer.glm(c.hiddenSize, c.nHeads, c.nKvHeads,
                    c.intermediateSize, c.ropeTheta, idx);
            case GEMMA -> new DecoderLayer.DecoderLayerBuilder(c.hiddenSize, c.nHeads)
                    .nKvHeads(c.nKvHeads).headDim(c.headDim)
                    .intermediateSize(c.intermediateSize).ropeTheta(c.ropeTheta)
                    .mlpType(DecoderLayer.MlpType.GEGLU)
                    .rmsNormEps(c.rmsNormEps).layerIdx(idx).build();
            case MIXTRAL, DEEPSEEK_MOE -> {
                if (useMoe) {
                    yield DecoderLayer.deepseekMoe(c.hiddenSize, c.nHeads, c.nKvHeads,
                            c.intermediateSize, c.ropeTheta,
                            c.moeNumExperts, c.moeTopK, idx);
                }
                yield DecoderLayer.llama(c.hiddenSize, c.nHeads, c.nKvHeads,
                        c.intermediateSize, c.ropeTheta, idx);
            }
            default -> DecoderLayer.llama(c.hiddenSize, c.nHeads, c.nKvHeads,
                    c.intermediateSize, c.ropeTheta, idx);
        };
    }
}
