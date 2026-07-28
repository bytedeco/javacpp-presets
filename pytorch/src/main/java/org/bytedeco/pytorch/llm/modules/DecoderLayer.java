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
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;

/**
 * Transformer decoder block used by modern causal LMs.
 *
 * <p>Pre-Norm residual (Llama / Qwen / DeepSeek / GPT-NeoX style):
 * <pre>
 *   x = x + attn(norm1(x))
 *   x = x + mlp(norm2(x))
 * </pre>
 *
 * <p>Post-Norm residual (classic GPT-2):
 * <pre>
 *   x = norm1(x + attn(x))
 *   x = norm2(x + mlp(x))
 * </pre>
 *
 * <p>Supports RMSNorm or LayerNorm, SwiGLU / GELU / fused / MoE FFN, residual dropout.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DecoderLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public enum NormType { RMS, LAYER }
    public enum MlpType { SWIGLU, FUSED_SWIGLU, GELU, RELU, GEGLU, MOE }
    public enum ResidualStyle { PRE_NORM, POST_NORM }

    public final Module input_layernorm;
    public final Attention self_attn;
    public final Module post_attention_layernorm;
    public final Module mlp;
    public final DropoutImpl resid_dropout;

    private final ResidualStyle residualStyle;
    private final long hiddenSize;
    private final int layerIdx;

    public DecoderLayer(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                        long intermediateSize, double ropeTheta,
                        NormType normType, MlpType mlpType, ResidualStyle residualStyle,
                        boolean qkvBias, boolean useQkNorm, double rmsNormEps,
                        double residDropout, int layerIdx,
                        // MoE optional
                        int moeNumExperts, int moeTopK) {
        super("DecoderLayer" + layerIdx);
        this.hiddenSize = hiddenSize;
        this.residualStyle = residualStyle == null ? ResidualStyle.PRE_NORM : residualStyle;
        this.layerIdx = layerIdx;

        double eps = rmsNormEps > 0 ? rmsNormEps : 1e-6;
        this.input_layernorm = register_module("input_layernorm",
                makeNorm(normType, hiddenSize, eps));
        this.self_attn = register_module("self_attn",
                new Attention(hiddenSize, nHeads, nKvHeads,
                        headDim > 0 ? headDim : (int) (hiddenSize / nHeads),
                        ropeTheta, 1.0,
                        residualStyle != ResidualStyle.POST_NORM || ropeTheta > 0, // keep rope default on
                        qkvBias, residualStyle == ResidualStyle.POST_NORM /*oBias for gpt2*/,
                        useQkNorm, eps, false, -1, 0.0, true));
        // Fix: for GPT-2 post-norm we typically disable RoPE — rebuild if needed is heavy;
        // factories below set correct flags.

        this.post_attention_layernorm = register_module("post_attention_layernorm",
                makeNorm(normType, hiddenSize, eps));
        this.mlp = register_module("mlp",
                makeMlp(mlpType, hiddenSize, intermediateSize, moeNumExperts, moeTopK));
        this.resid_dropout = register_module("resid_dropout",
                new DropoutImpl(Math.max(0.0, residDropout)));
    }

    /** Llama / Mistral pre-norm + SwiGLU + GQA. */
    public static DecoderLayer llama(long hiddenSize, int nHeads, int nKvHeads,
                                     long intermediateSize, double ropeTheta, int layerIdx) {
        return new DecoderLayer(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                intermediateSize, ropeTheta,
                NormType.RMS, MlpType.SWIGLU, ResidualStyle.PRE_NORM,
                false, false, 1e-6, 0.0, layerIdx, 0, 0);
    }

    /** Qwen2 pre-norm + SwiGLU + qkv bias. */
    public static DecoderLayer qwen2(long hiddenSize, int nHeads, int nKvHeads,
                                     long intermediateSize, double ropeTheta, int layerIdx) {
        DecoderLayer layer = new DecoderLayer(hiddenSize, nHeads, nKvHeads,
                (int) (hiddenSize / nHeads), intermediateSize, ropeTheta,
                NormType.RMS, MlpType.SWIGLU, ResidualStyle.PRE_NORM,
                true, false, 1e-6, 0.0, layerIdx, 0, 0);
        return layer;
    }

    /** Qwen3 with QK-Norm. */
    public static DecoderLayer qwen3(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                                     long intermediateSize, double ropeTheta,
                                     double rmsNormEps, int layerIdx) {
        return new DecoderLayer(hiddenSize, nHeads, nKvHeads, headDim,
                intermediateSize, ropeTheta,
                NormType.RMS, MlpType.SWIGLU, ResidualStyle.PRE_NORM,
                false, true, rmsNormEps, 0.0, layerIdx, 0, 0);
    }

    /** GPT-2 post-norm + GELU + LayerNorm + MHA. */
    public static DecoderLayer gpt2(long hiddenSize, int nHeads, long intermediateSize,
                                    double residDropout, int layerIdx) {
        return new DecoderLayerBuilder(hiddenSize, nHeads)
                .nKvHeads(nHeads)
                .intermediateSize(intermediateSize)
                .normType(NormType.LAYER)
                .mlpType(MlpType.GELU)
                .residualStyle(ResidualStyle.POST_NORM)
                .qkvBias(true)
                .useRope(false)
                .residDropout(residDropout)
                .layerIdx(layerIdx)
                .build();
    }

    /** GLM fused SwiGLU. */
    public static DecoderLayer glm(long hiddenSize, int nHeads, int nKvHeads,
                                   long intermediateSize, double ropeTheta, int layerIdx) {
        return new DecoderLayer(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                intermediateSize, ropeTheta,
                NormType.RMS, MlpType.FUSED_SWIGLU, ResidualStyle.PRE_NORM,
                false, false, 1e-5, 0.0, layerIdx, 0, 0);
    }

    /** DeepSeek-style dense layer (pre-norm SwiGLU); pair with {@link MoE} for sparse layers. */
    public static DecoderLayer deepseekDense(long hiddenSize, int nHeads, int nKvHeads,
                                             long intermediateSize, double ropeTheta, int layerIdx) {
        return llama(hiddenSize, nHeads, nKvHeads, intermediateSize, ropeTheta, layerIdx);
    }

    /** DeepSeek-style MoE layer. */
    public static DecoderLayer deepseekMoe(long hiddenSize, int nHeads, int nKvHeads,
                                           long intermediateSize, double ropeTheta,
                                           int numExperts, int topK, int layerIdx) {
        return new DecoderLayer(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                intermediateSize, ropeTheta,
                NormType.RMS, MlpType.MOE, ResidualStyle.PRE_NORM,
                false, false, 1e-6, 0.0, layerIdx, numExperts, topK);
    }

    public long hiddenSize() { return hiddenSize; }
    public int layerIdx() { return layerIdx; }
    public ResidualStyle residualStyle() { return residualStyle; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    /** {out, newK, newV} */
    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
        if (residualStyle == ResidualStyle.PRE_NORM) {
            Tensor h = input_layernorm.forward(x);
            Tensor[] attOut = self_attn.forwardCached(h, positionOffset, pastK, pastV);
            Tensor out = x.add(resid_dropout.forward(attOut[0]));
            out = out.add(resid_dropout.forward(mlp.forward(post_attention_layernorm.forward(out))));
            return new Tensor[]{out, attOut[1], attOut[2]};
        } else {
            // Post-Norm GPT-2
            Tensor[] attOut = self_attn.forwardCached(x, positionOffset, pastK, pastV);
            Tensor out = input_layernorm.forward(x.add(resid_dropout.forward(attOut[0])));
            out = post_attention_layernorm.forward(out.add(resid_dropout.forward(mlp.forward(out))));
            return new Tensor[]{out, attOut[1], attOut[2]};
        }
    }

    private static Module makeNorm(NormType type, long hidden, double eps) {
        if (type == NormType.LAYER) {
            return new LayerNorm(hidden, eps);
        }
        return new RMSNorm(hidden, eps);
    }

    private static Module makeMlp(MlpType type, long hidden, long intermediate,
                                  int moeExperts, int moeTopK) {
        return switch (type == null ? MlpType.SWIGLU : type) {
            case FUSED_SWIGLU -> new Mlp.FusedSwiGLU(hidden, intermediate);
            case GELU -> new Mlp.GeluMlp(hidden, intermediate);
            case RELU -> new Mlp.ReluMlp(hidden, intermediate);
            case GEGLU -> new Mlp.GeGLU(hidden, intermediate);
            case MOE -> new MoE(hidden, intermediate,
                    Math.max(2, moeExperts), Math.max(1, moeTopK));
            default -> new Mlp.SwiGLU(hidden, intermediate);
        };
    }

    // ---- builder for full control (GPT-2 rope off etc.) ----

    public static final class DecoderLayerBuilder {
        private final long hiddenSize;
        private final int nHeads;
        private int nKvHeads;
        private int headDim;
        private long intermediateSize;
        private double ropeTheta = 10000.0;
        private NormType normType = NormType.RMS;
        private MlpType mlpType = MlpType.SWIGLU;
        private ResidualStyle residualStyle = ResidualStyle.PRE_NORM;
        private boolean qkvBias;
        private boolean useRope = true;
        private boolean useQkNorm;
        private double rmsNormEps = 1e-6;
        private double residDropout;
        private int layerIdx;
        private int moeNumExperts = 8;
        private int moeTopK = 2;
        private boolean oBias;

        public DecoderLayerBuilder(long hiddenSize, int nHeads) {
            this.hiddenSize = hiddenSize;
            this.nHeads = nHeads;
            this.nKvHeads = nHeads;
            this.headDim = (int) (hiddenSize / nHeads);
            this.intermediateSize = 4L * hiddenSize;
        }

        public DecoderLayerBuilder nKvHeads(int v) { this.nKvHeads = v; return this; }
        public DecoderLayerBuilder headDim(int v) { this.headDim = v; return this; }
        public DecoderLayerBuilder intermediateSize(long v) { this.intermediateSize = v; return this; }
        public DecoderLayerBuilder ropeTheta(double v) { this.ropeTheta = v; return this; }
        public DecoderLayerBuilder normType(NormType v) { this.normType = v; return this; }
        public DecoderLayerBuilder mlpType(MlpType v) { this.mlpType = v; return this; }
        public DecoderLayerBuilder residualStyle(ResidualStyle v) { this.residualStyle = v; return this; }
        public DecoderLayerBuilder qkvBias(boolean v) { this.qkvBias = v; return this; }
        public DecoderLayerBuilder oBias(boolean v) { this.oBias = v; return this; }
        public DecoderLayerBuilder useRope(boolean v) { this.useRope = v; return this; }
        public DecoderLayerBuilder useQkNorm(boolean v) { this.useQkNorm = v; return this; }
        public DecoderLayerBuilder rmsNormEps(double v) { this.rmsNormEps = v; return this; }
        public DecoderLayerBuilder residDropout(double v) { this.residDropout = v; return this; }
        public DecoderLayerBuilder layerIdx(int v) { this.layerIdx = v; return this; }
        public DecoderLayerBuilder moe(int experts, int topK) {
            this.mlpType = MlpType.MOE;
            this.moeNumExperts = experts;
            this.moeTopK = topK;
            return this;
        }

        public DecoderLayer build() {
            // Custom construction to honor useRope / oBias
            DecoderLayer layer = new DecoderLayer(hiddenSize, nHeads, nKvHeads, headDim,
                    intermediateSize, ropeTheta, normType, mlpType, residualStyle,
                    qkvBias, useQkNorm, rmsNormEps, residDropout, layerIdx,
                    moeNumExperts, moeTopK);
            // Attention already created inside; for useRope=false (GPT-2) we need a dedicated path.
            // Rebuild attention if flags differ from constructor defaults.
            if (!useRope || oBias || residualStyle == ResidualStyle.POST_NORM) {
                // Replace self_attn by re-registering is not trivial; instead construct via alt ctor path:
                return new DecoderLayer(hiddenSize, nHeads, nKvHeads, headDim, intermediateSize,
                        ropeTheta, useRope, qkvBias, oBias, useQkNorm, rmsNormEps,
                        normType, mlpType, residualStyle, residDropout, layerIdx,
                        moeNumExperts, moeTopK);
            }
            return layer;
        }
    }

    /** Full-control private-ish constructor used by builder. */
    DecoderLayer(long hiddenSize, int nHeads, int nKvHeads, int headDim, long intermediateSize,
                 double ropeTheta, boolean useRope, boolean qkvBias, boolean oBias,
                 boolean useQkNorm, double rmsNormEps,
                 NormType normType, MlpType mlpType, ResidualStyle residualStyle,
                 double residDropout, int layerIdx, int moeNumExperts, int moeTopK) {
        super("DecoderLayer" + layerIdx);
        this.hiddenSize = hiddenSize;
        this.residualStyle = residualStyle == null ? ResidualStyle.PRE_NORM : residualStyle;
        this.layerIdx = layerIdx;
        double eps = rmsNormEps > 0 ? rmsNormEps : 1e-6;
        this.input_layernorm = register_module("input_layernorm", makeNorm(normType, hiddenSize, eps));
        this.self_attn = register_module("self_attn",
                new Attention(hiddenSize, nHeads, nKvHeads,
                        headDim > 0 ? headDim : (int) (hiddenSize / nHeads),
                        ropeTheta, 1.0, useRope, qkvBias, oBias, useQkNorm, eps,
                        false, -1, 0.0, true));
        this.post_attention_layernorm = register_module("post_attention_layernorm",
                makeNorm(normType, hiddenSize, eps));
        this.mlp = register_module("mlp", makeMlp(mlpType, hiddenSize, intermediateSize,
                moeNumExperts, moeTopK));
        this.resid_dropout = register_module("resid_dropout",
                new DropoutImpl(Math.max(0.0, residDropout)));
    }
}
