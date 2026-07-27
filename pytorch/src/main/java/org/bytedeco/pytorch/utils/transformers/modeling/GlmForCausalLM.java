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
package org.bytedeco.pytorch.utils.transformers.modeling;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.utils.transformers.generation.Generator;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * GLM-Edge / ChatGLM-style causal LM ({@code GlmForCausalLM}).
 *
 * <p>HF-identical parameter names under {@code model.*} / {@code lm_head}:
 * <ul>
 *   <li>Attention: {@code q_proj}/{@code k_proj}/{@code v_proj}/{@code o_proj} (no bias)</li>
 *   <li>MLP: fused {@code gate_up_proj} + {@code down_proj} ({@link ModelingMlp.FusedSwiGLU})</li>
 *   <li>Norm: RMSNorm on {@code input_layernorm}/{@code post_attention_layernorm}/{@code norm}</li>
 *   <li>Optional {@code tie_word_embeddings} → share {@code lm_head.weight} with embed</li>
 * </ul>
 *
 * <p>Supports both full {@link #forward} and cache-aware {@link #forwardCached} for vLLM.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GlmForCausalLM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final PretrainedConfig config;
    private final GlmModel model;
    private final LinearImpl lm_head;

    public GlmForCausalLM(PretrainedConfig config) {
        super("GlmForCausalLM");
        this.config = Objects.requireNonNull(config, "config");
        this.model = register_module("model", new GlmModel(config));
        this.lm_head = register_module("lm_head",
                new LinearImpl(new LinearOptions(config.hiddenSize(), config.vocabSize()).bias(false)));
        // actual share after weight load (see retieWordEmbeddings / AutoModel)
        if (config.tieWordEmbeddings()) {
            try {
                lm_head.weight().requires_grad_(false);
                model.embed_tokens.weight().requires_grad_(false);
                lm_head.weight().set_(model.embed_tokens.weight());
            } catch (Throwable ignored) {}
        }
    }

    public static GlmForCausalLM fromConfig(PretrainedConfig config) {
        return new GlmForCausalLM(config);
    }

    public PretrainedConfig config() {
        return config;
    }

    public GlmModel model() {
        return model;
    }

    public LinearImpl lmHead() {
        return lm_head;
    }

    /** Re-apply lm_head ← embed_tokens share after ZERO_COPY/COPY load. */
    public boolean retieWordEmbeddings() {
        if (!config.tieWordEmbeddings() || lm_head == null || model == null) return false;
        try {
            Tensor dest = lm_head.weight();
            Tensor src = model.embed_tokens.weight();
            if (dest == null || src == null || !dest.defined() || !src.defined()) return false;
            try { dest.requires_grad_(false); } catch (Throwable ignored) {}
            try { src.requires_grad_(false); } catch (Throwable ignored) {}
            dest.set_(src);
            return true;
        } catch (Throwable t) {
            try {
                lm_head.weight().copy_(model.embed_tokens.weight());
                return true;
            } catch (Throwable t2) {
                System.out.println("[DEBUG] GlmForCausalLM.retieWordEmbeddings failed: " + t2.getMessage());
                return false;
            }
        }
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        return lm_head.forward(model.forward(inputIds));
    }

    public Tensor loss(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        Tensor logits = forward(ids);
        Tensor shiftLogits = logits.slice(1, new LongOptional(0), new LongOptional(logits.size(1) - 1), 1)
                .contiguous();
        Tensor shiftLabels = ids.slice(1, new LongOptional(1), new LongOptional(ids.size(1)), 1)
                .contiguous();
        long V = logits.size(2);
        return cross_entropy(shiftLogits.reshape(-1, V), shiftLabels.reshape(-1));
    }

    public int[] generate(int[] promptIds, int maxNewTokens) {
        return generate(promptIds, GenerationConfig.builder().maxNewTokens(maxNewTokens).build());
    }

    public int[] generate(int[] promptIds, GenerationConfig gen) {
        GenerationConfig g = gen == null ? GenerationConfig.greedy() : gen;
        if (g.eosTokenIds.isEmpty()) {
            GenerationConfig.Builder b = g.toBuilder().eosTokenId(config.eosTokenId());
            // multi-eos from generation_config / config extras when present
            Object eos = config.extra().get("eos_token_id");
            if (eos instanceof List<?> list) {
                for (Object o : list) {
                    if (o instanceof Number n) b.eosTokenId(n.intValue());
                }
            }
            g = b.build();
        }
        return Generator.generate(this, promptIds, g, config.maxPositionEmbeddings());
    }

    /**
     * Cache-aware causal LM forward for incremental decode serving (vLLM).
     */
    public CachedForwardResult forwardCached(Tensor inputIds, long positionOffset,
                                              Tensor[] pastKs, Tensor[] pastVs) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        long T = ids.size(1);
        if (positionOffset + T > config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("Sequence length " + (positionOffset + T)
                    + " exceeds max_position_embeddings=" + config.maxPositionEmbeddings());
        }
        Tensor x = model.embed_tokens.forward(ids);
        CachedForwardResult result = model.forwardCached(x, positionOffset, pastKs, pastVs);
        Tensor logits = lm_head.forward(result.hidden());
        return new CachedForwardResult(logits, result.newKs, result.newVs);
    }

    // ---- model / layers -----------------------------------------------------

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class GlmModel extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final EmbeddingImpl embed_tokens;
        public final List<GlmDecoderLayer> layers = new ArrayList<>();
        public final RMSNorm norm;
        private final PretrainedConfig config;

        public GlmModel(PretrainedConfig config) {
            super("GlmModel");
            this.config = config;
            this.embed_tokens = register_module("embed_tokens",
                    new EmbeddingImpl(config.vocabSize(), config.hiddenSize()));
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                layers.add(register_module("layers/" + i, new GlmDecoderLayer(config, i)));
            }
            this.norm = register_module("norm", new RMSNorm(config.hiddenSize(), config.rmsNormEps()));
        }

        @Override
        public Tensor forward(Tensor inputIds) {
            Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
            long T = ids.size(1);
            if (T > config.maxPositionEmbeddings()) {
                throw new IllegalArgumentException("Sequence length " + T
                        + " exceeds max_position_embeddings=" + config.maxPositionEmbeddings());
            }
            Tensor x = embed_tokens.forward(ids);
            for (GlmDecoderLayer layer : layers) {
                x = layer.forward(x);
            }
            return norm.forward(x);
        }

        public CachedForwardResult forwardCached(Tensor x, long positionOffset,
                                                  Tensor[] pastKs, Tensor[] pastVs) {
            Tensor[] newKs = new Tensor[config.numHiddenLayers()];
            Tensor[] newVs = new Tensor[config.numHiddenLayers()];
            for (int i = 0; i < layers.size(); i++) {
                Tensor[] out = layers.get(i).forwardCached(x, positionOffset,
                        pastKs != null ? pastKs[i] : null,
                        pastVs != null ? pastVs[i] : null);
                x = out[0];
                newKs[i] = out[1];
                newVs[i] = out[2];
            }
            x = norm.forward(x);
            return new CachedForwardResult(x, newKs, newVs);
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class GlmDecoderLayer extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final RMSNorm input_layernorm;
        public final ModelingAttention self_attn;
        public final RMSNorm post_attention_layernorm;
        public final ModelingMlp.FusedSwiGLU mlp;

        public GlmDecoderLayer(PretrainedConfig cfg, int layerIdx) {
            super("GlmDecoderLayer" + layerIdx);
            this.input_layernorm = register_module("input_layernorm",
                    new RMSNorm(cfg.hiddenSize(), cfg.rmsNormEps()));
            // GLM-Edge: explicit head_dim, GQA, RoPE, no qkv bias, no qk-norm
            int headDim = cfg.headDim() > 0 ? cfg.headDim() : (cfg.hiddenSize() / cfg.numAttentionHeads());
            this.self_attn = register_module("self_attn",
                    new ModelingAttention(
                            cfg.hiddenSize(),
                            cfg.numAttentionHeads(),
                            cfg.numKeyValueHeads(),
                            headDim,
                            cfg.ropeTheta(),
                            /*useRope=*/true,
                            /*qkvBias=*/cfg.attentionBias(),
                            /*useQkNorm=*/false,
                            cfg.rmsNormEps()));
            this.post_attention_layernorm = register_module("post_attention_layernorm",
                    new RMSNorm(cfg.hiddenSize(), cfg.rmsNormEps()));
            this.mlp = register_module("mlp",
                    new ModelingMlp.FusedSwiGLU(cfg.hiddenSize(), cfg.intermediateSize()));
        }

        @Override
        public Tensor forward(Tensor x) {
            x = x.add(self_attn.forward(input_layernorm.forward(x)));
            x = x.add(mlp.forward(post_attention_layernorm.forward(x)));
            return x;
        }

        /** Cache-aware layer forward. {out [B,T,C], newK, newV}. */
        public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
            Tensor h = input_layernorm.forward(x);
            Tensor[] attOut = self_attn.forwardCached(h, positionOffset, pastK, pastV);
            Tensor out = x.add(attOut[0]);
            out = out.add(mlp.forward(post_attention_layernorm.forward(out)));
            return new Tensor[]{out, attOut[1], attOut[2]};
        }
    }
}
