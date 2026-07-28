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
package org.bytedeco.pytorch.llm.transformers.modeling;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.generation.Generator;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Llama / Mistral-style causal LM with HF-identical parameter names
 * (same layout as Qwen2 but q/k/v without bias).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LlamaForCausalLM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final PretrainedConfig config;
    private final LlamaModel model;
    private final LinearImpl lm_head;

    public LlamaForCausalLM(PretrainedConfig config) {
        super("LlamaForCausalLM");
        this.config = Objects.requireNonNull(config, "config");
        this.model = register_module("model", new LlamaModel(config));
        this.lm_head = register_module("lm_head",
                new LinearImpl(new LinearOptions(config.hiddenSize(), config.vocabSize()).bias(false)));
        if (config.tieWordEmbeddings()) {
            try {
                lm_head.weight().set_(model.embed_tokens.weight());
            } catch (Throwable ignored) {}
        }
    }

    public static LlamaForCausalLM fromConfig(PretrainedConfig config) {
        return new LlamaForCausalLM(config);
    }

    public PretrainedConfig config() {
        return config;
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
            g = g.toBuilder().eosTokenId(config.eosTokenId()).build();
        }
        return Generator.generate(this, promptIds, g, config.maxPositionEmbeddings());
    }

    /**
     * Cache-aware causal LM forward for incremental decode serving.
     *
     * @param inputIds       [B,T] token ids
     * @param positionOffset RoPE start position
     * @param pastKs         [numLayers] past K
     * @param pastVs         [numLayers] past V
     * @return logits + per-layer new K/V
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

    public LlamaModel model() {
        return model;
    }

    public LinearImpl lmHead() {
        return lm_head;
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class LlamaModel extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final EmbeddingImpl embed_tokens;
        public final List<LlamaDecoderLayer> layers = new ArrayList<>();
        public final RMSNorm norm;
        private final PretrainedConfig config;

        public LlamaModel(PretrainedConfig config) {
            super("LlamaModel");
            this.config = config;
            this.embed_tokens = register_module("embed_tokens",
                    new EmbeddingImpl(config.vocabSize(), config.hiddenSize()));
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                layers.add(register_module("layers/" + i, new LlamaDecoderLayer(config, i)));
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
            for (LlamaDecoderLayer layer : layers) {
                x = layer.forward(x);
            }
            return norm.forward(x);
        }

        /** Cache-aware model forward. All layers share the same positionOffset. */
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
    public static class LlamaDecoderLayer extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final RMSNorm input_layernorm;
        public final ModelingAttention self_attn;
        public final RMSNorm post_attention_layernorm;
        public final ModelingMlp.SwiGLU mlp;

        public LlamaDecoderLayer(PretrainedConfig cfg, int layerIdx) {
            super("LlamaDecoderLayer" + layerIdx);
            this.input_layernorm = register_module("input_layernorm",
                    new RMSNorm(cfg.hiddenSize(), cfg.rmsNormEps()));
            // Llama: no qkv bias
            this.self_attn = register_module("self_attn",
                    new ModelingAttention(cfg.hiddenSize(), cfg.numAttentionHeads(),
                            cfg.numKeyValueHeads(), cfg.ropeTheta(), true, false));
            this.post_attention_layernorm = register_module("post_attention_layernorm",
                    new RMSNorm(cfg.hiddenSize(), cfg.rmsNormEps()));
            this.mlp = register_module("mlp",
                    new ModelingMlp.SwiGLU(cfg.hiddenSize(), cfg.intermediateSize()));
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
