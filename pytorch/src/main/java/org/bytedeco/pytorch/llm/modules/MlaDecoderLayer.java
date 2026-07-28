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

/**
 * DeepSeek-style decoder block: Pre-Norm + {@link MultiLatentAttention} + SwiGLU/MoE.
 *
 * <pre>
 *   x = x + mla(rms(x))
 *   x = x + ffn(rms(x))
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MlaDecoderLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final RMSNorm input_layernorm;
    public final MultiLatentAttention self_attn;
    public final RMSNorm post_attention_layernorm;
    public final Module mlp;
    private final int layerIdx;
    private final long hiddenSize;

    public MlaDecoderLayer(long hiddenSize, int nHeads, int kvLoraRank,
                           long intermediateSize, double ropeTheta,
                           boolean useMoe, int moeExperts, int moeTopK, int layerIdx) {
        super("MlaDecoderLayer" + layerIdx);
        this.hiddenSize = hiddenSize;
        this.layerIdx = layerIdx;
        this.input_layernorm = register_module("input_layernorm", new RMSNorm(hiddenSize));
        this.self_attn = register_module("self_attn",
                MultiLatentAttention.deepseek(hiddenSize, nHeads, kvLoraRank, ropeTheta));
        this.post_attention_layernorm = register_module("post_attention_layernorm",
                new RMSNorm(hiddenSize));
        if (useMoe) {
            this.mlp = register_module("mlp",
                    MoE.deepseek(hiddenSize, intermediateSize, moeExperts, moeTopK));
        } else {
            this.mlp = register_module("mlp", new Mlp.SwiGLU(hiddenSize, intermediateSize));
        }
    }

    public static MlaDecoderLayer dense(long hiddenSize, int nHeads, int kvLoraRank,
                                        long intermediateSize, double ropeTheta, int layerIdx) {
        return new MlaDecoderLayer(hiddenSize, nHeads, kvLoraRank, intermediateSize,
                ropeTheta, false, 0, 0, layerIdx);
    }

    public static MlaDecoderLayer moe(long hiddenSize, int nHeads, int kvLoraRank,
                                      long intermediateSize, double ropeTheta,
                                      int experts, int topK, int layerIdx) {
        return new MlaDecoderLayer(hiddenSize, nHeads, kvLoraRank, intermediateSize,
                ropeTheta, true, experts, topK, layerIdx);
    }

    public int layerIdx() { return layerIdx; }
    public long hiddenSize() { return hiddenSize; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    /** {out, newCkv, newKr} */
    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastCkv, Tensor pastKr) {
        Tensor[] att = self_attn.forwardCached(input_layernorm.forward(x), positionOffset, pastCkv, pastKr);
        Tensor out = x.add(att[0]);
        out = out.add(mlp.forward(post_attention_layernorm.forward(out)));
        return new Tensor[]{out, att[1], att[2]};
    }
}
