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
import org.bytedeco.pytorch.llm.modules.attn.AttentionOps;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * Classic encoder–decoder cross-attention (Vaswani et al., Attention Is All You Need).
 *
 * <p>Q is projected from decoder states {@code x}; K/V from encoder {@code memory}.
 * Optional cache of projected memory K/V for multi-step decode.
 *
 * <pre>
 *   q = Wq x;  k,v = Wk mem, Wv mem
 *   y = softmax(q kᵀ / √d) v;  out = Wo y
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CrossAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;

    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final double dropoutP;

    public CrossAttention(long hiddenSize, long memorySize, int nHeads, int nKvHeads, int headDim,
                          boolean qkvBias, boolean oBias, double dropoutP) {
        super("CrossAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.dropoutP = Math.max(0.0, dropoutP);
        long mem = memorySize > 0 ? memorySize : hiddenSize;
        long qDim = (long) nHeads * hd;
        long kvDim = (long) nKvHeads * hd;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(mem, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(mem, kvDim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(oBias)));
    }

    public static CrossAttention mha(long hiddenSize, int nHeads) {
        return new CrossAttention(hiddenSize, hiddenSize, nHeads, nHeads,
                (int) (hiddenSize / nHeads), false, false, 0.0);
    }

    public static CrossAttention gqa(long hiddenSize, int nHeads, int nKvHeads) {
        return new CrossAttention(hiddenSize, hiddenSize, nHeads, nKvHeads,
                (int) (hiddenSize / nHeads), false, false, 0.0);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }

    @Override
    public Tensor forward(Tensor x) {
        // Without memory, treat as self-cross (memory=x) for Module.forward compatibility.
        return forwardCross(x, x, null, null)[0];
    }

    /**
     * @param x       decoder input [B, Tq, C]
     * @param memory  encoder states [B, Tm, Cm]
     * @param pastK   optional cached memory K [B,H,Tm,D]
     * @param pastV   optional cached memory V
     * @return {out [B,Tq,C], memK, memV}
     */
    public Tensor[] forwardCross(Tensor x, Tensor memory, Tensor pastK, Tensor pastV) {
        long B = x.size(0);
        long Tq = x.size(1);

        Tensor q = q_proj.forward(x).view(B, Tq, nHeads, headDim).transpose(1, 2);

        Tensor k;
        Tensor v;
        if (pastK != null && pastK.defined() && pastV != null && pastV.defined()) {
            k = pastK;
            v = pastV;
        } else {
            long Tm = memory.size(1);
            k = k_proj.forward(memory).view(B, Tm, nKvHeads, headDim).transpose(1, 2);
            v = v_proj.forward(memory).view(B, Tm, nKvHeads, headDim).transpose(1, 2);
            int nRep = nHeads / nKvHeads;
            k = AttentionOps.repeatKv(k, nRep);
            v = AttentionOps.repeatKv(v, nRep);
        }

        double sc = AttentionOps.scale(headDim);
        Tensor[] sdpa = AttentionOps.denseSdpa(q, k, v, null, sc, dropoutP, is_training());
        Tensor y = sdpa[0].transpose(1, 2).contiguous().view(B, Tq, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), k, v};
    }
}
