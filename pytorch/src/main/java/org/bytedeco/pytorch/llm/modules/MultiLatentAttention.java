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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * Multi-head Latent Attention (MLA) — DeepSeek-V2 / V3 style compressed KV.
 *
 * <p>Instead of caching full K/V per head, MLA projects input down to a low-rank
 * latent {@code kv_lora_rank}, then up-projects to K/V. Q is optionally also
 * low-rank ({@code q_lora_rank}). RoPE is applied on a decoupled rope dim
 * ({@code qk_rope_head_dim}) while the no-rope content dim is
 * {@code qk_nope_head_dim}.
 *
 * <p>This is a didactic / engineering reference matching the high-level DeepSeek
 * equations, suitable for composing small models and benchmarks. It exposes
 * cache-friendly compressed KV via {@link #forwardCached}.
 *
 * <pre>
 *   c_kv = kv_a_proj(x)            # [B,T, kv_lora_rank]
 *   k_nope, v = split(kv_b_proj(c_kv))
 *   q = q_b_proj(q_a_proj(x)) or q_proj(x)
 *   apply RoPE on rope slices of q/k; attend; o_proj
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MultiLatentAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_a_proj;   // optional low-rank Q down
    public final RMSNorm q_a_layernorm;
    public final LinearImpl q_b_proj;   // Q up (or full q_proj when no q-lora)
    public final LinearImpl kv_a_proj_with_mqa; // down to kv_lora + rope dim
    public final RMSNorm kv_a_layernorm;
    public final LinearImpl kv_b_proj;  // up to k_nope + v
    public final LinearImpl o_proj;

    private final int nHeads;
    private final int qkNopeHeadDim;
    private final int qkRopeHeadDim;
    private final int vHeadDim;
    private final int kvLoraRank;
    private final int qLoraRank; // 0 = full Q
    private final double ropeTheta;
    private final long hiddenSize;

    public MultiLatentAttention(long hiddenSize, int nHeads,
                                int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
                                int kvLoraRank, int qLoraRank, double ropeTheta) {
        super("MultiLatentAttention");
        if (nHeads <= 0 || kvLoraRank <= 0) {
            throw new IllegalArgumentException("nHeads/kvLoraRank must be > 0");
        }
        this.hiddenSize = hiddenSize;
        this.nHeads = nHeads;
        this.qkNopeHeadDim = qkNopeHeadDim;
        this.qkRopeHeadDim = qkRopeHeadDim;
        this.vHeadDim = vHeadDim;
        this.kvLoraRank = kvLoraRank;
        this.qLoraRank = Math.max(0, qLoraRank);
        this.ropeTheta = ropeTheta;

        int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        long qOut = (long) nHeads * qHeadDim;
        long kvAOut = (long) kvLoraRank + qkRopeHeadDim; // compressed + k_rope
        long kvBOut = (long) nHeads * (qkNopeHeadDim + vHeadDim);

        if (this.qLoraRank > 0) {
            this.q_a_proj = register_module("q_a_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, this.qLoraRank).bias(false)));
            this.q_a_layernorm = register_module("q_a_layernorm", new RMSNorm(this.qLoraRank));
            this.q_b_proj = register_module("q_b_proj",
                    new LinearImpl(new LinearOptions(this.qLoraRank, qOut).bias(false)));
        } else {
            this.q_a_proj = null;
            this.q_a_layernorm = null;
            this.q_b_proj = register_module("q_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, qOut).bias(false)));
        }

        this.kv_a_proj_with_mqa = register_module("kv_a_proj_with_mqa",
                new LinearImpl(new LinearOptions(hiddenSize, kvAOut).bias(false)));
        this.kv_a_layernorm = register_module("kv_a_layernorm", new RMSNorm(kvLoraRank));
        this.kv_b_proj = register_module("kv_b_proj",
                new LinearImpl(new LinearOptions(kvLoraRank, kvBOut).bias(false)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions((long) nHeads * vHeadDim, hiddenSize).bias(false)));
    }

    /** Compact DeepSeek-V2-ish defaults scaled for tiny models. */
    public static MultiLatentAttention deepseek(long hiddenSize, int nHeads,
                                                int kvLoraRank, double ropeTheta) {
        int rope = 64;
        int nope = Math.max(32, (int) (hiddenSize / nHeads) - rope);
        int v = Math.max(32, (int) (hiddenSize / nHeads));
        int qLora = (int) Math.min(hiddenSize, Math.max(kvLoraRank, hiddenSize / 4));
        return new MultiLatentAttention(hiddenSize, nHeads, nope, rope, v,
                kvLoraRank, qLora, ropeTheta);
    }

    public int nHeads() { return nHeads; }
    public int kvLoraRank() { return kvLoraRank; }
    public int qkRopeHeadDim() { return qkRopeHeadDim; }
    public int vHeadDim() { return vHeadDim; }
    public long hiddenSize() { return hiddenSize; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    /**
     * @param pastCkv compressed KV latent cache [B, past, kv_lora_rank] or null
     * @param pastKr  rope-K cache [B, past, qk_rope_head_dim] or null
     * @return {out [B,T,H], newCkv [B,T,kv_lora], newKr [B,T,rope_dim]}
     */
    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastCkv, Tensor pastKr) {
        long B = x.size(0);
        long T = x.size(1);
        int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

        // Q
        Tensor q;
        if (q_a_proj != null) {
            q = q_b_proj.forward(q_a_layernorm.forward(q_a_proj.forward(x)));
        } else {
            q = q_b_proj.forward(x);
        }
        q = q.view(B, T, nHeads, qHeadDim).transpose(1, 2); // [B,H,T,Dq]
        Tensor qNope = q.slice(3, new LongOptional(0), new LongOptional(qkNopeHeadDim), 1);
        Tensor qRope = q.slice(3, new LongOptional(qkNopeHeadDim), new LongOptional(qHeadDim), 1);

        // compressed KV + k_rope
        Tensor kvA = kv_a_proj_with_mqa.forward(x); // [B,T, kv_lora + rope]
        Tensor ckv = kvA.slice(2, new LongOptional(0), new LongOptional(kvLoraRank), 1);
        Tensor kRope = kvA.slice(2, new LongOptional(kvLoraRank),
                new LongOptional(kvLoraRank + qkRopeHeadDim), 1); // [B,T,rope]
        ckv = kv_a_layernorm.forward(ckv);

        Tensor newCkv = ckv;
        Tensor newKr = kRope;

        long pastLen = 0L;
        if (pastCkv != null && pastCkv.defined() && pastCkv.dim() == 3) {
            pastLen = pastCkv.size(1);
            ckv = cat(new TensorVector(pastCkv, ckv), 1);
            kRope = cat(new TensorVector(pastKr, kRope), 1);
        }
        long total = pastLen + T;

        // up-project compressed latent → k_nope + v for all positions in cache window
        Tensor kvB = kv_b_proj.forward(ckv); // [B, total, H*(nope+v)]
        kvB = kvB.view(B, total, nHeads, qkNopeHeadDim + vHeadDim).transpose(1, 2);
        Tensor kNope = kvB.slice(3, new LongOptional(0), new LongOptional(qkNopeHeadDim), 1);
        Tensor v = kvB.slice(3, new LongOptional(qkNopeHeadDim),
                new LongOptional(qkNopeHeadDim + vHeadDim), 1); // [B,H,total,Dv]

        // RoPE on q_rope / k_rope. k_rope is shared across heads (MQA-style).
        qRope = RotaryEmbedding.apply(
                qRope, ropeTheta, positionOffset, 1.0); // [B,H,T,rope]
        // k_rope: [B,T,rope] → [B,1,total,rope] then expand heads
        Tensor kRope4 = kRope.unsqueeze(1); // [B,1,total,rope]
        // apply rope only on the new chunk for past-aware: rebuild full with offset 0..total-1
        kRope4 = RotaryEmbedding.apply(kRope4.expand(new long[]{B, 1, total, qkRopeHeadDim}).contiguous(),
                ropeTheta, 0L, 1.0);
        kRope4 = kRope4.expand(new long[]{B, nHeads, total, qkRopeHeadDim});

        Tensor qFull = cat(new TensorVector(qNope, qRope), 3); // [B,H,T,Dq]
        Tensor kFull = cat(new TensorVector(kNope, kRope4), 3); // [B,H,total,Dq]

        double scale = 1.0 / Math.sqrt(qHeadDim);
        Tensor att = matmul(qFull, kFull.transpose(-2, -1)).mul(new Scalar(scale));
        if (pastLen == 0) {
            att = att.add(Attention.causalMask(T, -1));
        } else {
            att = att.add(Attention.causalMaskCached(pastLen, T, -1));
        }
        att = softmax(att, -1L);
        Tensor y = matmul(att, v); // [B,H,T,Dv]
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * vHeadDim);
        return new Tensor[]{o_proj.forward(y), newCkv, newKr};
    }
}
