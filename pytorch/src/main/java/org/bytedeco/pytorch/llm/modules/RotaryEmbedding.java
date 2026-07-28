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

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.arange;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.exp;

/**
 * Rotary Position Embedding helpers (Llama / Qwen / DeepSeek / Mistral style).
 *
 * <p>Input layout for apply: {@code [B, n_heads, T, head_dim]} with even head_dim.
 *
 * <p>Also provides GQA KV-head repeat and optional NTK / YaRN-style scale factor
 * via {@code ropeScaling} multiplier on inverse frequencies.
 */
public final class RotaryEmbedding {

    private RotaryEmbedding() {}

    /**
     * Apply RoPE to q or k.
     *
     * @param x          [B, H, T, D], D even
     * @param theta      base frequency (10000 classic, 1e6 Qwen2, 500000 Llama3)
     * @param posOffset  starting position (for incremental decode)
     * @param ropeScaling multiplies inv-freq (1.0 = none; &gt;1 stretches positions)
     */
    public static Tensor apply(Tensor x, double theta, long posOffset, double ropeScaling) {
        long T = x.size(2);
        long D = x.size(3);
        if (D % 2 != 0) {
            return x;
        }
        long half = D / 2;
        double scale = ropeScaling <= 0 ? 1.0 : ropeScaling;

        Tensor pos = arange(new Scalar(posOffset), new Scalar(posOffset + T), new Scalar(1L),
                new TensorOptions(ScalarType.Float));
        Tensor idx = arange(new Scalar(0L), new Scalar(half), new Scalar(1L),
                new TensorOptions(ScalarType.Float));
        // inv_freq = (theta ^ -(2i/D)) / scale
        Tensor freq = idx.mul(new Scalar(2.0)).div(new Scalar((double) D));
        Tensor invFreq = exp(freq.neg().mul(new Scalar(Math.log(theta)))).div(new Scalar(scale));
        Tensor angles = pos.unsqueeze(1).mul(invFreq.unsqueeze(0)); // [T, half]
        Tensor cos = angles.cos().unsqueeze(0).unsqueeze(0); // [1,1,T,half]
        Tensor sin = angles.sin().unsqueeze(0).unsqueeze(0);

        Tensor x1 = x.slice(3, new LongOptional(0), new LongOptional(half), 1);
        Tensor x2 = x.slice(3, new LongOptional(half), new LongOptional(D), 1);
        Tensor r1 = x1.mul(cos).sub(x2.mul(sin));
        Tensor r2 = x1.mul(sin).add(x2.mul(cos));
        return cat(new TensorVector(r1, r2), 3);
    }

    public static Tensor apply(Tensor x, double theta, long posOffset) {
        return apply(x, theta, posOffset, 1.0);
    }

    public static Tensor apply(Tensor x, double theta) {
        return apply(x, theta, 0L, 1.0);
    }

    /**
     * Repeat KV heads for GQA: [B, kv_heads, T, D] → [B, q_heads, T, D]
     * when {@code n_q % n_kv == 0}.
     */
    public static Tensor repeatKv(Tensor x, int nRep) {
        if (nRep <= 1) {
            return x;
        }
        long B = x.size(0);
        long kv = x.size(1);
        long T = x.size(2);
        long D = x.size(3);
        Tensor t = x.unsqueeze(2).expand(new long[]{B, kv, nRep, T, D}).contiguous();
        return t.reshape(B, kv * nRep, T, D);
    }

    /**
     * Interleaved half-dim style used by some GPT-NeoX / Falcon exports
     * (rotate pairs (x0,x1), (x2,x3), … instead of first-half / second-half).
     * For even D this is numerically close for many configs; kept for completeness.
     */
    public static Tensor applyInterleaved(Tensor x, double theta, long posOffset) {
        // Convert interleaved view → half-split style by rearrange, apply, rearrange back.
        // x: [B,H,T,D] with pairs along D.
        long B = x.size(0);
        long H = x.size(1);
        long T = x.size(2);
        long D = x.size(3);
        if (D % 2 != 0) {
            return x;
        }
        // reshape to [B,H,T,D/2,2], transpose last two → half-split layout, apply, invert
        Tensor paired = x.reshape(B, H, T, D / 2, 2).permute(0, 1, 2, 4, 3).contiguous()
                .reshape(B, H, T, D);
        Tensor rotated = apply(paired, theta, posOffset, 1.0);
        return rotated.reshape(B, H, T, 2, D / 2).permute(0, 1, 2, 4, 3).contiguous()
                .reshape(B, H, T, D);
    }
}
