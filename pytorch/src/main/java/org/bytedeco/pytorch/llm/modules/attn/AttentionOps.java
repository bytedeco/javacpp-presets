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
package org.bytedeco.pytorch.llm.modules.attn;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.modules.RotaryEmbedding;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.arange;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.full;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.maximum;
import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.tril;
import static org.bytedeco.pytorch.global.torch.triu;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Shared attention primitives used by paper-level variants in
 * {@code org.bytedeco.pytorch.llm.modules}.
 *
 * <p>Masks and scale match {@link org.bytedeco.pytorch.llm.modules.Attention}
 * so Flash / sparse / paged paths can be parity-checked against dense MHA.
 */
public final class AttentionOps {

    private AttentionOps() {}

    public static double scale(int headDim) {
        return 1.0 / Math.sqrt(Math.max(1, headDim));
    }

    /** Causal (+ optional sliding-window) additive mask {@code [T,T]} with {@code -1e9} blocked. */
    public static Tensor causalMask(long T, int window) {
        Tensor onesT = ones(new long[]{T, T});
        Tensor tri = triu(onesT, 1L);
        if (window > 0 && window < T) {
            Tensor lowerFar = tril(ones(new long[]{T, T}), -(long) window);
            tri = tri.add(lowerFar);
        }
        return tri.mul(new Scalar(-1e9f));
    }

    /** Cached decode mask rows {@code [T, past+T]}. */
    public static Tensor causalMaskCached(long pastLen, long T, int window) {
        long total = pastLen + T;
        Tensor full = causalMask(total, window);
        return full.narrow(0, pastLen, T);
    }

    /**
     * ALiBi slopes bias {@code [1, H, T_q, T_k]}.
     * {@code slope_h = 2^(-8/H)^(h+1)}; values are {@code -slope * |i-j|}.
     */
    public static Tensor alibiBias(int nHeads, long totalLen, long pastLen, long T) {
        double base = Math.pow(2.0, -8.0 / Math.max(1, nHeads));
        Tensor qPos = arange(new Scalar(pastLen), new Scalar(pastLen + T), new Scalar(1L),
                new TensorOptions(ScalarType.Float));
        Tensor kPos = arange(new Scalar(0L), new Scalar(totalLen), new Scalar(1L),
                new TensorOptions(ScalarType.Float));
        Tensor absDist = kPos.unsqueeze(0).sub(qPos.unsqueeze(1)).abs();
        Tensor out = zeros(new long[]{1, nHeads, T, totalLen}, new TensorOptions(ScalarType.Float));
        for (int h = 0; h < nHeads; h++) {
            double slope = Math.pow(base, h + 1);
            out.select(1, h).copy_(absDist.mul(new Scalar(-slope)));
        }
        return out;
    }

    /**
     * Longformer-style band + global additive mask {@code [T,T]}.
     * Global tokens (first {@code nGlobal}) attend fully and are fully attended;
     * others use a sliding window of {@code window} (each side inclusive of self).
     */
    public static Tensor longformerMask(long T, int window, int nGlobal) {
        Tensor mask = full(new long[]{T, T}, new Scalar(-1e9f));
        int half = Math.max(0, window);
        for (long i = 0; i < T; i++) {
            long lo = Math.max(0, i - half);
            long hi = Math.min(T, i + half + 1);
            mask.select(0, i).narrow(0, lo, hi - lo).fill_(new Scalar(0.0f));
            // causal: block future
            if (i + 1 < T) {
                mask.select(0, i).narrow(0, i + 1, T - i - 1).fill_(new Scalar(-1e9f));
            }
        }
        int g = Math.min(nGlobal, (int) T);
        for (int gi = 0; gi < g; gi++) {
            // global row/col fully visible under causal
            for (long j = 0; j <= gi; j++) {
                mask.select(0, gi).narrow(0, j, 1).fill_(new Scalar(0.0f));
            }
            for (long i = gi; i < T; i++) {
                mask.select(0, i).narrow(0, gi, 1).fill_(new Scalar(0.0f));
            }
        }
        return mask;
    }

    /** StreamingLLM sink + recent window causal mask {@code [Tq, Tk]} for full sequence. */
    public static Tensor sinkWindowMask(long totalLen, long pastLen, long T, int sink, int window) {
        long Tk = totalLen;
        Tensor mask = full(new long[]{T, Tk}, new Scalar(-1e9f));
        for (long qi = 0; qi < T; qi++) {
            long absQ = pastLen + qi;
            // sink tokens
            long sinkEnd = Math.min(sink, absQ + 1);
            if (sinkEnd > 0) {
                mask.select(0, qi).narrow(0, 0, sinkEnd).fill_(new Scalar(0.0f));
            }
            // recent window ending at absQ
            long lo = Math.max(sinkEnd, absQ - Math.max(0, window) + 1);
            long hi = absQ + 1;
            if (hi > lo) {
                mask.select(0, qi).narrow(0, lo, hi - lo).fill_(new Scalar(0.0f));
            }
        }
        return mask;
    }

    /**
     * Dense scaled-dot-product attention on already-projected Q/K/V
     * layouts {@code [B,H,T,D]} (K/V already GQA-repeated to H).
     *
     * @return {@code [out, attnWeights]} where attn is {@code [B,H,Tq,Tk]} after softmax
     */
    public static Tensor[] denseSdpa(Tensor q, Tensor k, Tensor v, Tensor additiveMask, double scale,
                                     double dropoutP, boolean training) {
        Tensor att = matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale));
        if (additiveMask != null && additiveMask.defined()) {
            att = att.add(additiveMask);
        }
        att = softmax(att, -1L);
        if (dropoutP > 0 && training) {
            att = org.bytedeco.pytorch.global.torch.dropout(att, dropoutP, true);
        }
        Tensor y = matmul(att, v);
        return new Tensor[]{y, att};
    }

    /**
     * FlashAttention-style <b>online softmax</b> over K/V blocks (Dao et al.).
     * Pure Tensor reference — same math as dense SDPA, blockwise to avoid full score matrix.
     *
     * <p>Q/K/V: {@code [B,H,T,D]}. Optional additive mask broadcastable to scores.
     * When {@code causal} and mask is null, applies standard causal blocking inside tiles.
     */
    public static Tensor flashOnlineSdpa(Tensor q, Tensor k, Tensor v, boolean causal, int window,
                                         int blockQ, int blockK, double scale) {
        long B = q.size(0);
        long H = q.size(1);
        long Tq = q.size(2);
        long Tk = k.size(2);
        long D = q.size(3);
        int Br = Math.max(1, blockQ);
        int Bc = Math.max(1, blockK);

        // Accumulators in fp32-ish path via Tensor ops
        Tensor o = zeros(new long[]{B, H, Tq, D}, q.options());
        Tensor rowMax = full(new long[]{B, H, Tq, 1}, new Scalar(-1e9), q.options());
        Tensor rowSum = zeros(new long[]{B, H, Tq, 1}, q.options());

        for (long startK = 0; startK < Tk; startK += Bc) {
            long endK = Math.min(Tk, startK + Bc);
            long lenK = endK - startK;
            Tensor kBlock = k.narrow(2, startK, lenK); // [B,H,Bc,D]
            Tensor vBlock = v.narrow(2, startK, lenK);

            for (long startQ = 0; startQ < Tq; startQ += Br) {
                long endQ = Math.min(Tq, startQ + Br);
                long lenQ = endQ - startQ;
                Tensor qBlock = q.narrow(2, startQ, lenQ); // [B,H,Br,D]

                Tensor s = matmul(qBlock, kBlock.transpose(-2, -1)).mul(new Scalar(scale)); // [B,H,Br,Bc]

                if (causal || window > 0) {
                    // Build local mask for absolute positions
                    Tensor localMask = zeros(new long[]{lenQ, lenK}, q.options());
                    for (long qi = 0; qi < lenQ; qi++) {
                        long absQ = startQ + qi; // if past concatenated, caller must offset q positions externally
                        for (long kj = 0; kj < lenK; kj++) {
                            long absK = startK + kj;
                            boolean block = false;
                            if (causal && absK > absQ) {
                                block = true;
                            }
                            if (window > 0 && absK < absQ - window + 1) {
                                block = true;
                            }
                            if (block) {
                                localMask.select(0, qi).narrow(0, kj, 1).fill_(new Scalar(-1e9f));
                            }
                        }
                    }
                    s = s.add(localMask);
                }

                Tensor mBlock = s.amax(new long[]{-1L}, true); // [B,H,Br,1]
                Tensor mOld = rowMax.narrow(2, startQ, lenQ);
                Tensor mNew = maximum(mOld, mBlock);

                Tensor p = s.sub(mNew).exp(); // [B,H,Br,Bc]
                Tensor lBlock = p.sum(new long[]{-1L}, true, new org.bytedeco.pytorch.ScalarTypeOptional()); // [B,H,Br,1]

                Tensor alpha = mOld.sub(mNew).exp();
                Tensor lOld = rowSum.narrow(2, startQ, lenQ);
                Tensor lNew = lOld.mul(alpha).add(lBlock);

                Tensor oBlock = o.narrow(2, startQ, lenQ);
                Tensor oCorr = oBlock.mul(alpha).add(matmul(p, vBlock));
                oBlock.copy_(oCorr);
                rowMax.narrow(2, startQ, lenQ).copy_(mNew);
                rowSum.narrow(2, startQ, lenQ).copy_(lNew);
            }
        }
        return o.div(rowSum.clamp_min(new Scalar(1e-9)));
    }

    /**
     * Same as {@link #flashOnlineSdpa} but with absolute query position offset
     * (for decode where q positions start at {@code qPosOffset}).
     */
    public static Tensor flashOnlineSdpaCached(Tensor q, Tensor k, Tensor v, long qPosOffset,
                                               boolean causal, int window, int blockQ, int blockK,
                                               double scale) {
        long B = q.size(0);
        long H = q.size(1);
        long Tq = q.size(2);
        long Tk = k.size(2);
        long D = q.size(3);
        int Br = Math.max(1, blockQ);
        int Bc = Math.max(1, blockK);

        Tensor o = zeros(new long[]{B, H, Tq, D}, q.options());
        Tensor rowMax = full(new long[]{B, H, Tq, 1}, new Scalar(-1e9), q.options());
        Tensor rowSum = zeros(new long[]{B, H, Tq, 1}, q.options());

        for (long startK = 0; startK < Tk; startK += Bc) {
            long endK = Math.min(Tk, startK + Bc);
            long lenK = endK - startK;
            Tensor kBlock = k.narrow(2, startK, lenK);
            Tensor vBlock = v.narrow(2, startK, lenK);

            for (long startQ = 0; startQ < Tq; startQ += Br) {
                long endQ = Math.min(Tq, startQ + Br);
                long lenQ = endQ - startQ;
                Tensor qBlock = q.narrow(2, startQ, lenQ);
                Tensor s = matmul(qBlock, kBlock.transpose(-2, -1)).mul(new Scalar(scale));

                if (causal || window > 0) {
                    Tensor localMask = zeros(new long[]{lenQ, lenK}, q.options());
                    for (long qi = 0; qi < lenQ; qi++) {
                        long absQ = qPosOffset + startQ + qi;
                        for (long kj = 0; kj < lenK; kj++) {
                            long absK = startK + kj;
                            boolean block = false;
                            if (causal && absK > absQ) {
                                block = true;
                            }
                            if (window > 0 && absK < absQ - window + 1) {
                                block = true;
                            }
                            if (block) {
                                localMask.select(0, qi).narrow(0, kj, 1).fill_(new Scalar(-1e9f));
                            }
                        }
                    }
                    s = s.add(localMask);
                }

                Tensor mBlock = s.amax(new long[]{-1L}, true);
                Tensor mOld = rowMax.narrow(2, startQ, lenQ);
                Tensor mNew = maximum(mOld, mBlock);
                Tensor p = s.sub(mNew).exp();
                Tensor lBlock = p.sum(new long[]{-1L}, true, new org.bytedeco.pytorch.ScalarTypeOptional());
                Tensor alpha = mOld.sub(mNew).exp();
                Tensor lOld = rowSum.narrow(2, startQ, lenQ);
                Tensor lNew = lOld.mul(alpha).add(lBlock);
                Tensor oBlock = o.narrow(2, startQ, lenQ);
                oBlock.copy_(oBlock.mul(alpha).add(matmul(p, vBlock)));
                rowMax.narrow(2, startQ, lenQ).copy_(mNew);
                rowSum.narrow(2, startQ, lenQ).copy_(lNew);
            }
        }
        return o.div(rowSum.clamp_min(new Scalar(1e-9)));
    }

    public static Tensor repeatKv(Tensor x, int nRep) {
        return RotaryEmbedding.repeatKv(x, nRep);
    }

    /** Concat past and present along sequence dim (2 for [B,H,T,D]). */
    public static Tensor[] concatPast(Tensor pastK, Tensor pastV, Tensor k, Tensor v, long B) {
        long pastLen = 0L;
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            if (pastK.dim() == 3) {
                pastK = pastK.permute(1, 0, 2).unsqueeze(0);
                pastV = pastV.permute(1, 0, 2).unsqueeze(0);
                if (B > 1) {
                    pastK = pastK.expand(new long[]{B, pastK.size(1), pastK.size(2), pastK.size(3)});
                    pastV = pastV.expand(new long[]{B, pastV.size(1), pastV.size(2), pastV.size(3)});
                }
            }
            pastLen = pastK.size(2);
        }
        Tensor fullK = k;
        Tensor fullV = v;
        if (pastLen > 0) {
            fullK = cat(new TensorVector(pastK, k), 2);
            fullV = cat(new TensorVector(pastV, v), 2);
        }
        return new Tensor[]{fullK, fullV, fullK.narrow(2, 0, 1).mul(new Scalar(0)).add(new Scalar((double) pastLen))};
    }

    public static long pastLength(Tensor pastK, Tensor pastV, long B) {
        if (pastK == null || !pastK.defined() || pastK.dim() < 3) {
            return 0L;
        }
        if (pastK.dim() == 3) {
            return pastK.size(0); // [T,H,D] style rare
        }
        return pastK.size(2);
    }

    public static Tensor[] mergePast(Tensor pastK, Tensor pastV, Tensor k, Tensor v, long B) {
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            if (pastK.dim() == 3) {
                pastK = pastK.permute(1, 0, 2).unsqueeze(0);
                pastV = pastV.permute(1, 0, 2).unsqueeze(0);
                if (B > 1) {
                    pastK = pastK.expand(new long[]{B, pastK.size(1), pastK.size(2), pastK.size(3)});
                    pastV = pastV.expand(new long[]{B, pastV.size(1), pastV.size(2), pastV.size(3)});
                }
            }
            k = cat(new TensorVector(pastK, k), 2);
            v = cat(new TensorVector(pastV, v), 2);
        }
        return new Tensor[]{k, v};
    }
}
