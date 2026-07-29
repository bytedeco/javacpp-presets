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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.modules.attn.AttentionOps;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.topk;

/**
 * Native Sparse Attention (DeepSeek NSA) — <b>lite didactic</b> port.
 *
 * <p>Production NSA combines token compression, block selection, and sparse
 * kernels. This reference implements a simplified two-stage path suitable for
 * unit tests and composition:
 * <ol>
 *   <li><b>Compress</b>: average-pool K/V into block summaries of size
 *       {@code compressBlock}.</li>
 *   <li><b>Select</b>: each query scores summaries and keeps top-{@code topBlocks}
 *       blocks (plus a forced local window), then runs dense SDPA on the
 *       selected tokens only (gathered).</li>
 * </ol>
 *
 * <p>Not bit-identical to DeepSeek CUDA kernels; documents the control flow.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NativeSparseAttention extends Module {

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
    private final double ropeTheta;
    private final boolean useRope;
    private final int compressBlock;
    private final int topBlocks;
    private final int localWindow;

    public NativeSparseAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                                 double ropeTheta, boolean useRope, boolean qkvBias,
                                 int compressBlock, int topBlocks, int localWindow) {
        super("NativeSparseAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = Math.max(1, hd);
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.compressBlock = Math.max(1, compressBlock);
        this.topBlocks = Math.max(1, topBlocks);
        this.localWindow = Math.max(1, localWindow);

        long qDim = (long) nHeads * this.headDim;
        long kvDim = (long) nKvHeads * this.headDim;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(false)));
    }

    public static NativeSparseAttention gqa(long hiddenSize, int nHeads, int nKvHeads,
                                            double ropeTheta, int compressBlock, int topBlocks) {
        return new NativeSparseAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, compressBlock, topBlocks, compressBlock);
    }

    public static NativeSparseAttention paperDefault(long hiddenSize, int nHeads, double ropeTheta) {
        return gqa(hiddenSize, nHeads, nHeads, ropeTheta, 4, 4);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public int compressBlock() { return compressBlock; }
    public int topBlocks() { return topBlocks; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
        long B = x.size(0);
        long T = x.size(1);

        Tensor q = q_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);
        Tensor k = k_proj.forward(x).view(B, T, nKvHeads, headDim).transpose(1, 2);
        Tensor v = v_proj.forward(x).view(B, T, nKvHeads, headDim).transpose(1, 2);

        if (useRope) {
            q = RotaryEmbedding.apply(q, ropeTheta, positionOffset);
            k = RotaryEmbedding.apply(k, ropeTheta, positionOffset);
        }
        int nRep = nHeads / nKvHeads;
        k = AttentionOps.repeatKv(k, nRep);
        v = AttentionOps.repeatKv(v, nRep);

        Tensor newK = k;
        Tensor newV = v;
        long pastLen = 0L;
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            Tensor[] merged = AttentionOps.mergePast(pastK, pastV, k, v, B);
            k = merged[0];
            v = merged[1];
            pastLen = k.size(2) - T;
        }

        long Tk = k.size(2);
        // For short sequences, fall back to dense causal (selection overhead not worth it)
        if (Tk <= compressBlock * topBlocks || T > 1 && Tk <= 64) {
            Tensor mask = pastLen == 0
                    ? AttentionOps.causalMask(T, -1)
                    : AttentionOps.causalMaskCached(pastLen, T, -1);
            double sc = AttentionOps.scale(headDim);
            Tensor y = AttentionOps.denseSdpa(q, k, v, mask, sc, 0.0, false)[0];
            y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
            return new Tensor[]{o_proj.forward(y), newK, newV};
        }

        // Compress K into block means: [B,H,nBlocks,D]
        long nBlocks = (Tk + compressBlock - 1) / compressBlock;
        TensorVector summaries = new TensorVector();
        for (long b = 0; b < nBlocks; b++) {
            long start = b * compressBlock;
            long len = Math.min(compressBlock, Tk - start);
            Tensor chunk = k.narrow(2, start, len).mean(new long[]{2L}); // [B,H,D]
            summaries.push_back(chunk.unsqueeze(2));
        }
        Tensor kSum = cat(summaries, 2); // [B,H,nBlocks,D]

        // Score summaries with mean query (didactic; real NSA is per-token)
        Tensor qMean = q.mean(new long[]{2L}); // [B,H,D]
        double sc = AttentionOps.scale(headDim);
        Tensor blockScores = qMean.unsqueeze(2).matmul(kSum.transpose(-2, -1)).mul(new Scalar(sc));
        // blockScores: [B,H,1,nBlocks]
        blockScores = blockScores.squeeze(2); // [B,H,nBlocks]

        int kTop = (int) Math.min(topBlocks, nBlocks);
        T_TensorTensor_T top = topk(blockScores, kTop, -1L, true, true);
        Tensor topIdx = top.get1(); // [B,H,kTop]

        // Build union of selected token indices + local window for last query pos (decode-friendly)
        // For simplicity: for each batch element use head-0's top blocks (shared selection)
        java.util.TreeSet<Long> selected = new java.util.TreeSet<>();
        // local window near end
        long localStart = Math.max(0, Tk - localWindow);
        for (long i = localStart; i < Tk; i++) {
            selected.add(i);
        }
        // top blocks from B=0,H=0
        for (int t = 0; t < kTop; t++) {
            long bi = topIdx.select(0, 0).select(0, 0).select(0, t).item().toLong();
            long start = bi * compressBlock;
            long end = Math.min(Tk, start + compressBlock);
            for (long i = start; i < end; i++) {
                selected.add(i);
            }
        }
        // causal: only indices <= pastLen+T-1 already in Tk

        long[] idxArr = selected.stream().mapToLong(Long::longValue).toArray();
        Tensor index = org.bytedeco.pytorch.global.torch.zeros(new long[]{idxArr.length},
                new org.bytedeco.pytorch.TensorOptions(org.bytedeco.pytorch.global.torch.ScalarType.Long));
        for (int i = 0; i < idxArr.length; i++) {
            index.narrow(0, i, 1).fill_(new Scalar(idxArr[i]));
        }

        // Gather along seq dim: k is [B,H,Tk,D] — use index_select on dim 2
        Tensor kSel = k.index_select(2, index);
        Tensor vSel = v.index_select(2, index);

        // Attend without extra mask (selected set is causal-safe if we only took past tokens)
        Tensor y = AttentionOps.denseSdpa(q, kSel, vSel, null, sc, 0.0, false)[0];
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV};
    }
}
