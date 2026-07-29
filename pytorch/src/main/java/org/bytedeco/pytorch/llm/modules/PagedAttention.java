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
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.kvcache.PagedBlockManager;
import org.bytedeco.pytorch.llm.modules.attn.AttentionOps;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.cat;

/**
 * PagedAttention control-plane + correct gather math (vLLM / Kwon et al.).
 *
 * <p>Attends over KV stored in non-contiguous physical blocks addressed by a
 * per-sequence <b>block table</b>. This is a pure-Tensor didactic port: pages
 * are gathered into a contiguous K/V then run through dense/Flash SDPA. Real
 * GPU kernels fuse the gather; the numerics and block-table contract match.
 *
 * <p>Two paths:
 * <ul>
 *   <li>{@link #forwardCached} — contiguous pastK/V (same as dense)</li>
 *   <li>{@link #forwardPaged} — {@link PagedBlockManager} + block table</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PagedAttention extends Module {

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
    private final boolean isCausal;
    private final boolean useFlash;

    public PagedAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                          double ropeTheta, boolean useRope, boolean qkvBias,
                          boolean isCausal, boolean useFlash) {
        super("PagedAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.isCausal = isCausal;
        this.useFlash = useFlash;

        long qDim = (long) nHeads * hd;
        long kvDim = (long) nKvHeads * hd;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(false)));
    }

    public static PagedAttention gqa(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new PagedAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, true, false);
    }

    public static PagedAttention flashGqa(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new PagedAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, true, true);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    /** Contiguous-cache path (parity with dense / gathered pages). */
    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
        long B = x.size(0);
        long T = x.size(1);
        Tensor q = projectQ(x, B, T, positionOffset);
        Tensor[] kv = projectKV(x, B, T, positionOffset);
        Tensor k = kv[0];
        Tensor v = kv[1];
        Tensor newK = k;
        Tensor newV = v;
        long pastLen = 0L;
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            Tensor[] merged = AttentionOps.mergePast(pastK, pastV, k, v, B);
            k = merged[0];
            v = merged[1];
            pastLen = k.size(2) - T;
        }
        Tensor y = attend(q, k, v, pastLen, T, positionOffset);
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV};
    }

    /**
     * Block-table path: gather physical pages then attend.
     *
     * @param pool        physical block pool (layout [maxBlocks, layers, 2, block, H, D] or compatible)
     * @param blockTable  physical block ids for this sequence (shared across layers in multi-layer pools)
     * @param ctxLen      valid token length covered by the table
     * @param layer       layer index inside the pool
     * @return {out, newK, newV} for the <em>current</em> x tokens only (caller appends to cache)
     */
    public Tensor[] forwardPaged(Tensor x, long positionOffset,
                                 PagedBlockManager pool, int[] blockTable, int ctxLen, int layer) {
        long B = x.size(0);
        long T = x.size(1);
        if (B != 1) {
            throw new IllegalArgumentException("forwardPaged currently supports B=1 didactic path");
        }
        Tensor q = projectQ(x, B, T, positionOffset);
        Tensor[] kvNew = projectKV(x, B, T, positionOffset);

        Tensor gatheredK;
        Tensor gatheredV;
        if (blockTable == null || blockTable.length == 0 || ctxLen <= 0) {
            gatheredK = kvNew[0];
            gatheredV = kvNew[1];
        } else {
            Tensor[] pages = gatherPages(pool, blockTable, ctxLen, layer);
            // pages: [1, H, ctx, D] at nKvHeads or nHeads depending on pool; repeat if needed
            Tensor pk = pages[0];
            Tensor pv = pages[1];
            if (pk.size(1) == nKvHeads && nHeads != nKvHeads) {
                pk = AttentionOps.repeatKv(pk, nHeads / nKvHeads);
                pv = AttentionOps.repeatKv(pv, nHeads / nKvHeads);
            } else if (pk.size(1) == 1 && nHeads > 1) {
                pk = AttentionOps.repeatKv(pk, nHeads);
                pv = AttentionOps.repeatKv(pv, nHeads);
            }
            gatheredK = cat(new TensorVector(pk, kvNew[0]), 2);
            gatheredV = cat(new TensorVector(pv, kvNew[1]), 2);
        }
        long pastLen = gatheredK.size(2) - T;
        Tensor y = attend(q, gatheredK, gatheredV, pastLen, T, positionOffset);
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), kvNew[0], kvNew[1]};
    }

    /**
     * Gather K/V from pool blocks via {@link PagedBlockManager#gather}.
     * Returns {@code [1, H, ctxLen, D]} for K and V.
     */
    public static Tensor[] gatherPages(PagedBlockManager pool, int[] blockTable, int ctxLen, int layer) {
        java.util.ArrayList<Integer> table = new java.util.ArrayList<>(blockTable.length);
        for (int id : blockTable) {
            table.add(id);
        }
        Tensor[] kv = pool.gather(table, layer, ctxLen); // [length, H, D]
        Tensor k = kv[0].permute(1, 0, 2).unsqueeze(0); // [1, H, T, D]
        Tensor v = kv[1].permute(1, 0, 2).unsqueeze(0);
        return new Tensor[]{k, v};
    }

    private Tensor projectQ(Tensor x, long B, long T, long positionOffset) {
        Tensor q = q_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);
        if (useRope) {
            q = RotaryEmbedding.apply(q, ropeTheta, positionOffset);
        }
        return q;
    }

    private Tensor[] projectKV(Tensor x, long B, long T, long positionOffset) {
        Tensor k = k_proj.forward(x).view(B, T, nKvHeads, headDim).transpose(1, 2);
        Tensor v = v_proj.forward(x).view(B, T, nKvHeads, headDim).transpose(1, 2);
        if (useRope) {
            k = RotaryEmbedding.apply(k, ropeTheta, positionOffset);
        }
        int nRep = nHeads / nKvHeads;
        k = AttentionOps.repeatKv(k, nRep);
        v = AttentionOps.repeatKv(v, nRep);
        return new Tensor[]{k, v};
    }

    private Tensor attend(Tensor q, Tensor k, Tensor v, long pastLen, long T, long positionOffset) {
        double sc = AttentionOps.scale(headDim);
        if (useFlash) {
            if (pastLen == 0 && positionOffset == 0) {
                return AttentionOps.flashOnlineSdpa(q, k, v, isCausal, -1, 16, 64, sc);
            }
            return AttentionOps.flashOnlineSdpaCached(q, k, v, positionOffset, isCausal, -1, 16, 64, sc);
        }
        Tensor mask = null;
        if (isCausal) {
            mask = pastLen == 0
                    ? AttentionOps.causalMask(T, -1)
                    : AttentionOps.causalMaskCached(pastLen, T, -1);
        }
        return AttentionOps.denseSdpa(q, k, v, mask, sc, 0.0, false)[0];
    }
}
