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
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.llm.modules.attn.AttentionOps;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.full;

/**
 * Sparse attention — Longformer (Beltagy et al.) window+global and optional
 * BigBird-style random extra links (didactic dense-mask realization).
 *
 * <p>Complexity remains O(T²) in this reference (mask is materialised); the
 * pattern of allowed positions matches the papers so unit tests can verify
 * far positions receive ~0 mass.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SparseAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public enum Pattern { LONGFORMER, BIGBIRD }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;

    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final double ropeTheta;
    private final boolean useRope;
    private final int window;
    private final int nGlobal;
    private final int nRandom;
    private final Pattern pattern;
    private final long randomSeed;

    public SparseAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                           double ropeTheta, boolean useRope, boolean qkvBias,
                           int window, int nGlobal, int nRandom, Pattern pattern, long randomSeed) {
        super("SparseAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.window = Math.max(1, window);
        this.nGlobal = Math.max(0, nGlobal);
        this.nRandom = Math.max(0, nRandom);
        this.pattern = pattern == null ? Pattern.LONGFORMER : pattern;
        this.randomSeed = randomSeed;

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

    public static SparseAttention longformer(long hiddenSize, int nHeads, double ropeTheta,
                                             int window, int nGlobal) {
        return new SparseAttention(hiddenSize, nHeads, nHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, window, nGlobal, 0, Pattern.LONGFORMER, 0L);
    }

    public static SparseAttention bigbird(long hiddenSize, int nHeads, double ropeTheta,
                                          int window, int nGlobal, int nRandom) {
        return new SparseAttention(hiddenSize, nHeads, nHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, window, nGlobal, nRandom, Pattern.BIGBIRD, 42L);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public int window() { return window; }
    public int nGlobal() { return nGlobal; }
    public Pattern pattern() { return pattern; }

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

        long Tk = pastLen + T;
        Tensor mask;
        if (pastLen == 0) {
            mask = AttentionOps.longformerMask(T, window, nGlobal);
            if (pattern == Pattern.BIGBIRD && nRandom > 0) {
                mask = addRandomLinks(mask, T, nRandom, randomSeed);
            }
        } else {
            // For cached path, fall back to sliding-window causal (global less meaningful mid-decode)
            mask = AttentionOps.causalMaskCached(pastLen, T, window);
        }

        double sc = AttentionOps.scale(headDim);
        Tensor[] sdpa = AttentionOps.denseSdpa(q, k, v, mask, sc, 0.0, false);
        Tensor y = sdpa[0].transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV};
    }

    /** Deterministic extra BigBird random (causal) edges. */
    static Tensor addRandomLinks(Tensor mask, long T, int nRandom, long seed) {
        // LCG for reproducibility without Math.random
        long state = seed == 0 ? 1L : seed;
        for (long i = 0; i < T; i++) {
            for (int r = 0; r < nRandom; r++) {
                state = (state * 6364136223846793005L + 1L);
                long j = Long.remainderUnsigned(state, i + 1); // attend only past/self
                mask.select(0, i).narrow(0, j, 1).fill_(new Scalar(0.0f));
            }
        }
        return mask;
    }
}
