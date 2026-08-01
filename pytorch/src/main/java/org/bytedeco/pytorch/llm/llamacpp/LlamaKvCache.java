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

package org.bytedeco.pytorch.llm.llamacpp;

import java.util.Arrays;

/**
 * Simple per-layer KV cache holding float arrays {@code [n_ctx, n_head_kv * head_dim]}.
 * Used by the pure-Java in-process path; process-server keeps KV inside llama.cpp.
 */
public final class LlamaKvCache {

    private final int nLayer;
    private final int nCtx;
    private final int nEmbdKv; // n_head_kv * head_dim
    private final float[][] k; // [layer][n_ctx * nEmbdKv]
    private final float[][] v;
    private int nPast;

    public LlamaKvCache(LlamaHParams hp, int nCtx) {
        this.nLayer = hp.nLayer();
        this.nCtx = Math.max(8, nCtx);
        this.nEmbdKv = Math.max(1, hp.nHeadKv() * hp.headDim());
        this.k = new float[nLayer][this.nCtx * nEmbdKv];
        this.v = new float[nLayer][this.nCtx * nEmbdKv];
        this.nPast = 0;
    }

    public int nPast() { return nPast; }
    public int nCtx() { return nCtx; }
    public int nLayer() { return nLayer; }
    public int nEmbdKv() { return nEmbdKv; }

    public void reset() {
        nPast = 0;
        for (int i = 0; i < nLayer; i++) {
            Arrays.fill(k[i], 0f);
            Arrays.fill(v[i], 0f);
        }
    }

    /** Append one position of K/V for a layer. {@code row} length = nEmbdKv. */
    public void append(int layer, float[] kRow, float[] vRow) {
        if (layer < 0 || layer >= nLayer) throw new IllegalArgumentException("layer");
        if (nPast >= nCtx) {
            // drop oldest (shift left by 1) — simple sliding window
            System.arraycopy(k[layer], nEmbdKv, k[layer], 0, (nCtx - 1) * nEmbdKv);
            System.arraycopy(v[layer], nEmbdKv, v[layer], 0, (nCtx - 1) * nEmbdKv);
            nPast = nCtx - 1;
        }
        int off = nPast * nEmbdKv;
        System.arraycopy(kRow, 0, k[layer], off, Math.min(nEmbdKv, kRow.length));
        System.arraycopy(vRow, 0, v[layer], off, Math.min(nEmbdKv, vRow.length));
    }

    /** Call once after all layers appended for a token. */
    public void advance() {
        if (nPast < nCtx) nPast++;
    }

    public float[] keys(int layer) { return k[layer]; }
    public float[] values(int layer) { return v[layer]; }
}
