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

package org.bytedeco.pytorch.llm.llamacpp.model;

import org.bytedeco.pytorch.llm.llamacpp.LlamaKvCache;

/**
 * Multi-head self-attention over float arrays with optional KV cache.
 * Shapes: q/k/v projected to n_embd, split into heads.
 */
public final class AttentionOp {
    private AttentionOp() {}

    public static float[] forward(float[] x,
                                  float[] wq, float[] wk, float[] wv, float[] wo,
                                  int nEmbd, int nHead, int nHeadKv, int headDim,
                                  RopeCache rope, int pos,
                                  LlamaKvCache kv, int layer,
                                  boolean useCache) {
        float[] q = MlpOp.matvec(x, wq, nEmbd, nHead * headDim);
        float[] k = MlpOp.matvec(x, wk, nEmbd, nHeadKv * headDim);
        float[] v = MlpOp.matvec(x, wv, nEmbd, nHeadKv * headDim);

        // RoPE on each head of q and k
        if (rope != null) {
            for (int h = 0; h < nHead; h++) {
                float[] qh = new float[headDim];
                System.arraycopy(q, h * headDim, qh, 0, headDim);
                rope.apply(qh, pos);
                System.arraycopy(qh, 0, q, h * headDim, headDim);
            }
            for (int h = 0; h < nHeadKv; h++) {
                float[] kh = new float[headDim];
                System.arraycopy(k, h * headDim, kh, 0, headDim);
                rope.apply(kh, pos);
                System.arraycopy(kh, 0, k, h * headDim, headDim);
            }
        }

        int seq = 1;
        float[] kAll = k;
        float[] vAll = v;
        int nPast = 0;
        if (useCache && kv != null) {
            nPast = kv.nPast();
            kv.append(layer, k, v);
            // read full cache including current after append but before advance
            kAll = new float[(nPast + 1) * nHeadKv * headDim];
            vAll = new float[(nPast + 1) * nHeadKv * headDim];
            System.arraycopy(kv.keys(layer), 0, kAll, 0, (nPast + 1) * nHeadKv * headDim);
            System.arraycopy(kv.values(layer), 0, vAll, 0, (nPast + 1) * nHeadKv * headDim);
            seq = nPast + 1;
        }

        float[] ctx = new float[nHead * headDim];
        float scale = (float) (1.0 / Math.sqrt(headDim));
        int rep = Math.max(1, nHead / Math.max(1, nHeadKv)); // GQA

        for (int h = 0; h < nHead; h++) {
            int hkv = h / rep;
            float[] scores = new float[seq];
            float max = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < seq; t++) {
                double dot = 0;
                int qOff = h * headDim;
                int kOff = t * nHeadKv * headDim + hkv * headDim;
                for (int d = 0; d < headDim; d++) {
                    dot += q[qOff + d] * kAll[kOff + d];
                }
                scores[t] = (float) (dot * scale);
                if (scores[t] > max) max = scores[t];
            }
            double sum = 0;
            for (int t = 0; t < seq; t++) {
                scores[t] = (float) Math.exp(scores[t] - max);
                sum += scores[t];
            }
            if (sum <= 0) sum = 1;
            for (int t = 0; t < seq; t++) scores[t] /= sum;

            int oOff = h * headDim;
            for (int d = 0; d < headDim; d++) {
                double acc = 0;
                for (int t = 0; t < seq; t++) {
                    int vOff = t * nHeadKv * headDim + hkv * headDim + d;
                    acc += scores[t] * vAll[vOff];
                }
                ctx[oOff + d] = (float) acc;
            }
        }
        return MlpOp.matvec(ctx, wo, nHead * headDim, nEmbd);
    }
}
