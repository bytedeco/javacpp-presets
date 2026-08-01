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

/** SwiGLU-style MLP used by Llama: silu(x@Wgate) * (x@Wup) @ Wdown */
public final class MlpOp {
    private MlpOp() {}

    public static float[] forward(float[] x, float[] wGate, float[] wUp, float[] wDown,
                                  int nEmbd, int nFF) {
        float[] gate = matvec(x, wGate, nEmbd, nFF);
        float[] up = matvec(x, wUp, nEmbd, nFF);
        for (int i = 0; i < nFF; i++) {
            gate[i] = silu(gate[i]) * up[i];
        }
        return matvec(gate, wDown, nFF, nEmbd);
    }

    /** Plain GELU MLP for GPT-2-ish: gelu(x@Wfc) @ Wproj */
    public static float[] forwardGpt(float[] x, float[] wFc, float[] wProj, int nEmbd, int nFF) {
        float[] h = matvec(x, wFc, nEmbd, nFF);
        for (int i = 0; i < nFF; i++) h[i] = gelu(h[i]);
        return matvec(h, wProj, nFF, nEmbd);
    }

    public static float silu(float x) {
        return x / (1.f + (float) Math.exp(-x));
    }

    public static float gelu(float x) {
        return 0.5f * x * (1.f + (float) Math.tanh(0.79788456 * (x + 0.044715 * x * x * x)));
    }

    /** y[out] = W[out, in] @ x[in] ; W row-major */
    public static float[] matvec(float[] x, float[] w, int in, int out) {
        float[] y = new float[out];
        if (w == null || x == null) return y;
        int need = in * out;
        if (w.length < need) {
            // best-effort partial
            need = w.length;
        }
        for (int o = 0; o < out; o++) {
            double sum = 0;
            int row = o * in;
            int lim = Math.min(in, Math.max(0, w.length - row));
            for (int i = 0; i < lim; i++) {
                sum += w[row + i] * (i < x.length ? x[i] : 0f);
            }
            y[o] = (float) sum;
        }
        return y;
    }
}
