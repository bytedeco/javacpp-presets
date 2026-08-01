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

/**
 * Rotary position embedding cos/sin cache (llama-style).
 */
public final class RopeCache {
    private final int nRot;
    private final int nCtx;
    private final float[] cos; // [nCtx * nRot]
    private final float[] sin;

    public RopeCache(int nRot, int nCtx, float theta) {
        this.nRot = Math.max(2, nRot);
        this.nCtx = Math.max(1, nCtx);
        this.cos = new float[this.nCtx * this.nRot];
        this.sin = new float[this.nCtx * this.nRot];
        float base = theta > 0 ? theta : 10000f;
        for (int pos = 0; pos < this.nCtx; pos++) {
            for (int i = 0; i < this.nRot; i += 2) {
                double freq = 1.0 / Math.pow(base, (double) i / (double) this.nRot);
                double ang = pos * freq;
                int idx = pos * this.nRot + i;
                float c = (float) Math.cos(ang);
                float s = (float) Math.sin(ang);
                cos[idx] = c;
                cos[idx + 1] = c;
                sin[idx] = s;
                sin[idx + 1] = s;
            }
        }
    }

    /** Apply RoPE in-place on pair dims of vector length nRot (or head_dim). */
    public void apply(float[] x, int pos) {
        if (x == null || x.length < 2) return;
        int p = Math.min(Math.max(pos, 0), nCtx - 1);
        int n = Math.min(x.length, nRot);
        if ((n & 1) == 1) n--;
        int base = p * nRot;
        for (int i = 0; i < n; i += 2) {
            float c = cos[base + i];
            float s = sin[base + i];
            float x0 = x[i];
            float x1 = x[i + 1];
            x[i] = x0 * c - x1 * s;
            x[i + 1] = x0 * s + x1 * c;
        }
    }
}
