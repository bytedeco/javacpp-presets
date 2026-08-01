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

/** RMSNorm: x * weight / sqrt(mean(x^2) + eps) */
public final class RmsNormOp {
    private RmsNormOp() {}

    public static void forward(float[] x, float[] weight, float eps) {
        if (x == null || x.length == 0) return;
        double ss = 0;
        for (float v : x) ss += (double) v * v;
        float scale = (float) (1.0 / Math.sqrt(ss / x.length + eps));
        if (weight != null && weight.length >= x.length) {
            for (int i = 0; i < x.length; i++) x[i] = x[i] * scale * weight[i];
        } else {
            for (int i = 0; i < x.length; i++) x[i] = x[i] * scale;
        }
    }
}
