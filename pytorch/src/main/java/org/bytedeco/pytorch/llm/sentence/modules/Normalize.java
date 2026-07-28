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
package org.bytedeco.pytorch.llm.sentence.modules;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

/**
 * Sentence-Transformers Normalize module — L2-normalizes the last dimension
 * so each row has unit norm (cosine-ready embeddings).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Normalize extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final double eps;

    public Normalize() {
        this(1e-12);
    }

    public Normalize(double eps) {
        super("Normalize");
        this.eps = eps;
    }

    public double eps() { return eps; }

    @Override
    public Tensor forward(Tensor x) {
        long last = x.dim() - 1L;
        Tensor n = x.norm(new ScalarOptional(new Scalar(2.0)), new long[]{last}, true);
        return x.div(n.clamp_min(new Scalar(eps)));
    }
}
