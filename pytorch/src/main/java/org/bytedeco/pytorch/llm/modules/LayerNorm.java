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
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;

/**
 * GPT-2 / BERT style LayerNorm wrapper with HF-friendly defaults.
 *
 * <p>Uses native {@link LayerNormImpl}. Parameter names match PyTorch
 * ({@code weight}, {@code bias}).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LayerNorm extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long hiddenSize;
    private final double eps;
    private final LayerNormImpl inner;

    public LayerNorm(long hiddenSize, double eps) {
        super("LayerNorm");
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be > 0");
        }
        this.hiddenSize = hiddenSize;
        this.eps = eps;
        LongVector shape = new LongVector().put(hiddenSize);
        LayerNormOptions opts = new LayerNormOptions(shape).eps(eps).elementwise_affine(true);
        this.inner = register_module("inner", new LayerNormImpl(opts));
    }

    public LayerNorm(long hiddenSize) {
        this(hiddenSize, 1e-5);
    }

    public long hiddenSize() {
        return hiddenSize;
    }

    public double eps() {
        return eps;
    }

    public LayerNormImpl inner() {
        return inner;
    }

    @Override
    public Tensor forward(Tensor x) {
        return inner.forward(x);
    }
}
