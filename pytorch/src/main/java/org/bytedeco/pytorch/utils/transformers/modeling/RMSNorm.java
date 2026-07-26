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
package org.bytedeco.pytorch.utils.transformers.modeling;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.pow;
import static org.bytedeco.pytorch.global.torch.rsqrt;

/**
 * RMSNorm as used by Llama / Qwen2 / Mistral.
 *
 * <pre>
 *   weight * x / sqrt(mean(x^2) + eps)
 * </pre>
 *
 * Parameter name {@code weight} matches HF {@code *.input_layernorm.weight} etc.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RMSNorm extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long hiddenSize;
    private final double eps;
    /** Registered parameter {@code weight}, shape {@code [hidden]}. */
    private Tensor weight;

    public RMSNorm(long hiddenSize, double eps) {
        super("RMSNorm");
        this.hiddenSize = hiddenSize;
        this.eps = eps;
        this.weight = register_parameter("weight", ones(hiddenSize), /*requires_grad=*/true);
    }

    public RMSNorm(long hiddenSize) {
        this(hiddenSize, 1e-6);
    }

    public Tensor weight() {
        return weight;
    }

    public long hiddenSize() {
        return hiddenSize;
    }

    public double eps() {
        return eps;
    }

    @Override
    public Tensor forward(Tensor x) {
        // variance over last dim, keepdim — explicit 3-arg overload to avoid varargs ambiguity
        Tensor sq = pow(x, new Scalar(2.0));
        Tensor variance = sq.mean(new long[]{x.dim() - 1}, /*keepdim=*/true, new ScalarTypeOptional());
        Tensor invRms = rsqrt(variance.add(new Scalar(eps)));
        return weight.mul(x.mul(invRms));
    }
}
