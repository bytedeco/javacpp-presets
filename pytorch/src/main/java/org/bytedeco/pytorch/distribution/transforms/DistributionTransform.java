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
package org.bytedeco.pytorch.distribution.transforms;

import org.bytedeco.pytorch.Tensor;

/**
 * Invertible (or half-invertible) map used by
 * {@link org.bytedeco.pytorch.distribution.TransformedDistribution}.
 *
 * <p>Mirrors {@code torch.distributions.transforms.Transform}: {@code forward},
 * {@code inverse}, and {@code logAbsDetJacobian}, plus {@code eventDim} for
 * event-shape reductions.
 */
public abstract class DistributionTransform implements AutoCloseable {

    /** Number of rightmost event dims this transform operates on. */
    public abstract int eventDim();

    /** Forward map {@code Y = T(X)}. */
    public abstract Tensor forward(Tensor x);

    /** Inverse map {@code X = T^{-1}(Y)} (may be a right-inverse if not bijective). */
    public abstract Tensor inverse(Tensor y);

    /** {@code log |det dy/dx|} evaluated at {@code (x, y=T(x))}. */
    public abstract Tensor logAbsDetJacobian(Tensor x, Tensor y);

    /** Whether {@code inverse} is a true two-sided inverse. Default {@code true}. */
    public boolean bijective() {
        return true;
    }

    /** Release owned tensors; default no-op. */
    @Override
    public void close() {
        // subclasses with owned state override
    }
}
