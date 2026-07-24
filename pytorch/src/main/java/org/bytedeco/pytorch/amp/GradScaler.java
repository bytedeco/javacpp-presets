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
package org.bytedeco.pytorch.amp;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

import static org.bytedeco.pytorch.global.torch.isfinite;

/**
 * Mixed-precision gradient scaler (Java port of Python {@code torch.cuda.amp.GradScaler}).
 *
 * <p>Scales the loss before {@code backward()}, unscales gradients before
 * {@code optimizer.step()}, and halves the scale factor when non-finite
 * gradients are detected.
 *
 * <pre>{@code
 * GradScaler scaler = new GradScaler();
 * Tensor loss = scaler.scale(rawLoss);
 * loss.backward();
 * scaler.step(optimizer, model.parameters());
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GradScaler {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final float DEFAULT_INIT_SCALE = 65536.0f;
    private static final float GROWTH_FACTOR = 1.01f;
    private static final float BACKOFF_FACTOR = 0.5f;
    private static final float MAX_SCALE = 65536.0f;

    private final boolean enabled;
    private float scaleFactor;

    public GradScaler() {
        this(true, DEFAULT_INIT_SCALE);
    }

    public GradScaler(boolean enabled) {
        this(enabled, DEFAULT_INIT_SCALE);
    }

    public GradScaler(boolean enabled, float initScale) {
        this.enabled = enabled;
        this.scaleFactor = initScale;
    }

    /** Multiply {@code loss} by the current scale factor (no-op when disabled). */
    public Tensor scale(Tensor loss) {
        if (!enabled) {
            return loss;
        }
        return loss.mul(new Scalar(scaleFactor));
    }

    /**
     * Unscale gradients (if any are finite), then call {@code optimizer.step()}.
     * On non-finite grads: zero them, backoff the scale, skip the step.
     */
    public void step(Optimizer optimizer, TensorVector params) {
        if (!enabled) {
            optimizer.step();
            return;
        }

        if (hasNonFiniteGrad(params)) {
            zeroGrads(params);
            scaleFactor *= BACKOFF_FACTOR;
            return;
        }

        Scalar inv = new Scalar(1.0f / scaleFactor);
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor g = params.get(i).grad();
            if (g != null && g.defined()) {
                g.mul_(inv);
            }
        }
        optimizer.step();
        scaleFactor = Math.min(scaleFactor * GROWTH_FACTOR, MAX_SCALE);
    }

    /** No-op placeholder matching the Python API surface. */
    public void update() { }

    public float getScale() { return scaleFactor; }
    public boolean isEnabled() { return enabled; }

    private static boolean hasNonFiniteGrad(TensorVector params) {
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor g = params.get(i).grad();
            if (g != null && g.defined() && !isfinite(g).all().item_bool()) {
                return true;
            }
        }
        return false;
    }

    private static void zeroGrads(TensorVector params) {
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor g = params.get(i).grad();
            if (g != null && g.defined()) {
                g.zero_();
            }
        }
    }
}
