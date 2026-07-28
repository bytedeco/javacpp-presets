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
package org.bytedeco.pytorch.distributed;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Shared loss helpers for distributed trainers.
 *
 * <p>Handles the common MockLLM / LM case where logits are {@code [B, T, V]}
 * (or {@code [N, V]}) and labels are class indices. Flattens to
 * {@code cross_entropy([N, C], [N])} as required by ATen.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DistributedLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private DistributedLoss() {}

    /**
     * Cross-entropy that accepts:
     * <ul>
     *   <li>logits {@code [N, C]} + target {@code [N]}</li>
     *   <li>logits {@code [B, T, V]} + target {@code [B, T]} or {@code [B]} (broadcast last)</li>
     *   <li>logits {@code [B, T, V]} + target {@code [B*T]}</li>
     * </ul>
     */
    public static Tensor crossEntropy(Tensor logits, Tensor target) {
        Tensor y = target;
        ScalarType st = y.scalar_type().intern();
        if (st != ScalarType.Long && st != ScalarType.Byte && st != ScalarType.Char) {
            y = y.to(ScalarType.Long);
        }

        int ld = (int) logits.dim();
        if (ld >= 3) {
            // [B, T, ..., V] → [N, V]
            long v = logits.size(ld - 1);
            long n = logits.numel() / v;
            Tensor flatLogits = logits.reshape(n, v);
            Tensor flatTarget;
            if (y.numel() == n) {
                flatTarget = y.reshape(n);
            } else if (y.dim() == 1 && y.size(0) == logits.size(0) && logits.size(0) * seqLen(logits) == n) {
                // target [B] with logits [B, T, V]: expand per-token by repeating
                // For smoke demos, take first token position only via narrow
                // Safer: require matching numel — fall through to error otherwise.
                // Use mean over batch only first timestep if T>1 and target is [B].
                long b = logits.size(0);
                long t = seqLen(logits);
                // labels [B] → take logits at last time step [B, V]
                Tensor last = logits.select(1, t - 1); // [B, V]
                return cross_entropy(last, y.reshape(b));
            } else {
                // best-effort reshape; may throw with clear ATen message
                flatTarget = y.reshape(n);
            }
            return cross_entropy(flatLogits, flatTarget);
        }
        if (ld == 2) {
            // [N, C]
            return cross_entropy(logits, y.reshape(logits.size(0)));
        }
        // 1D logits unlikely; pass through
        return cross_entropy(logits, y);
    }

    private static long seqLen(Tensor logits) {
        // logits [B, T, V]
        return logits.dim() >= 3 ? logits.size(1) : 1L;
    }
}
