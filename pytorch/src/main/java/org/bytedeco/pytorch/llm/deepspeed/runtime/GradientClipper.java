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
package org.bytedeco.pytorch.llm.deepspeed.runtime;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.llm.deepspeed.zero.PartitionedParameter;

import java.util.ArrayList;
import java.util.List;

/**
 * Global gradient L2 clipping with optional cross-rank norm reduction.
 */
public final class GradientClipper {

    private GradientClipper() {}

    /**
     * @return total L2 norm before clipping (local or global if pg provided)
     */
    public static double clipGradNorm(List<PartitionedParameter> partitions,
                                      double maxNorm,
                                      ProcessGroupWrapper processGroup) {
        if (maxNorm <= 0 || partitions == null || partitions.isEmpty()) {
            return 0.0;
        }
        List<Tensor> grads = new ArrayList<>();
        double totalSq = 0.0;
        for (PartitionedParameter part : partitions) {
            try {
                Tensor g = part.param.grad();
                if (g != null && !g.isNull() && g.defined()) {
                    grads.add(g);
                    double v = g.norm().item_float();
                    totalSq += v * v;
                }
            } catch (Exception ignored) {
            }
        }
        // Allreduce sum of squares for global norm
        if (processGroup != null && processGroup.getWorldSize() > 1) {
            try {
                Tensor t = org.bytedeco.pytorch.global.torch.tensor(new float[]{(float) totalSq});
                processGroup.allreduce(t);
                totalSq = t.item_float();
            } catch (Exception ignored) {
            }
        }
        double total = Math.sqrt(Math.max(0.0, totalSq));
        if (total > maxNorm && total > 0 && !grads.isEmpty()) {
            double scale = maxNorm / (total + 1e-6);
            Scalar s = new Scalar(scale);
            for (Tensor g : grads) {
                g.mul_(s);
            }
        }
        return total;
    }

    public static double computeGradNorm(List<PartitionedParameter> partitions) {
        double totalSq = 0.0;
        if (partitions == null) return 0.0;
        for (PartitionedParameter part : partitions) {
            try {
                Tensor g = part.param.grad();
                if (g != null && !g.isNull() && g.defined()) {
                    double v = g.norm().item_float();
                    totalSq += v * v;
                }
            } catch (Exception ignored) {
            }
        }
        return Math.sqrt(Math.max(0.0, totalSq));
    }
}
