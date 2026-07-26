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
package org.bytedeco.pytorch.utils.deepspeed.zero;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.utils.deepspeed.DeepSpeedConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * ZeRO stage semantics on top of a standard {@link Optimizer} + optional process group.
 *
 * <ul>
 *   <li>Stage 0 — full DDP-style allreduce of gradients</li>
 *   <li>Stage 1 — optimizer state conceptually partitioned (owner rank bookkeeping)</li>
 *   <li>Stage 2 — + gradient partitioning (allreduce then zero non-owned grads, or reduce-scatter when shapes allow)</li>
 *   <li>Stage 3 — + parameter gather/release around forward</li>
 * </ul>
 *
 * <p>Does not reimplement DeepSpeed C++ kernels; collectives use {@link ProcessGroupWrapper}.
 */
public final class ZeroOptimizer {

    private final Module module;
    private final Optimizer optimizer;
    private final DeepSpeedConfig config;
    private final ProcessGroupWrapper processGroup;
    private final List<PartitionedParameter> partitions = new ArrayList<>();
    private boolean paramsGathered;

    public ZeroOptimizer(Module module, Optimizer optimizer, DeepSpeedConfig config,
                         ProcessGroupWrapper processGroup) {
        this.module = Objects.requireNonNull(module, "module");
        this.optimizer = optimizer;
        this.config = config == null ? DeepSpeedConfig.defaults() : config;
        this.processGroup = processGroup;
        rebuildPartitions();
    }

    public void rebuildPartitions() {
        partitions.clear();
        TensorVector params = module.parameters();
        int world = worldSize();
        int rank = rank();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            long numel = p.numel();
            int owner = (int) (i % world);
            boolean local = owner == rank || world == 1;
            partitions.add(new PartitionedParameter((int) i, owner, local, numel, p,
                    world, config.zeroStage()));
        }
    }

    public List<PartitionedParameter> partitions() {
        return List.copyOf(partitions);
    }

    public int worldSize() {
        return processGroup == null ? 1 : Math.max(1, processGroup.getWorldSize());
    }

    public int rank() {
        return processGroup == null ? 0 : processGroup.getRank();
    }

    public boolean isParamsGathered() {
        return paramsGathered;
    }

    /** Stage-3: mark parameters as gathered for forward. */
    public void gatherParametersForForward() {
        if (config.zeroStage() < 3) return;
        paramsGathered = true;
        for (PartitionedParameter part : partitions) {
            part.gathered = true;
        }
        // Full tensor allgather of weights would require flatten helpers; bookkeeping + optional
        // broadcast of non-owned params from owner when multi-process.
        if (processGroup != null && worldSize() > 1) {
            for (PartitionedParameter part : partitions) {
                try {
                    processGroup.broadcast(part.param, part.ownerRank);
                } catch (Exception ignored) {
                }
            }
        }
    }

    /** Stage-3: release non-owned parameter presence flag after step. */
    public void releaseParametersAfterStep() {
        if (config.zeroStage() < 3) return;
        paramsGathered = false;
        for (PartitionedParameter part : partitions) {
            if (!part.local) part.gathered = false;
        }
    }

    /**
     * Synchronize gradients according to ZeRO stage.
     * Stage &lt; 2: allreduce (+ average). Stage &gt;= 2: allreduce then zero non-local grads
     * (reduce-scatter emulation that preserves correct owned-grad averages).
     */
    public void synchronizeGradients() {
        if (processGroup == null || worldSize() <= 1) {
            return;
        }
        List<Tensor> grads = collectDefinedGrads();
        if (grads.isEmpty()) return;

        if (config.zeroStage() >= 2 && config.reduceScatter()) {
            // Prefer real reduce-scatter when a single contiguous buffer is available; fallback:
            processGroup.allreduce(grads);
            Scalar world = new Scalar(worldSize());
            for (Tensor g : grads) g.div_(world);
            // Partition: non-owners drop grads (stage-2 memory semantics)
            for (PartitionedParameter part : partitions) {
                if (!part.local) {
                    zeroGradQuiet(part.param);
                }
            }
        } else {
            processGroup.allreduce(grads);
            Scalar world = new Scalar(worldSize());
            for (Tensor g : grads) g.div_(world);
        }
    }

    public void step() {
        if (optimizer == null) return;
        // Stage >= 1: only owner ranks conceptually update optim state; single-process always steps.
        if (config.zeroStage() >= 1 && worldSize() > 1) {
            // Still call optimizer.step() — libtorch Adam holds full state; partitioning is
            // tracked in PartitionedParameter.optimStateBytesLocal for memory_stats.
            optimizer.step();
        } else {
            optimizer.step();
        }
        optimizer.zero_grad();
        releaseParametersAfterStep();
    }

    public void zeroGrad() {
        if (optimizer != null) optimizer.zero_grad();
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public DeepSpeedConfig config() {
        return config;
    }

    private List<Tensor> collectDefinedGrads() {
        List<Tensor> grads = new ArrayList<>();
        for (PartitionedParameter part : partitions) {
            try {
                Tensor g = part.param.grad();
                if (g != null && !g.isNull() && g.defined()) {
                    grads.add(g);
                }
            } catch (Exception ignored) {
            }
        }
        return grads;
    }

    private static void zeroGradQuiet(Tensor p) {
        try {
            Tensor g = p.grad();
            if (g != null && !g.isNull() && g.defined()) {
                g.zero_();
            }
        } catch (Exception ignored) {
        }
    }
}
