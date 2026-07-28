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
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.empty;

/**
 * Tensor-parallel helpers (column / row parallel Linear) + thin trainer.
 *
 * <p>Not full Megatron-LM (no sequence parallel / attention head split). Uses
 * real allgather / allreduce on the TP process group (or full world when mesh
 * is 1D). Suitable for engineering demos and hybrid DP+TP benchmarks.
 *
 * <ul>
 *   <li>{@link ColumnParallelLinear}: weight sharded on out-features; allgather outputs</li>
 *   <li>{@link RowParallelLinear}: weight sharded on in-features; reduce outputs</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class TensorParallel {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private TensorParallel() {}

    public static int tpWorld(ProcessGroupWrapper pg, DeviceMesh tpMesh) {
        if (tpMesh != null) return Math.max(1, tpMesh.size());
        return Math.max(1, pg.getWorldSize());
    }

    public static int tpRank(ProcessGroupWrapper pg, DeviceMesh tpMesh) {
        if (tpMesh != null) {
            int lr = tpMesh.localRank();
            return lr >= 0 ? lr : pg.getRank();
        }
        return pg.getRank();
    }

    /**
     * Column-parallel linear: each rank holds {@code out/tp} columns.
     * Forward: local matmul then allgather along last dim (equal shards).
     */
    public static final class ColumnParallelLinear extends Module {
        private final LinearImpl local;
        private final ProcessGroupWrapper pg;
        private final DeviceMesh tpMesh;
        private final long fullOutFeatures;
        private final long localOutFeatures;
        private final int tpSize;
        private final int tpRank;

        public ColumnParallelLinear(long inFeatures, long outFeatures,
                                    ProcessGroupWrapper pg, DeviceMesh tpMesh) {
            super("ColumnParallelLinear");
            this.pg = Objects.requireNonNull(pg, "pg");
            this.tpMesh = tpMesh;
            this.tpSize = tpWorld(pg, tpMesh);
            this.tpRank = tpRank(pg, tpMesh);
            if (outFeatures % tpSize != 0) {
                throw new IllegalArgumentException(
                        "outFeatures=" + outFeatures + " not divisible by tpSize=" + tpSize);
            }
            this.fullOutFeatures = outFeatures;
            this.localOutFeatures = outFeatures / tpSize;
            this.local = register_module("local", new LinearImpl(inFeatures, localOutFeatures));
        }

        public Tensor forward(Tensor input) {
            Tensor partial = local.forward(input);
            if (tpSize <= 1) return partial;
            int dim = (int) partial.dim();
            long[] shape = new long[dim];
            for (int i = 0; i < dim; i++) shape[i] = partial.sizes().get(i);
            long last = shape[dim - 1];
            long prefix = partial.numel() / last;
            Tensor flat = partial.reshape(prefix * last);
            Tensor gathered = empty(flat.numel() * (long) tpSize)
                    .to(partial.device(), partial.scalar_type());
            pg.allgatherBase(gathered, flat)._wait();
            long[] outShape = shape.clone();
            outShape[dim - 1] = last * tpSize;
            Tensor view = gathered.reshape(tpSize, prefix, last);
            Tensor perm = view.permute(1L, 0L, 2L).contiguous();
            return perm.reshape(outShape);
        }

        public long fullOutFeatures() { return fullOutFeatures; }
        public long localOutFeatures() { return localOutFeatures; }
        public LinearImpl localLinear() { return local; }
    }

    /**
     * Row-parallel linear: each rank holds {@code in/tp} rows of weight
     * (local in-features). Input should be split or already local; output is
     * allreduced (SUM) across TP ranks.
     */
    public static final class RowParallelLinear extends Module {
        private final LinearImpl local;
        private final ProcessGroupWrapper pg;
        private final long localInFeatures;
        private final int tpSize;

        public RowParallelLinear(long inFeatures, long outFeatures,
                                 ProcessGroupWrapper pg, DeviceMesh tpMesh) {
            super("RowParallelLinear");
            this.pg = Objects.requireNonNull(pg, "pg");
            this.tpSize = tpWorld(pg, tpMesh);
            if (inFeatures % tpSize != 0) {
                throw new IllegalArgumentException(
                        "inFeatures=" + inFeatures + " not divisible by tpSize=" + tpSize);
            }
            this.localInFeatures = inFeatures / tpSize;
            this.local = register_module("local", new LinearImpl(localInFeatures, outFeatures));
        }

        /**
         * @param input full or already-split last-dim input. If last dim == full in,
         *              this rank slices its shard; if last dim == localIn, used as-is.
         */
        public Tensor forward(Tensor input) {
            Tensor localIn = input;
            int dim = (int) input.dim();
            long last = input.sizes().get(dim - 1);
            if (last == localInFeatures * (long) tpSize && tpSize > 1) {
                int rank = pg.getRank() % tpSize;
                long start = (long) rank * localInFeatures;
                localIn = input.narrow(dim - 1, start, localInFeatures);
            }
            Tensor partial = local.forward(localIn);
            if (tpSize > 1) {
                pg.allreduce(partial);
            }
            return partial;
        }

        public LinearImpl localLinear() { return local; }
    }

    /**
     * Minimal TP trainer: wraps a Module, optional allreduce of grads on TP group
     * after backward (when weights are replicated on DP and sharded on TP, grad
     * sync on DP is separate — here we allreduce on the provided PG for simplicity).
     */
    public static final class TPTrainer implements AutoCloseable {
        private final Module model;
        private final ProcessGroupWrapper pg;
        private final DeviceMesh tpMesh;
        private final ModuleForward forward;
        private long steps;

        public TPTrainer(Module model, ProcessGroupWrapper pg, DeviceMesh tpMesh) {
            this.model = Objects.requireNonNull(model, "model");
            this.pg = Objects.requireNonNull(pg, "pg");
            this.tpMesh = tpMesh;
            this.forward = ModuleForward.of(model);
            model.to(pg.getDevice(), true);
        }

        public static TPTrainer create(Module model, ProcessGroupWrapper pg) {
            return new TPTrainer(model, pg, null);
        }

        public static TPTrainer create(Module model, ProcessGroupWrapper pg, DeviceMesh tpMesh) {
            return new TPTrainer(model, pg, tpMesh);
        }

        public Tensor forward(Tensor input) {
            return forward.apply(model, input);
        }

        public Tensor step(Tensor input, Tensor target, Optimizer opt) {
            Tensor out = forward(input);
            Tensor loss = DistributedLoss.crossEntropy(out, target);
            opt.zero_grad();
            loss.backward();
            // Data-parallel style grad sync on full PG (hybrid: caller may pass dp mesh PG)
            if (pg.getWorldSize() > 1) {
                java.util.ArrayList<Tensor> grads = new java.util.ArrayList<>();
                var params = model.parameters();
                for (long i = 0, n = params.size(); i < n; i++) {
                    Tensor p = params.get(i);
                    if (p == null || p.isNull()) continue;
                    try {
                        Tensor g = p.grad();
                        if (g != null && !g.isNull() && g.defined()) grads.add(g);
                    } catch (Exception ignored) {}
                }
                if (!grads.isEmpty()) {
                    pg.averageGradients(grads);
                }
            }
            opt.step();
            steps++;
            return loss;
        }

        public Module getModule() { return model; }
        public ProcessGroupWrapper getProcessGroup() { return pg; }
        public DeviceMesh getTpMesh() { return tpMesh; }
        public long getSteps() { return steps; }

        @Override
        public void close() {}
    }
}
