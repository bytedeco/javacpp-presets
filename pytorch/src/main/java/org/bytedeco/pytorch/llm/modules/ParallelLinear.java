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
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.DeviceMesh;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.TensorParallel;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.silu;

/**
 * Tensor-parallel Linear and MLP shells for distributed LLM training / inference.
 *
 * <p>When {@code pg == null} or world size == 1, falls back to ordinary
 * {@link LinearImpl} so the same module graph runs single-process.
 *
 * <ul>
 *   <li>{@link ColumnParallelLinear} — shard out-features (Megatron column)</li>
 *   <li>{@link RowParallelLinear} — shard in-features (Megatron row)</li>
 *   <li>{@link ParallelSwiGLU} — column-parallel gate/up + row-parallel down</li>
 * </ul>
 *
 * <p>Wraps {@link TensorParallel} from {@code org.bytedeco.pytorch.distributed}
 * with an LLM-friendly single-rank fallback.
 */
public final class ParallelLinear {

    private ParallelLinear() {}

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class ColumnParallelLinear extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        private final LinearImpl local;
        private final TensorParallel.ColumnParallelLinear distributed;
        private final boolean multiRank;
        private final long fullOutFeatures;
        private final long localOutFeatures;

        public ColumnParallelLinear(long inFeatures, long outFeatures,
                                    ProcessGroupWrapper pg, DeviceMesh tpMesh,
                                    boolean bias) {
            super("ColumnParallelLinear");
            int tp = tpSize(pg, tpMesh);
            this.multiRank = tp > 1 && pg != null;
            this.fullOutFeatures = outFeatures;
            if (multiRank) {
                if (outFeatures % tp != 0) {
                    throw new IllegalArgumentException(
                            "outFeatures=" + outFeatures + " not divisible by tp=" + tp);
                }
                this.localOutFeatures = outFeatures / tp;
                this.distributed = register_module("dist",
                        new TensorParallel.ColumnParallelLinear(inFeatures, outFeatures, pg, tpMesh));
                this.local = null;
            } else {
                this.localOutFeatures = outFeatures;
                this.distributed = null;
                this.local = register_module("local",
                        new LinearImpl(new LinearOptions(inFeatures, outFeatures).bias(bias)));
            }
        }

        public ColumnParallelLinear(long inFeatures, long outFeatures) {
            this(inFeatures, outFeatures, null, null, true);
        }

        @Override
        public Tensor forward(Tensor x) {
            return multiRank ? distributed.forward(x) : local.forward(x);
        }

        public long fullOutFeatures() { return fullOutFeatures; }
        public long localOutFeatures() { return localOutFeatures; }
        public boolean multiRank() { return multiRank; }
        public LinearImpl localLinear() {
            return multiRank ? distributed.localLinear() : local;
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class RowParallelLinear extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        private final LinearImpl local;
        private final TensorParallel.RowParallelLinear distributed;
        private final boolean multiRank;
        private final long fullInFeatures;
        private final long localInFeatures;

        public RowParallelLinear(long inFeatures, long outFeatures,
                                 ProcessGroupWrapper pg, DeviceMesh tpMesh,
                                 boolean bias) {
            super("RowParallelLinear");
            int tp = tpSize(pg, tpMesh);
            this.multiRank = tp > 1 && pg != null;
            this.fullInFeatures = inFeatures;
            if (multiRank) {
                if (inFeatures % tp != 0) {
                    throw new IllegalArgumentException(
                            "inFeatures=" + inFeatures + " not divisible by tp=" + tp);
                }
                this.localInFeatures = inFeatures / tp;
                this.distributed = register_module("dist",
                        new TensorParallel.RowParallelLinear(inFeatures, outFeatures, pg, tpMesh));
                this.local = null;
            } else {
                this.localInFeatures = inFeatures;
                this.distributed = null;
                this.local = register_module("local",
                        new LinearImpl(new LinearOptions(inFeatures, outFeatures).bias(bias)));
            }
        }

        public RowParallelLinear(long inFeatures, long outFeatures) {
            this(inFeatures, outFeatures, null, null, true);
        }

        @Override
        public Tensor forward(Tensor x) {
            return multiRank ? distributed.forward(x) : local.forward(x);
        }

        public long fullInFeatures() { return fullInFeatures; }
        public long localInFeatures() { return localInFeatures; }
        public boolean multiRank() { return multiRank; }
    }

    /**
     * Tensor-parallel SwiGLU: gate/up are column-parallel (sharded intermediate),
     * down is row-parallel. Single-rank falls back to {@link Mlp.SwiGLU}.
     */
    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class ParallelSwiGLU extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        private final Mlp.SwiGLU dense;
        private final ColumnParallelLinear gate_proj;
        private final ColumnParallelLinear up_proj;
        private final RowParallelLinear down_proj;
        private final boolean multiRank;
        private final long hiddenSize;
        private final long intermediateSize;

        public ParallelSwiGLU(long hiddenSize, long intermediateSize,
                              ProcessGroupWrapper pg, DeviceMesh tpMesh) {
            super("ParallelSwiGLU");
            this.hiddenSize = hiddenSize;
            this.intermediateSize = intermediateSize;
            int tp = tpSize(pg, tpMesh);
            this.multiRank = tp > 1 && pg != null;
            if (multiRank) {
                this.dense = null;
                this.gate_proj = register_module("gate_proj",
                        new ColumnParallelLinear(hiddenSize, intermediateSize, pg, tpMesh, false));
                this.up_proj = register_module("up_proj",
                        new ColumnParallelLinear(hiddenSize, intermediateSize, pg, tpMesh, false));
                this.down_proj = register_module("down_proj",
                        new RowParallelLinear(intermediateSize, hiddenSize, pg, tpMesh, false));
            } else {
                this.dense = register_module("dense", new Mlp.SwiGLU(hiddenSize, intermediateSize));
                this.gate_proj = null;
                this.up_proj = null;
                this.down_proj = null;
            }
        }

        public ParallelSwiGLU(long hiddenSize, long intermediateSize) {
            this(hiddenSize, intermediateSize, null, null);
        }

        @Override
        public Tensor forward(Tensor x) {
            if (!multiRank) {
                return dense.forward(x);
            }
            // Note: column-parallel gate/up allgather to full intermediate, then
            // row-parallel down expects full intermediate input and reduces.
            Tensor gate = gate_proj.forward(x);
            Tensor up = up_proj.forward(x);
            return down_proj.forward(silu(gate).mul(up));
        }

        public long hiddenSize() { return hiddenSize; }
        public long intermediateSize() { return intermediateSize; }
        public boolean multiRank() { return multiRank; }
    }

    private static int tpSize(ProcessGroupWrapper pg, DeviceMesh tpMesh) {
        if (pg == null) {
            return 1;
        }
        return TensorParallel.tpWorld(pg, tpMesh);
    }
}
