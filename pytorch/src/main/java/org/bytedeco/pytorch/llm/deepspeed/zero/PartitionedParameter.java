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
package org.bytedeco.pytorch.llm.deepspeed.zero;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Tensor;

/**
 * ZeRO parameter partition metadata for one model parameter tensor.
 */
public final class PartitionedParameter {
    public final int index;
    public final int ownerRank;
    public final boolean local;
    public final long numel;
    public final Tensor param;
    /** Stage-3: whether a full gather is currently held on this rank. */
    public boolean gathered;
    /** Estimated optimizer-state bytes owned by this rank (stage >= 1). */
    public final long optimStateBytesLocal;

    public PartitionedParameter(int index, int ownerRank, boolean local, long numel, Tensor param,
                                int worldSize, int zeroStage) {
        this.index = index;
        this.ownerRank = ownerRank;
        this.local = local;
        this.numel = numel;
        this.param = param;
        this.gathered = false;
        long world = Math.max(1, worldSize);
        // Adam-like: 2 states * 4 bytes ≈ 8 bytes / param, sharded at stage >= 1
        this.optimStateBytesLocal = zeroStage >= 1 ? (numel * 8L) / world : numel * 8L;
    }

    public long gradBytesLocal(int worldSize, int zeroStage) {
        long world = Math.max(1, worldSize);
        if (zeroStage >= 2) {
            return local ? (numel * 4L) : 0L;
        }
        return numel * 4L;
    }

    public long paramBytesLocal(int worldSize, int zeroStage) {
        long world = Math.max(1, worldSize);
        if (zeroStage >= 3) {
            return local ? (numel * 4L) : 0L;
        }
        return numel * 4L;
    }

    @Override
    public String toString() {
        return "PartitionedParameter{i=" + index + ", owner=" + ownerRank
                + ", local=" + local + ", numel=" + numel + ", gathered=" + gathered + '}';
    }
}
