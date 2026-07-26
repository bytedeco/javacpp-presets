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
package org.bytedeco.pytorch.utils.accelerate.plugins;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.distributed.FSDPTrainer;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.ShardingStrategy;
import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;

/**
 * HF Accelerate {@code FullyShardedDataParallelPlugin} — wraps {@link FSDPTrainer}.
 */
public final class FullyShardedDataParallelPlugin {

    private final ShardingStrategy shardingStrategy;
    private final boolean reshardAfterForward;
    private final boolean useFullPrecision;
    private FSDPTrainer trainer;

    public FullyShardedDataParallelPlugin() {
        this(ShardingStrategy.FULL_SHARD, true, true);
    }

    public FullyShardedDataParallelPlugin(ShardingStrategy shardingStrategy,
                                          boolean reshardAfterForward,
                                          boolean useFullPrecision) {
        this.shardingStrategy = shardingStrategy == null ? ShardingStrategy.FULL_SHARD : shardingStrategy;
        this.reshardAfterForward = reshardAfterForward;
        this.useFullPrecision = useFullPrecision;
    }

    public static FullyShardedDataParallelPlugin fullShard() {
        return new FullyShardedDataParallelPlugin();
    }

    public ShardingStrategy shardingStrategy() { return shardingStrategy; }
    public boolean reshardAfterForward() { return reshardAfterForward; }
    public boolean useFullPrecision() { return useFullPrecision; }
    public FSDPTrainer trainer() { return trainer; }
    public boolean isInitialized() { return trainer != null; }

    public FSDPTrainer wrap(Module module, ProcessGroupWrapper pg) {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(pg, "processGroup");
        this.trainer = new FSDPTrainer(module, pg, shardingStrategy, reshardAfterForward, useFullPrecision);
        return trainer;
    }
}
