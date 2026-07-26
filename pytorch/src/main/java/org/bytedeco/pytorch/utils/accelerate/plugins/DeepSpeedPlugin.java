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
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.utils.deepspeed.DeepSpeed;
import org.bytedeco.pytorch.utils.deepspeed.DeepSpeedConfig;
import org.bytedeco.pytorch.utils.deepspeed.DeepSpeedEngine;

import java.util.Map;
import java.util.Objects;

/**
 * HF Accelerate {@code DeepSpeedPlugin} equivalent — holds a {@link DeepSpeedConfig}
 * and can wrap model+optimizer into a {@link DeepSpeedEngine}.
 */
public final class DeepSpeedPlugin {

    private final DeepSpeedConfig config;
    private DeepSpeedEngine engine;

    public DeepSpeedPlugin(DeepSpeedConfig config) {
        this.config = config == null ? DeepSpeedConfig.defaults() : config;
    }

    public DeepSpeedPlugin(Map<String, Object> configMap) {
        this(DeepSpeedConfig.fromMap(configMap));
    }

    public static DeepSpeedPlugin zero2() {
        return new DeepSpeedPlugin(DeepSpeedConfig.builder().zeroStage(2).build());
    }

    public static DeepSpeedPlugin zero3() {
        return new DeepSpeedPlugin(DeepSpeedConfig.builder().zeroStage(3).build());
    }

    public DeepSpeedConfig config() { return config; }
    public DeepSpeedEngine engine() { return engine; }
    public boolean isInitialized() { return engine != null; }

    public DeepSpeedEngine initialize(Module model, Optimizer optimizer, ProcessGroupWrapper pg) {
        Objects.requireNonNull(model, "model");
        this.engine = DeepSpeed.initialize(model, optimizer, config, pg);
        return engine;
    }

    public DeepSpeedEngine initialize(Module model, Optimizer optimizer) {
        return initialize(model, optimizer, null);
    }
}
