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
package org.bytedeco.pytorch.llm.deepspeed;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.llm.deepspeed.inference.InferenceEngine;

import java.util.Map;

/**
 * Microsoft DeepSpeed-style training / inference facade (Java port).
 *
 * <p>Implements ZeRO stage semantics, gradient accumulation, checkpointing and
 * an inference engine at the API / collective level on top of libtorch and
 * {@link ProcessGroupWrapper}. Native DeepSpeed CUDA ops are <em>not</em>
 * reimplemented — fusion flags are configuration bookkeeping only.
 *
 * <pre>{@code
 * DeepSpeedConfig cfg = DeepSpeedConfig.builder().zeroStage(2).cpuOffload(true).build();
 * DeepSpeedEngine engine = DeepSpeed.initialize(model, optimizer, cfg);
 * engine.backward(loss);
 * engine.step();
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DeepSpeed {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";

    private DeepSpeed() {}

    public static String version() { return VERSION; }

    public static DeepSpeedEngine initialize(Module model, Optimizer optimizer, DeepSpeedConfig config) {
        return new DeepSpeedEngine(model, optimizer, config, null);
    }

    public static DeepSpeedEngine initialize(Module model, Optimizer optimizer,
                                             DeepSpeedConfig config, ProcessGroupWrapper pg) {
        return new DeepSpeedEngine(model, optimizer, config, pg);
    }

    public static DeepSpeedEngine initialize(Module model, Optimizer optimizer, Map<String, Object> configMap) {
        return initialize(model, optimizer, DeepSpeedConfig.fromMap(configMap), null);
    }

    public static DeepSpeedEngine initialize(Module model, Optimizer optimizer,
                                             Map<String, Object> configMap, ProcessGroupWrapper pg) {
        return initialize(model, optimizer, DeepSpeedConfig.fromMap(configMap), pg);
    }

    public static InferenceEngine initInference(Module model) {
        return InferenceEngine.create(model);
    }

    public static InferenceEngine initInference(Module model, InferenceEngine.InferenceConfig config) {
        return InferenceEngine.create(model, config);
    }

    public static InferenceEngine init_inference(Module model) {
        return initInference(model);
    }
}
