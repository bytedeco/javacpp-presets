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
package org.bytedeco.pytorch.llm.llamafactory.train;

import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.PPOTrainer;
import org.bytedeco.pytorch.llm.trl.config.PPOConfig;
import org.bytedeco.pytorch.optim.Optimizer;

/**
 * PPO stage bridge → {@link PPOTrainer}.
 *
 * <p>Expects host-supplied precomputed batch keys ({@code old_logprobs},
 * {@code advantages}, {@code returns}, …) as documented on {@link PPOTrainer}.
 * Online PolicyValueForward can be attached by constructing PPOTrainer directly.
 */
public final class PpoTrainerBridge {
    private PpoTrainerBridge() {}

    public static PPOTrainer create(FactoryArgs args, LoadedModel loaded, int maxSteps) {
        BaseTrainer t = TrainerFactory.create(args, loaded, maxSteps);
        if (t instanceof PPOTrainer ppo) {
            return ppo;
        }
        Optimizer opt = TrainerFactory.buildOptimizer(args, loaded);
        PPOConfig cfg = TrainerFactory.ppoConfig(args, maxSteps);
        return new PPOTrainer(loaded.module(), opt, cfg);
    }
}
