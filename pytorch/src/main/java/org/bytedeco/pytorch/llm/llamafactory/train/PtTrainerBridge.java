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
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.optim.Optimizer;

/**
 * PT (continual pre-train) stage bridge — reuses {@link SFTTrainer} with
 * pretrain-collated batches ({@code input_ids}/{@code labels} full-sequence CE).
 */
public final class PtTrainerBridge {

    private PtTrainerBridge() {}

    public static SFTTrainer create(FactoryArgs args, LoadedModel loaded, int maxSteps) {
        BaseTrainer t = TrainerFactory.create(
                args.withFinetuning(args.finetuning()), // stage already PT
                loaded,
                maxSteps);
        if (t instanceof SFTTrainer sft) {
            return sft;
        }
        // Fallback construct
        Optimizer opt = TrainerFactory.buildOptimizer(args, loaded);
        SFTConfig cfg = TrainerFactory.sftConfig(args, maxSteps);
        return new SFTTrainer(loaded.module(), TrainerFactory.causalForward(loaded), opt, cfg);
    }
}
