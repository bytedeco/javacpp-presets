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
import org.bytedeco.pytorch.llm.llamafactory.data.collator.KtoCollator;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;

import java.util.logging.Logger;

/**
 * KTO stage bridge.
 *
 * <p>When {@code llm.trl.KTOTrainer} is present on the classpath it is preferred;
 * otherwise the factory falls back to {@link SFTTrainer} while still using
 * {@link KtoCollator} features so
 * hosts can swap in a real KTO loss later without changing the data plane.
 *
 * <p>Plan P2 adds {@code KTOTrainer}/{@code KTOLoss}/{@code KTOConfig} under
 * {@code llm.trl}; this bridge is the stable factory entry point.
 */
public final class KtoTrainerBridge {
    private static final Logger LOG = Logger.getLogger(KtoTrainerBridge.class.getName());

    private KtoTrainerBridge() {}

    public static BaseTrainer create(FactoryArgs args, LoadedModel loaded, int maxSteps) {
        // Reflective prefer real KTO when merged into trl
        try {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.llm.trl.KTOTrainer");
            // Prefer TrainerFactory path which already documents the SFT fallback
            BaseTrainer t = TrainerFactory.create(args, loaded, maxSteps);
            if (cls.isInstance(t)) {
                return t;
            }
        } catch (ClassNotFoundException ignored) {
            LOG.info("KTOTrainer not on classpath — SFT bridge; collator still emits kto tags");
        }
        BaseTrainer t = TrainerFactory.create(args, loaded, maxSteps);
        if (t instanceof SFTTrainer) {
            return t;
        }
        return SftTrainerBridge.create(args, loaded, maxSteps);
    }
}
