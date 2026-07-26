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
package org.bytedeco.pytorch.llm.trl;

import java.util.Map;

/**
 * Optional hooks for TRL-style trainers (mirrors HF {@code TrainerCallback} subset).
 */
public interface TrainerCallback {
    /** Called once before the training loop starts. */
    default void onTrainBegin(BaseTrainer trainer) {}

    /** Called once after the training loop ends. */
    default void onTrainEnd(BaseTrainer trainer) {}

    /**
     * Called after each optimizer step.
     *
     * @param metrics step metrics (loss, lr, …); may be empty
     */
    default void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {}

    /** Called when {@code loggingSteps} fires. */
    default void onLog(BaseTrainer trainer, int step, Map<String, Double> metrics) {}
}
