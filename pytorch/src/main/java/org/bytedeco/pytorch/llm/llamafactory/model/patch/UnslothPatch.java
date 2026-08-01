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
package org.bytedeco.pytorch.llm.llamafactory.model.patch;

import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.nn.Module;

import java.util.logging.Logger;

/**
 * Delegates to {@code org.bytedeco.pytorch.llm.unsloth.FastLanguageModel} when
 * present. Does not replace the already-constructed module graph in-place
 * (that would drop host references); instead marks the run for Unsloth-style
 * trainer selection in {@code TrainerFactory}.
 */
public final class UnslothPatch {

    private static final Logger LOG = Logger.getLogger(UnslothPatch.class.getName());

    private UnslothPatch() {}

    public static boolean apply(Module model, ModelArgs args) {
        try {
            Class.forName("org.bytedeco.pytorch.llm.unsloth.FastLanguageModel");
            System.setProperty("jnitorch.llm.use_unsloth", "true");
            if (args != null && args.modelNameOrPath() != null) {
                System.setProperty("jnitorch.llm.unsloth.model", args.modelNameOrPath());
            }
            LOG.info("UnslothPatch: FastLanguageModel available — trainer may use unsloth path");
            return true;
        } catch (ClassNotFoundException e) {
            LOG.fine("Unsloth not on classpath; flag ignored");
            return false;
        }
    }

    public static boolean isEnabled() {
        return "true".equalsIgnoreCase(System.getProperty("jnitorch.llm.use_unsloth", "false"));
    }
}
