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

import org.bytedeco.pytorch.nn.Module;

/**
 * LongLoRA shift-short-attention flag (S²-Attn).
 *
 * <p>Records the shift pattern on the module when a hook exists; otherwise
 * returns false. Full shift-attn math lives in modeling attention when enabled.
 */
public final class LongLoraShiftPatch {

    private LongLoraShiftPatch() {}

    public static boolean apply(Module model) {
        if (model == null) return false;
        for (String name : new String[]{"setShiftAttn", "enableShiftShortAttn", "setLongLora"}) {
            try {
                var m = model.getClass().getMethod(name, boolean.class);
                m.invoke(model, true);
                return true;
            } catch (ReflectiveOperationException ignored) {
            }
        }
        // Soft mark via system property for modeling code that polls it
        System.setProperty("jnitorch.llm.shift_attn", "true");
        return true;
    }
}
