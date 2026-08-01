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

import java.util.logging.Logger;

/**
 * KTransformers expert-offload hooks (documented stub / real).
 *
 * <p>When a KTransformers integration class is present it is notified; otherwise
 * the flag is stored for MoE modeling to optionally offload experts to CPU /
 * disk. Fidelity: not claiming full KTransformers kernel parity.
 */
public final class KTransformersPatch {

    private static final Logger LOG = Logger.getLogger(KTransformersPatch.class.getName());

    private KTransformersPatch() {}

    public static boolean apply(Module model) {
        System.setProperty("jnitorch.llm.use_ktransformers", "true");
        try {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.llm.modules.KTransformersMoE");
            if (model != null) {
                try {
                    var m = cls.getMethod("attach", Module.class);
                    m.invoke(null, model);
                    LOG.info("KTransformersPatch: attached KTransformersMoE");
                    return true;
                } catch (ReflectiveOperationException e) {
                    LOG.fine("KTransformersMoE.attach missing: " + e.getMessage());
                }
            }
            return true;
        } catch (ClassNotFoundException e) {
            LOG.info("KTransformersPatch: no native integration — flag only (expert offload deferred)");
            return true; // flag recorded; MoE path may still read the property
        }
    }
}
