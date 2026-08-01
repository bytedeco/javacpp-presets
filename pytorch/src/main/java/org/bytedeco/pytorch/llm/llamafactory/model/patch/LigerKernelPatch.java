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
 * Liger Kernel fused MLP / RMSNorm capability flag.
 *
 * <p>When fused ops exist under {@code llm.modules} they may be selected by
 * modeling code that reads {@code jnitorch.llm.use_liger=true}. Pure-Java
 * reference path keeps standard Linear+SiLU+RMSNorm.
 */
public final class LigerKernelPatch {

    private static final Logger LOG = Logger.getLogger(LigerKernelPatch.class.getName());

    private LigerKernelPatch() {}

    public static boolean apply(Module model) {
        System.setProperty("jnitorch.llm.use_liger", "true");
        boolean fused = false;
        for (String cls : new String[]{
                "org.bytedeco.pytorch.llm.modules.FusedMLP",
                "org.bytedeco.pytorch.llm.modules.LigerRMSNorm",
                "org.bytedeco.pytorch.llm.modules.SwiGLU"
        }) {
            try {
                Class.forName(cls);
                fused = true;
                break;
            } catch (ClassNotFoundException ignored) {
            }
        }
        if (model != null) {
            for (String name : new String[]{"setUseLiger", "enableLigerKernel"}) {
                try {
                    var m = model.getClass().getMethod(name, boolean.class);
                    m.invoke(model, true);
                    fused = true;
                    break;
                } catch (ReflectiveOperationException ignored) {
                }
            }
        }
        LOG.info("LigerKernelPatch: fused_ops_present=" + fused);
        return true; // flag recorded either way
    }
}
