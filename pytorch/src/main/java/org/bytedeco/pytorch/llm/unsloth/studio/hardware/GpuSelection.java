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

package org.bytedeco.pytorch.llm.unsloth.studio.hardware;

import org.bytedeco.pytorch.llm.unsloth.studio.model.HardwareProfile;

import java.util.ArrayList;
import java.util.List;

/** Picks a GPU subset that fits an estimated VRAM budget. */
public final class GpuSelection {

    private GpuSelection() {}

    public static List<Integer> autoSelect(HardwareProfile profile, long requiredMb) {
        List<Integer> selected = new ArrayList<>();
        if (profile == null || profile.gpus().isEmpty()) {
            return selected;
        }
        long acc = 0;
        for (HardwareProfile.GpuDevice g : profile.gpus()) {
            selected.add(g.index());
            long cap = g.totalMemoryMb() > 0 ? g.totalMemoryMb() : 8192;
            acc += cap;
            if (requiredMb <= 0 || acc >= requiredMb) break;
        }
        return selected;
    }
}
