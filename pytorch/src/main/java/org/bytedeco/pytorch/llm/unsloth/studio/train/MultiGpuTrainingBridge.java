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

package org.bytedeco.pytorch.llm.unsloth.studio.train;

import org.bytedeco.pytorch.llm.unsloth.studio.hardware.DeviceProbe;
import org.bytedeco.pytorch.llm.unsloth.studio.model.HardwareProfile;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Bridge toward accelerate / deepspeed multi-GPU. Reports plan; actual distributed
 * launch is host-owned (ByteDance/Taobao/Tencent meshes typically wrap this).
 */
public final class MultiGpuTrainingBridge {

    public Map<String, Object> plan(List<Integer> gpuIds, String strategy) {
        HardwareProfile hw = DeviceProbe.probe();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("strategy", strategy != null ? strategy : "ddp");
        m.put("requested_gpus", gpuIds);
        m.put("detected_gpus", hw.gpus().size());
        m.put("cuda_available", hw.cudaAvailable());
        m.put("recommended_device", hw.recommendedDevice());
        boolean multi = gpuIds != null && gpuIds.size() > 1 || hw.gpus().size() > 1;
        m.put("multi_gpu", multi);
        if (multi) {
            m.put("accelerate", "org.bytedeco.pytorch.llm.accelerate.Accelerator");
            m.put("deepspeed", "org.bytedeco.pytorch.llm.deepspeed.DeepSpeedEngine");
            m.put("note", "Attach Accelerator/DeepSpeed plugins in host launcher");
        }
        return m;
    }
}
