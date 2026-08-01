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

package org.bytedeco.pytorch.llm.unsloth.studio.observe;

import java.util.LinkedHashMap;
import java.util.Map;

/** Best-effort GPU utilization sampling via nvidia-smi. */
public final class GpuMetricsSampler {

    public Map<String, Double> sample() {
        Map<String, Double> m = new LinkedHashMap<>();
        try {
            Process p = new ProcessBuilder("nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits")
                    .redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes());
            p.waitFor();
            String[] lines = out.trim().split("\\R");
            int i = 0;
            for (String line : lines) {
                if (line.isBlank()) continue;
                String[] parts = line.split(",");
                if (parts.length >= 4) {
                    m.put("gpu" + i + "_mem_used_mb", Double.parseDouble(parts[1].trim()));
                    m.put("gpu" + i + "_mem_total_mb", Double.parseDouble(parts[2].trim()));
                    m.put("gpu" + i + "_util", Double.parseDouble(parts[3].trim()));
                    i++;
                }
            }
        } catch (Throwable ignored) {
            Runtime rt = Runtime.getRuntime();
            m.put("jvm_used_mb", (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0));
            m.put("jvm_max_mb", rt.maxMemory() / (1024.0 * 1024.0));
        }
        return m;
    }
}
