/*
 * Copyright (C) 2019-2020 Samuel Audet
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

package org.bytedeco.cpu_features.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            include = {
                "cpu_features/cpu_features_macros.h",
                "cpu_features/cpu_features_cache_info.h",
                "cpu_features/internal/hwcaps.h",
                "cpu_features/cpuinfo_aarch64.h",
                "cpu_features/cpuinfo_arm.h",
                "cpu_features/cpuinfo_mips.h",
                "cpu_features/cpuinfo_ppc.h",
                "cpu_features/cpuinfo_x86.h",
            },
            link = "cpu_features",
            resource = {"include", "lib"}
        ),
    },
    target = "org.bytedeco.cpu_features",
    global = "org.bytedeco.cpu_features.global.cpu_features"
)
@NoException
public class cpu_features implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "cpu_features"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("defined(__cplusplus)").define())
               .put(new Info("CPU_FEATURES_START_CPP_NAMESPACE", "CPU_FEATURES_END_CPP_NAMESPACE",
                             "CPU_FEATURES_COMPILED_X86_AES", "CPU_FEATURES_COMPILED_X86_F16C",
                             "CPU_FEATURES_COMPILED_X86_BMI", "CPU_FEATURES_COMPILED_X86_BMI2",
                             "CPU_FEATURES_COMPILED_X86_SSE", "CPU_FEATURES_COMPILED_X86_SSE2",
                             "CPU_FEATURES_COMPILED_X86_SSE3", "CPU_FEATURES_COMPILED_X86_SSSE3",
                             "CPU_FEATURES_COMPILED_X86_SSE4_1", "CPU_FEATURES_COMPILED_X86_SSE4_2",
                             "CPU_FEATURES_COMPILED_X86_AVX", "CPU_FEATURES_COMPILED_x86_AVX2",
                             "CPU_FEATURES_COMPILED_ANY_ARM_NEON", "CPU_FEATURES_COMPILED_MIPS_MSA").cppTypes().annotations())
               .put(new Info("cpu_features::CpuFeatures_IsHwCapsSet").skip());
    }
}
