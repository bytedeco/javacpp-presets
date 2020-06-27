/*
 * Copyright (C) 2018-2020 Samuel Audet
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

package org.bytedeco.cuda.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = cudart.class, value = {
    @Platform(value = {"linux-x86_64", "linux-ppc64le", "windows-x86_64"}, define = "NVML_NO_UNVERSIONED_FUNC_DEFS", include = "<nvml.h>", link = "nvidia-ml@.1#"),
    @Platform(value = "windows-x86_64", link = "nvml", preloadpath = "C:/Program Files/NVIDIA Corporation/NVSMI/")},
        target = "org.bytedeco.cuda.nvml", global = "org.bytedeco.cuda.global.nvml")
@NoException
public class nvml implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("DECLDIR", "nvmlInit", "nvmlDeviceGetPciInfo", "nvmlDeviceGetCount", "nvmlDeviceGetHandleByIndex",
                             "nvmlDeviceGetHandleByPciBusId", "nvmlDeviceGetNvLinkRemotePciInfo", "nvmlDeviceRemoveGpu",
                             "nvmlDeviceGetGridLicensableFeatures", "nvmlEccBitType_t").cppTypes().annotations())
               .put(new Info("NVML_NO_UNVERSIONED_FUNC_DEFS").define(true))
               .put(new Info("NVML_SINGLE_BIT_ECC", "NVML_DOUBLE_BIT_ECC").translate(false))
               .put(new Info("NVML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION", "NVML_VGPU_VIRTUALIZATION_CAP_MIGRATION").skip(true))
               .put(new Info("nvmlDevice_t").valueTypes("nvmlDevice_st").pointerTypes("@ByPtrPtr nvmlDevice_st", "@Cast(\"nvmlDevice_st**\") PointerPointer"))
               .put(new Info("nvmlUnit_t").valueTypes("nvmlUnit_st").pointerTypes("@ByPtrPtr nvmlUnit_st", "@Cast(\"nvmlUnit_st**\") PointerPointer"))
               .put(new Info("nvmlEventSet_t").valueTypes("nvmlEventSet_st").pointerTypes("@ByPtrPtr nvmlEventSet_st", "@Cast(\"nvmlEventSet_st**\") PointerPointer"))
               .put(new Info("nvmlGpuInstance_t").valueTypes("nvmlGpuInstance_st").pointerTypes("@ByPtrPtr nvmlGpuInstance_st", "@Cast(\"nvmlGpuInstance_st**\") PointerPointer"))
               .put(new Info("nvmlComputeInstance_t").valueTypes("nvmlComputeInstance_st").pointerTypes("@ByPtrPtr nvmlComputeInstance_st", "@Cast(\"nvmlComputeInstance_st**\") PointerPointer"));
    }
}
