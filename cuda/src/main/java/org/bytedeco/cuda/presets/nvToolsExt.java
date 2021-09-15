/*
 * Copyright (C) 2021 Samuel Audet
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
    @Platform(include = {"<nvToolsExt.h>", "<nvToolsExtCuda.h>", "<nvToolsExtCudaRt.h>"/*, "<nvToolsExtOpenCL.h>"*/},
              link = "nvToolsExt@.1"),
    @Platform(value = "windows-x86_64", link = "nvToolsExt64_1",
              includepath = "C:/Program Files/NVIDIA Corporation/NvToolsExt/include/",
              linkpath    = "C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/",
              preloadpath = "C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/")},
        target = "org.bytedeco.cuda.nvToolsExt", global = "org.bytedeco.cuda.global.nvToolsExt")
@NoException
public class nvToolsExt implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("NVTX_DECLSPEC", "NVTX_API", "NVTX_INLINE_STATIC").cppTypes().annotations())
               .put(new Info("nvToolsExt.h", "nvToolsExtCuda.h", "nvToolsExtCudaRt.h").linePatterns("#ifdef UNICODE", "#endif").skip())
               .put(new Info("nvtxStringHandle", "nvtxDomainHandle", "nvtxResourceHandle").skip())
               .put(new Info("nvtxStringHandle_t", "nvtxDomainHandle_t", "nvtxResourceHandle_t").cast().valueTypes("Pointer"))
               .put(new Info("nvtxInitializationAttributes_v2", "nvtxMessageValue_t", "nvtxResourceAttributes_v0",
                             "nvtxInitializationAttributes_t", "nvtxResourceAttributes_t").cast().pointerTypes("Pointer"))
               .put(new Info("nvtxEventAttributes_v1", "nvtxEventAttributes_v2").pointerTypes("nvtxEventAttributes_t"))
               .put(new Info("NVTX_RESOURCE_TYPE_GENERIC_POINTER", "NVTX_RESOURCE_TYPE_GENERIC_HANDLE",
                             "NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE", "NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX",
                             "NVTX_RESOURCE_TYPE_CUDA_DEVICE", "NVTX_RESOURCE_TYPE_CUDA_CONTEXT",
                             "NVTX_RESOURCE_TYPE_CUDA_STREAM", "NVTX_RESOURCE_TYPE_CUDA_EVENT",
                             "NVTX_RESOURCE_TYPE_CUDART_DEVICE", "NVTX_RESOURCE_TYPE_CUDART_STREAM",
                             "NVTX_RESOURCE_TYPE_CUDART_EVENT").translate(false))
        ;
    }
}
