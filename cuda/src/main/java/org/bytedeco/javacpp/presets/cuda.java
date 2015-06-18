/*
 * Copyright (C) 2015 Samuel Audet
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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(value = {
    @Platform(include = {"<host_defines.h>", "<device_types.h>", "<driver_types.h>", "<surface_types.h>", "<texture_types.h>",
                         "<vector_types.h>", "<builtin_types.h>", "<cuda_runtime_api.h>", "<driver_functions.h>", "<vector_functions.h>",
                       /*"<cuda_device_runtime_api.h>", <cuda_runtime.h>"*/ "<cuComplex.h>"}, includepath = "/usr/local/cuda/include/",
              link = {"cudart@.6.5", "cuda@.6.5"}, linkpath = "/usr/local/cuda/lib/"),
    @Platform(value = "linux-x86_64", linkpath = "/usr/local/cuda/lib64/"),
    @Platform(value = "windows", includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/Include/",
                                 linkpath    = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/Lib/",
                                 preloadpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/Bin/") },
        target = "org.bytedeco.javacpp.cuda")
public class cuda implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__volatile__", "__no_return__", "__noinline__", "__forceinline__", "__thread__", "__restrict__",
                             "__inline__", "__specialization_static", "__host__", "__device__", "__global__", "__shared__",
                             "__constant__", "__managed__", "NV_CLANG_ATOMIC_NOEXCEPT", "cudaDevicePropDontCare",
                             "CUDART_DEVICE", "CUDART_CB", "__VECTOR_FUNCTIONS_DECL__").cppTypes().annotations().cppText(""))
               .put(new Info("defined(__CUDABE__) || !defined(__CUDACC__)").define())
               .put(new Info("!defined(__CUDACC__) && !defined(__CUDABE__) &&"
                       + "    defined(_WIN32) && !defined(_WIN64)",
                             "!defined(__CUDACC__) && !defined(__CUDABE__) && defined(__arm__) &&"
                       + "    defined(__ARM_PCS_VFP) && __GNUC__ == 4 && __GNUC_MINOR__ == 6",
                             "!defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDABE__) &&"
                       + "    defined(_WIN32) && !defined(_WIN64)", "defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM)").define(false))
               .put(new Info("cudaStreamLegacy", "cudaStreamPerThread").translate(false).cppTypes("cudaStream*"))
               .put(new Info("cudaArray_t", "cudaArray_const_t").valueTypes("cudaArray").pointerTypes("@ByPtrPtr cudaArray"))
               .put(new Info("cudaMipmappedArray_t", "cudaMipmappedArray_const_t").valueTypes("cudaMipmappedArray").pointerTypes("@ByPtrPtr cudaMipmappedArray"))
               .put(new Info("cudaGraphicsResource_t").valueTypes("cudaGraphicsResource").pointerTypes("@ByPtrPtr cudaGraphicsResource"))
               .put(new Info("CUstream_st").pointerTypes("cudaStream"))
               .put(new Info("CUevent_st").pointerTypes("cudaEvent"))
               .put(new Info("cudaStream_t").valueTypes("cudaStream").pointerTypes("@ByPtrPtr cudaStream"))
               .put(new Info("cudaEvent_t").valueTypes("cudaEvent").pointerTypes("@ByPtrPtr cudaEvent"));
    }
}
