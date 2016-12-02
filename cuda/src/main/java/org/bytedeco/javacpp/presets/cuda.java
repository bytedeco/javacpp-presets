/*
 * Copyright (C) 2015-2016 Samuel Audet
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
@Properties(names = {"linux-x86_64", "linux-arm64", "linux-ppc64le", "macosx-x86_64", "windows-x86_64"}, value = {
    @Platform(include = {"<cuda.h>", "<host_defines.h>", "<device_types.h>", "<driver_types.h>", "<surface_types.h>", "<texture_types.h>",
                         "<vector_types.h>", "<builtin_types.h>", "<cuda_runtime_api.h>", "<driver_functions.h>", "<vector_functions.h>",
                       /*"<cuda_device_runtime_api.h>", <cuda_runtime.h>"*/ "<cuComplex.h>", "<cuda_fp16.h>", "<library_types.h>"},
              includepath = "/usr/local/cuda-8.0/include/", link = {"cudart@.8.0", "cuda@.8.0"}, linkpath = "/usr/local/cuda-8.0/lib/"),
    @Platform(value = {"linux-x86_64", "linux-ppc64le"}, linkpath = "/usr/local/cuda-8.0/lib64/"),
    @Platform(value = "macosx-x86_64",  includepath =  "/Developer/NVIDIA/CUDA-8.0/include/",
                                           linkpath = {"/Developer/NVIDIA/CUDA-8.0/lib/", "/usr/local/cuda/lib/"}),
    @Platform(value = "windows-x86_64", includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include/",
                                        preloadpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/bin/",
                                           linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64/") },
        target = "org.bytedeco.javacpp.cuda")
public class cuda implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__volatile__", "__no_return__", "__noinline__", "__forceinline__", "__thread__", "__restrict__",
                             "__inline__", "__specialization_static", "__host__", "__device__", "__global__", "__shared__",
                             "__constant__", "__managed__", "NV_CLANG_ATOMIC_NOEXCEPT", "cudaDevicePropDontCare", "__LDG_PTR",
                             "CUDA_CB", "CUDAAPI", "CUDART_DEVICE", "CUDART_CB", "__VECTOR_FUNCTIONS_DECL__").cppTypes().annotations().cppText(""))
               .put(new Info("defined(__CUDABE__) || !defined(__CUDACC__)").define())
               .put(new Info("defined(CUDA_FORCE_API_VERSION)",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 3020",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 4000",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 4010",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 6050",
                             "defined(__CUDA_API_VERSION) && __CUDA_API_VERSION >= 3020 && __CUDA_API_VERSION < 4010",
                             "defined(__CUDA_API_VERSION_INTERNAL)", "defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION < 3020",
                             "defined(__CUDA_API_VERSION_INTERNAL) || (__CUDA_API_VERSION >= 3020 && __CUDA_API_VERSION < 4010)",
                             "!defined(__CUDACC__) && !defined(__CUDABE__) &&"
                       + "    defined(_WIN32) && !defined(_WIN64)",
                             "!defined(__CUDACC__) && !defined(__CUDABE__) && defined(__arm__) &&"
                       + "    defined(__ARM_PCS_VFP) && __GNUC__ == 4 && __GNUC_MINOR__ == 6",
                             "!defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDABE__) &&"
                       + "    defined(_WIN32) && !defined(_WIN64)", "defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM)").define(false))
               .put(new Info("__CUDA_FP16_DECL__", "__float_simpl_sinf(float)", "__float_simpl_cosf(float)",
                             "__internal_trig_reduction_kernel", "__internal_sin_cos_kernel", "cuDeviceGetP2PAttribute",
                             "cuMemRangeGetAttribute", "cuMemRangeGetAttributes").skip())
               .put(new Info("CUcontext").valueTypes("CUctx_st").pointerTypes("@ByPtrPtr CUctx_st"))
               .put(new Info("CUmodule").valueTypes("CUmod_st").pointerTypes("@ByPtrPtr CUmod_st"))
               .put(new Info("CUfunction").valueTypes("CUfunc_st").pointerTypes("@ByPtrPtr CUfunc_st"))
               .put(new Info("CUarray").valueTypes("CUarray_st").pointerTypes("@ByPtrPtr CUarray_st"))
               .put(new Info("CUmipmappedArray").valueTypes("CUmipmappedArray_st").pointerTypes("@ByPtrPtr CUmipmappedArray_st"))
               .put(new Info("CUtexref").valueTypes("CUtexref_st").pointerTypes("@ByPtrPtr CUtexref_st"))
               .put(new Info("CUsurfref").valueTypes("CUsurfref_st").pointerTypes("@ByPtrPtr CUsurfref_st"))
               .put(new Info("CUevent").valueTypes("CUevent_st").pointerTypes("@ByPtrPtr CUevent_st"))
               .put(new Info("CUstream").valueTypes("CUstream_st").pointerTypes("@ByPtrPtr CUstream_st"))
               .put(new Info("CUgraphicsResource").valueTypes("CUgraphicsResource_st").pointerTypes("@ByPtrPtr CUgraphicsResource_st"))
               .put(new Info("CUlinkState").valueTypes("CUlinkState_st").pointerTypes("@ByPtrPtr CUlinkState_st"))
               .put(new Info("CU_LAUNCH_PARAM_END", "CU_LAUNCH_PARAM_BUFFER_POINTER", "CU_LAUNCH_PARAM_BUFFER_SIZE").translate(false).cppTypes("void*"))
               .put(new Info("CU_DEVICE_CPU", "CU_DEVICE_INVALID").translate(false).cppTypes("int"))
               .put(new Info("CU_STREAM_LEGACY", "CU_STREAM_PER_THREAD", "cudaStreamLegacy", "cudaStreamPerThread").translate(false).cppTypes("CUstream_st*"))
               .put(new Info("cudaArray_t", "cudaArray_const_t").valueTypes("cudaArray").pointerTypes("@ByPtrPtr cudaArray"))
               .put(new Info("cudaMipmappedArray_t", "cudaMipmappedArray_const_t").valueTypes("cudaMipmappedArray").pointerTypes("@ByPtrPtr cudaMipmappedArray"))
               .put(new Info("cudaGraphicsResource_t").valueTypes("cudaGraphicsResource").pointerTypes("@ByPtrPtr cudaGraphicsResource"))
               .put(new Info("cudaStream_t").valueTypes("CUstream_st").pointerTypes("@ByPtrPtr CUstream_st"))
               .put(new Info("cudaEvent_t").valueTypes("CUevent_st").pointerTypes("@ByPtrPtr CUevent_st"));
    }
}
