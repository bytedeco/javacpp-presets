/*
 * Copyright (C) 2015-2025 Samuel Audet
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
@Properties(inherit = javacpp.class, names = {"linux-x86_64", "linux-arm64", "linux-ppc64le", "macosx-x86_64", "windows-x86_64"}, value = {
    @Platform(include = {"<cuda.h>", "<crt/host_defines.h>", "<device_types.h>", "<driver_types.h>", "<surface_types.h>", "<texture_types.h>",
                         "<vector_types.h>", "<builtin_types.h>", "<cuda_runtime_api.h>", "<driver_functions.h>", "<vector_functions.h>",
                       /*"<cuda_device_runtime_api.h>", <cuda_runtime.h>"*/ "<cuComplex.h>", "<cuda_fp16.h>", "<cuda_fp16.hpp>",
                         "<cuda_bf16.h>", "<cuda_bf16.hpp>", "<library_types.h>", "<cudaGL.h>", "<cuda_gl_interop.h>"},
              compiler = "cpp11", exclude = "<crt/host_defines.h>",
              includepath = {"/usr/local/cuda-13.1/include/", "/usr/local/cuda/include/", "/usr/include/"},
              link = {"cudart@.13", "cuda@.1#"}, linkpath = {"/usr/local/cuda-13.1/lib/", "/usr/local/cuda/lib/", "/usr/lib/"}),
    @Platform(value = {"linux-x86_64", "linux-arm64", "linux-ppc64le"}, linkpath = {"/usr/local/cuda-13.1/lib64/", "/usr/local/cuda/lib64/", "/usr/lib64/"}),
    @Platform(value = "macosx-x86_64",  includepath =  "/Developer/NVIDIA/CUDA-13.1/include/",
                                           linkpath = {"/Developer/NVIDIA/CUDA-13.1/lib/", "/usr/local/cuda/lib/"}),
    @Platform(value = "windows-x86_64",     preload = "cudart64_13",
                                        includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include/",
                                        preloadpath = {"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/",
                                                       "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64/"},
                                           linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64/") },
        target = "org.bytedeco.cuda.cudart", global = "org.bytedeco.cuda.global.cudart")
@NoException
public class cudart implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "cuda"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__volatile__", "__no_return__", "__noinline__", "__forceinline__", "__thread__", "__restrict__", "__grid_constant__",
                             "__inline__", "__inline_hint__", "__specialization_static", "__host__", "__device__", "__global__", "__shared__",
                             "__constant__", "__managed__", "__nv_pure__", "NV_CLANG_ATOMIC_NOEXCEPT", "cudaDevicePropDontCare", "__LDG_PTR", "__CUDA_ALIGN__",
                             "CUDA_CB", "CUDAAPI", "CUDART_DEVICE", "CUDART_CB", "__VECTOR_FUNCTIONS_DECL__", "__CUDA_HOSTDEVICE__",
                             "__CUDA_HOSTDEVICE_FP16_DECL__", "__CUDA_HOSTDEVICE_BF16_DECL__", "__CUDA_FP16_CONSTEXPR__", "__CUDA_BF16_CONSTEXPR__",
                             "__tile_global__", "__tile__", "__tile_builtin__", "__NV_SILENCE_HOST_DEPRECATION_BEGIN", "__NV_SILENCE_HOST_DEPRECATION_END").cppTypes().annotations().cppText(""))

//               .put(new Info("cuda_runtime_api.h").linePatterns("#define cudaSignalExternalSemaphoresAsync.*", "#define cudaGetDeviceProperties.*").skip())
               .put(new Info("cuda.h").linePatterns("#define cuDeviceTotalMem.*", "#define cuStreamUpdateCaptureDependencies.*", "#define cuMemBatchDecompressAsync.*").skip())
               .put(new Info("cudaGL.h").linePatterns("#define cuGLCtxCreate.*", "#define cuGLGetDevices.*").skip())

               .put(new Info("_CONCAT_OUTER").cppText("#define _CONCAT_OUTER(x, y) CUdevResource_st").cppTypes())
               .put(new Info("__CUDA_DEPRECATED").cppText("#define __CUDA_DEPRECATED deprecated").cppTypes())
               .put(new Info("CUDNN_DEPRECATED").cppText("#define CUDNN_DEPRECATED deprecated").cppTypes())
               .put(new Info("CUDNN_DEPRECATED_ENUM").cppText("#define CUDNN_DEPRECATED_ENUM").cppTypes())
               .put(new Info("CUSPARSE_DEPRECATED").cppText("#define CUSPARSE_DEPRECATED deprecated").cppTypes())
               .put(new Info("CUSPARSE_DEPRECATED_ENUM").cppText("#define CUSPARSE_DEPRECATED_ENUM deprecated").cppTypes())
               .put(new Info("CUSPARSE_DEPRECATED_HINT").cppText("#define CUSPARSE_DEPRECATED_HINT deprecated").cppTypes())
               .put(new Info("CUSOLVER_DEPRECATED").cppText("#define CUSOLVER_DEPRECATED(new_func) deprecated").cppTypes())
               .put(new Info("CUSOLVER_DEPRECATED_").cppText("#define CUSOLVER_DEPRECATED_ deprecated").cppTypes())
               .put(new Info("CUSOLVERMG_DEPRECATED").cppText("#define CUSOLVERMG_DEPRECATED deprecated").cppTypes())
               .put(new Info("NVCOMP_DEPRECATED").cppText("#define NVCOMP_DEPRECATED deprecated").cppTypes())
               .put(new Info("__DEPRECATED__").cppText("#define __DEPRECATED__(msg) deprecated").cppTypes())
               .put(new Info("DEPRECATED").cppText("#define DEPRECATED(ver) deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("cudaStreamAttrID", "cudaKernelNodeAttrID").cppTypes().pointerTypes("cudaLaunchAttributeID"))
               .put(new Info("cudaStreamAttributeAccessPolicyWindow", "cudaKernelNodeAttributeAccessPolicyWindow").cppTypes().pointerTypes("cudaLaunchAttributeAccessPolicyWindow"))
               .put(new Info("cudaStreamAttributeSynchronizationPolicy").cppTypes().pointerTypes("cudaLaunchAttributeSynchronizationPolicy"))
               .put(new Info("cudaStreamAttrValue", "cudaKernelNodeAttrValue").cppTypes().pointerTypes("cudaLaunchAttributeValue"))
               .put(new Info("cudaKernelNodeAttributeCooperative").cppTypes().pointerTypes("cudaLaunchAttributeCooperative"))
               .put(new Info("cudaKernelNodeAttributePriority").cppTypes().pointerTypes("cudaLaunchAttributePriority"))
               .put(new Info("cudaKernelNodeAttributeClusterDimension").cppTypes().pointerTypes("cudaLaunchAttributeClusterDimension"))
               .put(new Info("cudaKernelNodeAttributeClusterSchedulingPolicyPreference").cppTypes().pointerTypes("cudaLaunchAttributeClusterSchedulingPolicyPreference"))

               .put(new Info("defined(__CUDABE__) || !defined(__CUDACC__)").define())
               .put(new Info("defined(__CUDACC__)", "defined(CUDA_FORCE_API_VERSION)",
                             "defined(__CUDACC__) || defined(_NVHPC_CUDA)",
                             "defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))",
                             "defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800) || defined(_NVHPC_CUDA))",
                             "(defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 3020",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 4000",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 4010",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 6050",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION >= 10010",
                             "defined(__CUDA_API_VERSION) && __CUDA_API_VERSION >= 3020 && __CUDA_API_VERSION < 4010",
                             "defined(__CUDA_API_VERSION_INTERNAL)", "defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)",
                             "defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION < 3020",
                             "defined(__CUDA_API_VERSION_INTERNAL) || (__CUDA_API_VERSION >= 3020 && __CUDA_API_VERSION < 4010)",
                             "!defined(__CUDACC__) && !defined(__CUDACC_RTC__) &&"
                       + "    defined(_WIN32) && !defined(_WIN64)",
                             "!defined(__CUDACC__) && !defined(__CUDABE__) &&"
                       + "    defined(_WIN32) && !defined(_WIN64)",
                             "!defined(__CUDACC__) && !defined(__CUDABE__) && defined(__arm__) &&"
                       + "    defined(__ARM_PCS_VFP) && __GNUC__ == 4 && __GNUC_MINOR__ == 6",
                             "!defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDABE__) &&"
                       + "    defined(_WIN32) && !defined(_WIN64)", "defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM)",
                             "!defined(DISABLE_CUSPARSE_DEPRECATED)", "!defined(_WIN32)",
                         "defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) || defined(__CUDA_API_VERSION_INTERNAL)",
                         "defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM)").define(false))
               .put(new Info("__CUDART_API_VERSION").translate(false).cppTypes("int"))
               .put(new Info("__CUDA_FP16_DECL__", "__CUDA_BF16_DECL__", "__float_simpl_sinf(float)", "__float_simpl_cosf(float)",
                             "__internal_trig_reduction_kernel", "__internal_sin_cos_kernel", "cuDeviceGetP2PAttribute",
                             "cuMemRangeGetAttribute", "cuMemRangeGetAttributes", "float2::__cuda_gnu_arm_ice_workaround",
                             "cuDeviceGetLuid", "cuDeviceGetNvSciSyncAttributes", "cudaDeviceGetNvSciSyncAttributes",
                             "cuMemRetainAllocationHandle", "cudaWGLGetDevice", "cuWGLGetDevice").skip())
               .put(new Info("cudaGraphNodeParams", "cudaGraphKernelNodeUpdate").purify())
               .put(new Info("CUcontext").valueTypes("CUctx_st").pointerTypes("@ByPtrPtr CUctx_st"))
               .put(new Info("CUmodule").valueTypes("CUmod_st").pointerTypes("@ByPtrPtr CUmod_st"))
               .put(new Info("CUfunction").valueTypes("CUfunc_st").pointerTypes("@ByPtrPtr CUfunc_st"))
               .put(new Info("CUlibrary").valueTypes("CUlib_st").pointerTypes("@ByPtrPtr CUlib_st"))
               .put(new Info("CUkernel").valueTypes("CUkern_st").pointerTypes("@ByPtrPtr CUkern_st"))
               .put(new Info("CUarray").valueTypes("CUarray_st").pointerTypes("@ByPtrPtr CUarray_st"))
               .put(new Info("CUmipmappedArray").valueTypes("CUmipmappedArray_st").pointerTypes("@ByPtrPtr CUmipmappedArray_st"))
               .put(new Info("CUtexref").valueTypes("CUtexref_st").pointerTypes("@ByPtrPtr CUtexref_st"))
               .put(new Info("CUsurfref").valueTypes("CUsurfref_st").pointerTypes("@ByPtrPtr CUsurfref_st"))
               .put(new Info("CUevent").valueTypes("CUevent_st").pointerTypes("@ByPtrPtr CUevent_st"))
               .put(new Info("CUstream").valueTypes("CUstream_st").pointerTypes("@ByPtrPtr CUstream_st"))
               .put(new Info("CUexternalMemory").valueTypes("CUextMemory_st").pointerTypes("@ByPtrPtr CUextMemory_st"))
               .put(new Info("CUexternalSemaphore").valueTypes("CUextSemaphore_st").pointerTypes("@ByPtrPtr CUextSemaphore_st"))
               .put(new Info("const CUexternalSemaphore").valueTypes("CUextSemaphore_st").pointerTypes("@Cast(\"const CUexternalSemaphore*\") @ByPtrPtr CUextSemaphore_st"))
               .put(new Info("CUgraph").valueTypes("CUgraph_st").pointerTypes("@ByPtrPtr CUgraph_st"))
               .put(new Info("CUgraphNode").valueTypes("CUgraphNode_st").pointerTypes("@ByPtrPtr CUgraphNode_st"))
               .put(new Info("const CUgraphNode").valueTypes("CUgraphNode_st").pointerTypes("@Cast(\"const CUgraphNode*\") @ByPtrPtr CUgraphNode_st"))
               .put(new Info("CUgraphExec").valueTypes("CUgraphExec_st").pointerTypes("@ByPtrPtr CUgraphExec_st"))
               .put(new Info("CUgraphicsResource").valueTypes("CUgraphicsResource_st").pointerTypes("@ByPtrPtr CUgraphicsResource_st"))
               .put(new Info("CUlinkState").valueTypes("CUlinkState_st").pointerTypes("@ByPtrPtr CUlinkState_st"))
               .put(new Info("CUasyncCallbackHandle").valueTypes("CUasyncCallbackEntry_st").pointerTypes("@ByPtrPtr CUasyncCallbackEntry_st"))
               .put(new Info("CUdevResourceDesc", "cudaDevResourceDesc_t").valueTypes("CUdevResourceDesc_st").pointerTypes("@ByPtrPtr CUdevResourceDesc_st"))
               .put(new Info("CUdevSmResource").pointerTypes("CUdevSmResource_st"))
               .put(new Info("CUdevResource").pointerTypes("CUdevResource_st"))
               .put(new Info("cudaDevResource").pointerTypes("cudaDevResource_st"))
               .put(new Info("CUlogsCallbackHandle", "cudaLogsCallbackHandle").valueTypes("CUlogsCallbackEntry_st").pointerTypes("@ByPtrPtr CUlogsCallbackEntry_st"))
               .put(new Info("CU_LAUNCH_PARAM_END", "CU_LAUNCH_PARAM_BUFFER_POINTER", "CU_LAUNCH_PARAM_BUFFER_SIZE").translate(false).cppTypes("void*"))
               .put(new Info("CU_DEVICE_CPU", "CU_DEVICE_INVALID").translate(false).cppTypes("int"))
               .put(new Info("CU_STREAM_LEGACY", "CU_STREAM_PER_THREAD", "cudaStreamLegacy", "cudaStreamPerThread").translate(false).cppTypes("CUstream_st*"))
               .put(new Info("cudaExecutionContext_t").valueTypes("cudaExecutionContext_st").pointerTypes("@ByPtrPtr cudaExecutionContext_st"))
               .put(new Info("cudaArray_t", "cudaArray_const_t").valueTypes("cudaArray").pointerTypes("@ByPtrPtr cudaArray"))
               .put(new Info("cudaMipmappedArray_t", "cudaMipmappedArray_const_t").valueTypes("cudaMipmappedArray").pointerTypes("@ByPtrPtr cudaMipmappedArray"))
               .put(new Info("cudaGraphicsResource_t").valueTypes("cudaGraphicsResource").pointerTypes("@ByPtrPtr cudaGraphicsResource"))
               .put(new Info("cudaStream_t").valueTypes("CUstream_st").pointerTypes("@ByPtrPtr CUstream_st"))
               .put(new Info("cudaEvent_t").valueTypes("CUevent_st").pointerTypes("@ByPtrPtr CUevent_st"))
               .put(new Info("cudaExternalMemory_t").valueTypes("CUexternalMemory_st").pointerTypes("@ByPtrPtr CUexternalMemory_st"))
               .put(new Info("cudaExternalSemaphore_t").valueTypes("CUexternalSemaphore_st").pointerTypes("@ByPtrPtr CUexternalSemaphore_st"))
               .put(new Info("const cudaExternalSemaphore_t").valueTypes("CUexternalSemaphore_st").pointerTypes("@Cast(\"const cudaExternalSemaphore_t*\") @ByPtrPtr CUexternalSemaphore_st"))
               .put(new Info("cudaGraph_t").valueTypes("CUgraph_st").pointerTypes("@ByPtrPtr CUgraph_st"))
               .put(new Info("cudaGraphNode_t").valueTypes("CUgraphNode_st").pointerTypes("@ByPtrPtr CUgraphNode_st"))
               .put(new Info("const cudaGraphNode_t").valueTypes("CUgraphNode_st").pointerTypes("@Cast(\"const cudaGraphNode_t*\") @ByPtrPtr CUgraphNode_st"))
               .put(new Info("cudaFunction_t").valueTypes("CUfunc_st").pointerTypes("@ByPtrPtr CUfunc_st"))
               .put(new Info("cudaLibrary_t").valueTypes("CUlib_st").pointerTypes("@ByPtrPtr CUlib_st"))
               .put(new Info("cudaMemPool_t", "CUmemoryPool").valueTypes("CUmemPoolHandle_st").pointerTypes("@ByPtrPtr CUmemPoolHandle_st"))
               .put(new Info("cudaUserObject_t", "CUuserObject").valueTypes("CUuserObject_st").pointerTypes("@ByPtrPtr CUuserObject_st"))
               .put(new Info("cudaGraphExec_t").valueTypes("CUgraphExec_st").pointerTypes("@ByPtrPtr CUgraphExec_st"))
               .put(new Info("cudaKernel_t").valueTypes("CUkern_st").pointerTypes("@ByPtrPtr CUkern_st"))
               .put(new Info("cudaAsyncCallbackHandle_t").valueTypes("cudaAsyncCallbackEntry").pointerTypes("@ByPtrPtr cudaAsyncCallbackEntry"))
               .put(new Info("cudaGreenCtx_t", "CUgreenCtx").valueTypes("CUgreenCtx_st").pointerTypes("@ByPtrPtr CUgreenCtx_st"))
               .put(new Info("cudaGraphDeviceNode_t", "CUgraphDeviceNode").valueTypes("CUgraphDeviceUpdatableNode_st").pointerTypes("@ByPtrPtr CUgraphDeviceUpdatableNode_st"))
               .put(new Info("GLint", "GLuint", "GLenum").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"));
    }
}
