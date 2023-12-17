/*
 * Copyright (C) 2023 Hervé Guillemet
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
package org.bytedeco.pytorch.presets;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * @author Hervé Guillemet
 */
@Properties(
    inherit = torch.class,
    value = {
        @Platform(
            extension = "-gpu",
            include = {
                "ATen/cudnn/Descriptors.h",
                "ATen/cudnn/Types.h",
                "c10/cuda/CUDAGuard.h",

                // For inclusion in JNI only, not parsed
                "ATen/cuda/CUDAGeneratorImpl.h",
            },
            link = { "cudart", "cusparse" },
            linkpath = {
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib/x64/",
                "/usr/local/cuda-12.3/lib64/",
                "/usr/local/cuda/lib64/",
                "/usr/lib64/"
            }
        ),
    },
    target = "org.bytedeco.pytorch.cuda",
    global = "org.bytedeco.pytorch.global.torch_cuda"
)
public class torch_cuda implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        String extension = properties.getProperty("platform.extension");
        if (extension != null && extension.endsWith("-gpu"))
          torch.initIncludes(getClass(), properties);
    }

    @Override
    public void map(InfoMap infoMap) {

        torch.sharedMap(infoMap);

        infoMap
            .put(new Info("basic/containers").cppTypes("c10::optional"))

            .put(new Info().enumerate().friendly())
            .put(new Info().javaText("import org.bytedeco.pytorch.*;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.functions.*;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.Error;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.global.torch.DeviceType;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.global.torch.ScalarType;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.global.torch.MemoryFormat;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.Allocator;"))

            .put(new Info().javaText(
                "@Namespace(\"at\") public static native @ByVal @Name(\"make_generator<at::CUDAGeneratorImpl>\") Generator make_generator_cuda();\n" +
                "@Namespace(\"at\") public static native @ByVal @Name(\"make_generator<at::CUDAGeneratorImpl,int8_t>\") Generator make_generator_cuda(@Cast(\"int8_t&&\") byte device_index);\n"
            ))

            .put(new Info(
                "at::CUDAGeneratorImpl"
            ).skip())


            //// std::unordered_set
            .put(new Info("std::unordered_set<void*>").pointerTypes("PointerSet").define())

            //// std::atomic
            .put(new Info("c10::cuda::CUDACachingAllocator::allocator").skip()) // Relies on CUDACachingAllocator.get()
            .put(new Info("std::atomic<const c10::impl::PyInterpreter*>").cast().pointerTypes("PyInterpreter"))

            //// std::vector
            .put(new Info("std::vector<c10::cuda::DeviceAssertionsData>").pointerTypes("DeviceAssertionsDataVector").define())
            .put(new Info("std::vector<c10::cuda::CUDAKernelLaunchInfo>").pointerTypes("CUDAKernelLaunchInfoVector").define())
            .put(new Info("const std::vector<c10::cuda::CUDACachingAllocator::TraceEntry>", "std::vector<c10::cuda::CUDACachingAllocator::TraceEntry>").pointerTypes("TraceEntryVector").define())

            //// std::array
            .put(new Info("std::array<c10::cuda::CUDACachingAllocator::Stat,3>", "c10::cuda::CUDACachingAllocator::StatArray").cast().pointerTypes("Stat"))

            //// Function pointers
            // Function pointer returning shared_ptr don't compile on windows
            // "D:\a\javacpp-presets\javacpp-presets\pytorch\target\native\org\bytedeco\pytorch\windows-x86_64\jnitorch.cpp(98904): error C2526: 'JavaCPP_org_bytedeco_pytorch_functions_GatheredContextSupplier_allocate_callback': C linkage function cannot return C++ class 'std::shared_ptr<c10::GatheredContext>'"
            //.put(new Info("std::shared_ptr<c10::GatheredContext> (*)()", "c10::cuda::CUDACachingAllocator::CreateContextFn").pointerTypes("GatheredContextSupplier").valueTypes("GatheredContextSupplier").skip())
        ;

        //// Avoiding name clashes by skipping or renaming
        // Keep the instance methods of CUDAAllocator only, to not pollute global class
        infoMap.put(new Info("c10::cuda::CUDACachingAllocator::get").javaNames("getAllocator"));
        for (String s : new String[]{"get", "raw_alloc", "raw_alloc_with_stream", "raw_delete", "init",
            "setMemoryFraction", "emptyCache", "cacheInfo", "getBaseAllocation", "recordStream", "getDeviceStats",
            "resetAccumulatedStats", "resetPeakStats", "snapshot", "getCheckpointState", "setCheckpointPoolState",
            "beginAllocateStreamToPool", "endAllocateStreamToPool", "isHistoryEnabled", "recordHistory",
            "checkPoolLiveAllocations", "attachOutOfMemoryObserver", "releasePool", "getIpcDevPtr", "name",
            "memcpyAsync", "enablePeerAccess"}) {
            infoMap.put(new Info("c10::cuda::CUDACachingAllocator::CUDAAllocator::" + s)); // Necessary or the ns qualifying algorithm of Parser will pick c10::cuda::CUDACachingAllocator instead
            infoMap.put(new Info("c10::cuda::CUDACachingAllocator::" + s).skip());
        }

        //// Already defined in main torch
        infoMap
            .put(new Info("c10::Stream").pointerTypes("Stream"))
            .put(new Info("c10::optional<c10::Stream>").pointerTypes("StreamOptional"))
            .put(new Info("c10::optional<c10::Device>").pointerTypes("DeviceOptional"))
            .put(new Info("c10::Device").pointerTypes("Device"))
            .put(new Info("c10::impl::PyInterpreter").pointerTypes("PyInterpreter"))
            .put(new Info("std::tuple<int,int>").pointerTypes("T_IntInt_T"))
            .put(new Info("c10::optional<c10::DeviceIndex>").pointerTypes("ByteOptional"))
            .put(new Info("c10::IntArrayRef", "at::IntArrayRef").pointerTypes("LongArrayRef"))
            .put(new Info("std::vector<at::DataPtr>").pointerTypes("DataPtrVector"))

            .put(new Info("c10::DeviceIndex").valueTypes("byte"))
            .put(new Info("c10::StreamId").valueTypes("long"))
            .put(new Info("c10::cuda::CaptureStatus").valueTypes("int").cast().skip()) // Enum doesn't parse
            .put(new Info("std::pair<std::vector<c10::cuda::DeviceAssertionsData>,std::vector<c10::cuda::CUDAKernelLaunchInfo> >").pointerTypes("DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair").define())
            .put(new Info("c10::CuDNNError").purify())
            .put(new Info("c10::impl::GPUTrace::gpuTraceState").skip())
            .put(new Info("at::native::RNNDescriptor::dropout_desc_").skip())
            .put(new Info("at::native::operator <<(std::ostream&, at::native::TensorDescriptor&)",
                "at::native::operator <<(std::ostream&, at::native::FilterDescriptor&)",
                "at::native::cudnnTypeToString", "at::native::getCudnnDataType", "at::native::cudnn_version",
                "c10::cuda::c10_retrieve_device_side_assertion_info").skip())

            .put(new Info("c10::cuda::CUDACachingAllocator::CheckpointDelta").immutable()) // at::DataPtr is not constructible

            .put(new Info(
                "at::native::Descriptor<cudnnActivationStruct,cudnnCreateActivationDescriptor&,cudnnDestroyActivationDescriptor&>",
                "at::native::Descriptor<cudnnConvolutionStruct,cudnnCreateConvolutionDescriptor&,cudnnDestroyConvolutionDescriptor&>",
                "at::native::Descriptor<cudnnCTCLossStruct,cudnnCreateCTCLossDescriptor&,cudnnDestroyCTCLossDescriptor&>",
                "at::native::Descriptor<cudnnDropoutStruct,cudnnCreateDropoutDescriptor&,cudnnDestroyDropoutDescriptor&>",
                "at::native::Descriptor<cudnnFilterStruct,cudnnCreateFilterDescriptor&,cudnnDestroyFilterDescriptor&>",
                "at::native::Descriptor<cudnnRNNStruct,cudnnCreateRNNDescriptor&,cudnnDestroyRNNDescriptor&>",
                "at::native::Descriptor<cudnnSpatialTransformerStruct,cudnnCreateSpatialTransformerDescriptor&,cudnnDestroySpatialTransformerDescriptor&>",
                "at::native::Descriptor<cudnnTensorStruct,cudnnCreateTensorDescriptor&,cudnnDestroyTensorDescriptor&>",

                "std::hash<c10::cuda::CUDAStream>",

                "std::shared_ptr<c10::CreateContextFn> (*)()", "c10::cuda::CUDACachingAllocator::CreateContextFn"  // See comment for GatheredContextSupplier

            ).cast().pointerTypes("Pointer"))

            //// CUDA types
            .put(new Info( // Struct
                "cudaDeviceProp"
            ).pointerTypes("Pointer"))
            .put(new Info( // Pointers to opaque structs
                "cudaStream_t", "cusparseHandle_t", "cublasHandle_t", "cusolverDnHandle_t", "cudnnHandle_t", "cudaEvent_t"
            ).valueTypes("Pointer").cast())
            .put(new Info( // Enums
                "cudnnActivationMode_t", "cudnnLossNormalizationMode_t", "cudnnRNNInputMode_t",
                "cudnnDirectionMode_t", "cudnnRNNMode_t", "cudaStreamCaptureMode", "cudnnDataType_t", "cudnnNanPropagation_t",
                "cusparseStatus_t", "cusolverStatus_t", "cudnnRNNAlgo_t", "cudnnNanPropagation_t", "cublasStatus_t", "cudaError_t"
            ).valueTypes("int").cast())
        ;

        new torch.ArrayInfo("CUDAStream").elementTypes("c10::cuda::CUDAStream").mapArrayRef(infoMap);

        new torch.PointerInfo("c10::cuda::CUDACachingAllocator::AllocatorState").makeShared(infoMap);

        // Classes that are not part of the API (no TORCH_API nor C10_API) and are not argument nor return type of API methods.
        infoMap.put(new Info(
            "c10::cuda::OptionalCUDAGuard",
            "c10::cuda::OptionalCUDAStreamGuard",
            "c10::cuda::impl::CUDAGuardImpl",
            "c10::FreeMemoryCallback" // in API, but useless as long as we don't map FreeCudaMemoryCallbacksRegistry,
        ).skip())
        ;

    }
}
