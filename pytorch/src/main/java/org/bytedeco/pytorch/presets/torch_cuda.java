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
import org.bytedeco.pytorch.presets.torch.PointerInfo;

/**
 * @author Hervé Guillemet
 */
@Properties(
    inherit = torch.class,
    value = {
        @Platform(
            extension = "-gpu",
            // define = "USE_C10D_NCCL", // Not on Windows
            include = {
                "ATen/cudnn/Types.h",
                "ATen/cudnn/Descriptors.h",
                "ATen/cuda/CUDAEvent.h",
                "torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h",

                // For inclusion in JNI only, not parsed
                "ATen/cuda/CUDAGeneratorImpl.h",
            },
            library = "jnitorch"
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
            .put(new Info().enumerate().friendly())
            .put(new Info().javaText("import org.bytedeco.pytorch.*;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.helper.*;"))
            .put(new Info().javaText("import org.bytedeco.cuda.cudart.*;"))
            .put(new Info().javaText("import org.bytedeco.cuda.cusparse.*;"))
            .put(new Info().javaText("import org.bytedeco.cuda.cublas.*;"))
            .put(new Info().javaText("import org.bytedeco.cuda.cusolver.*;"))
            .put(new Info().javaText("import org.bytedeco.cuda.cudnn.*;"))
            // .put(new Info().javaText("import org.bytedeco.cuda.nccl.*;")) // Not on Windows
            .put(new Info().javaText("import org.bytedeco.pytorch.chrono.*;"))
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

            //// std::unordered_map
            ////.put(new Info("std::unordered_map<std::string,std::shared_ptr<c10d::NCCLComm> >").pointerTypes("StringNCCLCommMap").define())
            //.put(new Info("std::unordered_map<std::string,std::shared_ptr<c10d::NCCLComm> >").skip()) // See getNcclErrorDetailStr below. Not on Windows

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
            .put(new Info("std::array<void*,c10d::intra_node_comm::kMaxDevices>").cast().pointerTypes("PointerPointer<Pointer>"))
        ;

        //// Intrusive pointers
        /* Not on Windows
        for (PointerInfo pi : new PointerInfo[]{
            new PointerInfo("c10d::ProcessGroupNCCL::Options"),
            new PointerInfo("c10d::intra_node_comm::IntraNodeComm")
        }) {
            pi.makeIntrusive(infoMap);
        }
         */

        //// Function pointers
        infoMap
            .put(new Info("std::function<void(const c10::cuda::CUDACachingAllocator::TraceEntry&)>").pointerTypes("AllocatorTraceTracker"))
            .put(new Info("std::function<void(int64_t,int64_t,int64_t,int64_t)>").pointerTypes("OutOfMemoryObserver"))
            .put(new Info("std::function<bool(cudaStream_t)>").pointerTypes("StreamFilter"))

        // Function pointer returning shared_ptr don't compile on windows
        // "jnitorch.cpp(98904): error C2526: 'JavaCPP_org_bytedeco_pytorch_functions_GatheredContextSupplier_allocate_callback': C linkage function cannot return C++ class 'std::shared_ptr<c10::GatheredContext>'"
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

        //// Parsed in main torch
        // We need to help namespace resolution and to redefine names of template instances.
        infoMap
            .put(new Info("c10::Stream"))
            .put(new Info("std::optional<c10::Stream>").pointerTypes("StreamOptional"))
            .put(new Info("std::optional<c10::Device>", "std::optional<at::Device>", "optional<at::Device>").pointerTypes("DeviceOptional"))
            .put(new Info("c10::Device"))
            .put(new Info("c10::impl::PyInterpreter"))
            .put(new Info("std::tuple<int,int>").pointerTypes("T_IntInt_T"))
            .put(new Info("std::optional<c10::DeviceIndex>").pointerTypes("ByteOptional"))
            .put(new Info("c10::IntArrayRef", "at::IntArrayRef").pointerTypes("LongArrayRef"))
            .put(new Info("std::vector<at::DataPtr>").pointerTypes("DataPtrVector"))
            .put(new Info("c10::Allocator"))
            .put(new Info("c10d::Work"))
            .put(new Info("c10d::Store", "c10d::ScatterOptions", "c10d::ReduceScatterOptions", "c10d::AllToAllOptions", "c10d::BarrierOptions", "c10d::AllreduceCoalescedOptions"))
            .put(new Info("c10d::BroadcastOptions", "c10d::ReduceOptions", "c10d::AllreduceOptions", "c10d::AllgatherOptions", "c10d::GatherOptions"))
            .put(new Info("CUDAContextLight.h").linePatterns("struct Allocator;").skip()) // Prevent regeneration of Allocator class in cuda package
            .put(new Info("c10d::Backend::Options").pointerTypes("DistributedBackend.Options"))

            .put(new Info("c10::DeviceIndex", "at::DeviceIndex").valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
            .put(new Info("c10::StreamId").valueTypes("long"))
            .put(new Info("c10::cuda::CaptureStatus").valueTypes("int").cast().skip()) // Enum doesn't parse
            .put(new Info("std::pair<std::vector<c10::cuda::DeviceAssertionsData>,std::vector<c10::cuda::CUDAKernelLaunchInfo> >").pointerTypes("DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair").define())
            .put(new Info("c10::impl::GPUTrace::gpuTraceState").skip())
            .put(new Info("at::native::RNNDescriptor::dropout_desc_").skip())
            .put(new Info("at::native::operator <<(std::ostream&, at::native::TensorDescriptor&)",
                "at::native::operator <<(std::ostream&, at::native::FilterDescriptor&)",
                "at::native::cudnnTypeToString", "at::native::getCudnnDataType", "at::native::cudnn_version",
                "c10::cuda::c10_retrieve_device_side_assertion_info").skip())
            .put(new Info("std::function<void(std::shared_ptr<c10d::WorkInfo>)>", "std::function<void(c10d::WorkInfo*)>", "std::function<void(WorkInfo*)>").pointerTypes("WorkInfoConsumer"))

            .put(new Info("c10::cuda::CUDACachingAllocator::CheckpointDelta").immutable()) // at::DataPtr is not constructible

            .put(new Info("c10::cuda::CUDACachingAllocator::kLargeBuffer").skip()) // Triggers UnsatisfiedLinkException as of 2.2.0

            .put(new Info(
                "at::native::Descriptor<cudnnActivationStruct,cudnnCreateActivationDescriptor&,cudnnDestroyActivationDescriptor&>",
                "at::native::Descriptor<cudnnConvolutionStruct,cudnnCreateConvolutionDescriptor&,cudnnDestroyConvolutionDescriptor&>",
                "at::native::Descriptor<cudnnCTCLossStruct,cudnnCreateCTCLossDescriptor&,cudnnDestroyCTCLossDescriptor&>",
                "at::native::Descriptor<cudnnDropoutStruct,cudnnCreateDropoutDescriptor&,cudnnDestroyDropoutDescriptor&>",
                "at::native::Descriptor<cudnnFilterStruct,cudnnCreateFilterDescriptor&,cudnnDestroyFilterDescriptor&>",
                "at::native::Descriptor<cudnnRNNDataStruct,cudnnCreateRNNDataDescriptor&,cudnnDestroyRNNDataDescriptor&>",
                "at::native::Descriptor<cudnnRNNStruct,cudnnCreateRNNDescriptor&,cudnnDestroyRNNDescriptor&>",
                "at::native::Descriptor<cudnnSpatialTransformerStruct,cudnnCreateSpatialTransformerDescriptor&,cudnnDestroySpatialTransformerDescriptor&>",
                "at::native::Descriptor<cudnnTensorStruct,cudnnCreateTensorDescriptor&,cudnnDestroyTensorDescriptor&>",

                "std::hash<c10::cuda::CUDAStream>",

                "std::shared_ptr<c10::CreateContextFn> (*)()", "c10::cuda::CUDACachingAllocator::CreateContextFn"  // See comment for GatheredContextSupplier

                // "std::enable_shared_from_this<WorkNCCL>" // Not on Windows

            ).cast().pointerTypes("Pointer"));
        new PointerInfo("c10d::Store").makeIntrusive(infoMap);
        new PointerInfo("c10d::Work").makeIntrusive(infoMap);


        //// CUDA types
        infoMap
            .put(new Info("cudaStream_t").valueTypes("CUstream_st").pointerTypes("@ByPtrPtr CUstream_st"))
            .put(new Info("cudaEvent_t").valueTypes("CUevent_st").pointerTypes("@ByPtrPtr CUevent_st"))
            .put(new Info("cusparseHandle_t").valueTypes("cusparseContext").pointerTypes("@ByPtrPtr cusparseContext"))
            .put(new Info("cublasHandle_t").valueTypes("cublasContext").pointerTypes("@ByPtrPtr cublasContext"))
            .put(new Info("cublasLtHandle_t").valueTypes("cublasLtContext").pointerTypes("@ByPtrPtr cublasLtContext"))
            .put(new Info("cusolverDnHandle_t").valueTypes("cusolverDnContext").pointerTypes("@ByPtrPtr cusolverDnContext"))
            .put(new Info("cudnnHandle_t").valueTypes("cudnnContext").pointerTypes("@ByPtrPtr cudnnContext"))
            // .put(new Info("ncclComm_t").valueTypes("ncclComm").pointerTypes("@ByPtrPtr ncclComm", "@Cast(\"ncclComm**\") PointerPointer")) // Not on Windows

            .put(new Info( // Enums, cuda presets doesn't use Info.enumerate
                "cudnnActivationMode_t", "cudnnLossNormalizationMode_t", "cudnnRNNInputMode_t", "cudnnRNNDataLayout_t",
                "cudnnDirectionMode_t", "cudnnRNNMode_t", "cudaStreamCaptureMode", "cudnnDataType_t", "cudnnNanPropagation_t",
                "cusparseStatus_t", "cusolverStatus_t", "cudnnRNNAlgo_t", "cudnnNanPropagation_t", "cublasStatus_t", "cudaError_t",
                "cudaMemcpyKind", "ncclResult_t", "ncclDataType_t", "ncclRedOp_t", "ncclScalarResidence_t"
            ).valueTypes("int").cast())
        ;

        new torch.ArrayInfo("CUDAStream").elementTypes("c10::cuda::CUDAStream").mapArrayRef(infoMap);

        new PointerInfo("c10::cuda::CUDACachingAllocator::AllocatorState").makeShared(infoMap);
        //new PointerInfo("c10d::NCCLComm").makeShared(infoMap); // See getNcclErrorDetailStr below

        // Classes that are not part of the API (no TORCH_API nor C10_API) and are not argument nor return type of API methods.
        infoMap.put(new Info(
            "c10::cuda::OptionalCUDAGuard",
            "c10::cuda::OptionalCUDAStreamGuard",
            "c10::cuda::impl::CUDAGuardImpl",
            "c10::FreeMemoryCallback", // in API, but useless as long as we don't map FreeCudaMemoryCallbacksRegistry,
            "AT_DISALLOW_COPY_AND_ASSIGN",
            "c10d::NCCLComm", "std::shared_ptr<c10d::NCCLComm>" // See getNcclErrorDetailStr below
        ).skip())
        ;

        infoMap
            .put(new Info("USE_CUDNN_RNN_V8_API").define()) // Using CuDNN 8.9.7 or more recent
            .put(new Info("defined(IS_NCCL_EXP) && defined(NCCL_COMM_DUMP)").define(false))
        ;

        //// Different C++ API between platforms
        infoMap
            .put(new Info("at::cuda::getCurrentCUDABlasLtHandle").skip()) // No cublas lt with Microsoft compiler
        ;

        //// Don't map all custom pytorch errors since there is currently no way to catch them as objects from Java
        infoMap.put(new Info(
            "c10::CUDAError",
            "c10::CuDNNError"
        ).skip());

        //// Not part of public API or not exposed by libtorch
        infoMap
            .put(new Info(
                "c10d::DumpPipe",
                "c10d::nccl_use_nonblocking",
                "c10d::getNcclErrorDetailStr", // Prevents c10d::NCCLComm to be mapped
                "c10d::ncclGetErrorWithVersion",
                "c10d::nccl_nonblocking_timeout",
                "c10d::getNcclVersion",
                "c10d::ProcessGroupNCCL::operator <<"
                ).skip())

        ;

        //// Help namespace resolution
        infoMap
            .put(new Info("std::optional", "c10d::WorkInfo"))
        ;

        //// No way to map
        infoMap
            .put(new Info("std::optional<std::function<std::string()> >").skip())
        ;
    }
}
