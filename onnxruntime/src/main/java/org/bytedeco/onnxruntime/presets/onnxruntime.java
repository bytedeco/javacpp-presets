/*
 * Copyright (C) 2019-2025 Samuel Audet, Alexander Merritt
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
package org.bytedeco.onnxruntime.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.dnnl.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = dnnl.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            compiler = "cpp17",
            define = {"GENERIC_EXCEPTION_CLASS Ort::Exception", "GENERIC_EXCEPTION_TOSTRING what()"},
            include = {
                "onnxruntime/core/session/onnxruntime_c_api.h",
                "onnxruntime/core/session/onnxruntime_ep_c_api.h",
                "onnxruntime/core/session/onnxruntime_cxx_api.h",
                "onnxruntime/core/providers/cpu/cpu_provider_factory.h",
                "onnxruntime/core/providers/dnnl/dnnl_provider_options.h",
//                "onnxruntime/core/providers/cuda/cuda_provider_options.h",
//                "onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h",
//                "onnxruntime/core/providers/cuda/cuda_provider_factory.h",
//                "onnxruntime/core/providers/dnnl/dnnl_provider_factory.h",
//                "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h",
//                "onnxruntime/core/providers/nuphar/nuphar_provider_factory.h",
//                "onnxruntime/core/providers/openvino/openvino_provider_factory.h",
//                "onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h",
//                "onnxruntime/core/providers/migraphx/migraphx_provider_factory.h",
//                "onnxruntime/core/providers/acl/acl_provider_factory.h",
//                "onnxruntime/core/providers/armnn/armnn_provider_factory.h",
//                "onnxruntime/core/providers/coreml/coreml_provider_factory.h",
//                "onnxruntime/core/providers/rocm/rocm_provider_factory.h",
//                "onnxruntime/core/providers/dml/dml_provider_factory.h",
                "onnxruntime_training_c_api.h",
            },
            exclude = {"CL/opencl.h", "CL/cl_version.h", "CL/cl_platform.h", "CL/cl.h"/*, "CL/cl_gl.h", "CL/cl_gl_ext.h", "CL/cl_ext.h"*/,
                       "onnxruntime/core/session/onnxruntime_ep_c_api.h"},
            link = {"onnxruntime_providers_shared", "onnxruntime@.1", "onnxruntime_providers_dnnl"}
        ),
        @Platform(
            value = {"linux", "macosx", "windows"},
            extension = "-gpu",
            link = {"onnxruntime_providers_shared", "onnxruntime@.1", "onnxruntime_providers_dnnl", "onnxruntime_providers_cuda"}
        ),
    },
    target = "org.bytedeco.onnxruntime",
    global = "org.bytedeco.onnxruntime.global.onnxruntime"
)
public class onnxruntime implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "onnxruntime"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || extension == null || !extension.equals("-gpu")) {
            return;
        }
        int i = 0;
        if (platform.startsWith("windows")) {
            preloads.add(i++, "zlibwapi");
        }
        String[] libs = {"cudart", "cublasLt", "cublas", "cufft", "curand", "cudnn",
                         "cudnn_graph", "cudnn_engines_precompiled", "cudnn_engines_runtime_compiled",
                         "cudnn_heuristic", "cudnn_ops", "cudnn_adv", "cudnn_cnn"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.9" : lib.equals("cufft") ? "@.12" : lib.equals("curand") ? "@.10" : lib.equals("cudart") ? "@.13" : "@.13";
            } else if (platform.startsWith("windows")) {
                lib += lib.startsWith("cudnn") ? "64_9" : lib.equals("cufft") ? "64_12" : lib.equals("curand") ? "64_10" : lib.equals("cudart") ? "64_13" : "64_13";
            } else {
                continue; // no CUDA
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.putFirst(new Info("opencl.h", "cl_version.h", "cl_platform.h", "cl.h"/*, "cl_gl.h", "cl_gl_ext.h", "cl_ext.h"*/).skip())
               .put(new Info("ORTCHAR_T", "std::basic_string<ORTCHAR_T>",
                             "onnxruntime_float16::BFloat16Impl<BFloat16_t>",
                             "onnxruntime_float16::Float16Impl<Float16_t>", "Ort::ConstHardwareDevice").cppText("").cppTypes().cast().pointerTypes("Pointer"))
               .put(new Info("ORT_EXPORT", "ORT_API_CALL", "ORT_FILE", "NO_EXCEPTION", "ORT_ALL_ARGS_NONNULL", "OrtCustomOpApi").cppTypes().annotations())
               .put(new Info("ORT_API_MANUAL_INIT", "__cpp_if_constexpr").define(false))
               .put(new Info("USE_CUDA", "USE_DNNL").define(true))
               .put(new Info("Ort::stub_api", "Ort::Global<T>::api_", "std::nullptr_t", "Ort::Env::s_api", "std::vector<Ort::AllocatedStringPtr>", "not_defined_in_this_build").skip())
               .put(new Info("Ort::AllocatedStringPtr").valueTypes("@UniquePtr(\"char, Ort::detail::AllocatedFree\") @Cast(\"char*\") BytePointer"))
//               .put(new Info("std::string").annotations("@Cast({\"char*\", \"std::string&&\"}) @StdString").valueTypes("BytePointer", "String").pointerTypes("BytePointer"))
               .put(new Info("const std::vector<Ort::ValueInfo>", "std::vector<Ort::ValueInfo>").pointerTypes("ValueInfoVector").define())
               .put(new Info("const std::vector<Ort::OpAttr>", "std::vector<Ort::OpAttr>").pointerTypes("OpAttrVector").define())
               .put(new Info("std::pair<int,int>").pointerTypes("IntPair").define())
               .put(new Info("std::pair<std::string,int>").pointerTypes("StringIntPair").define())
               .put(new Info("std::vector<std::pair<std::string,int> >").pointerTypes("StringIntPairVector").define())
               .put(new Info("std::vector<float>", "Ort::ShapeInferContext::Floats").pointerTypes("FloatVector").define())
               .put(new Info("std::vector<int64_t>", "Ort::ShapeInferContext::Ints").pointerTypes("LongVector").define())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<Ort::Value>").valueTypes("@StdMove ValueVector").pointerTypes("ValueVector").define())
               .put(new Info("std::unordered_map<std::string,size_t>").pointerTypes("StringSizeTMap").define())
               .put(new Info("std::unordered_map<std::string,std::string>").pointerTypes("StringStringMap").define())
               .put(new Info("Ort::Value").valueTypes("@StdMove Value").pointerTypes("Value"))
               .put(new Info("Ort::Value::CreateTensor<float>").javaNames("CreateTensorFloat"))
               .put(new Info("Ort::Value::CreateTensor<double>").javaNames("CreateTensorDouble"))
               .put(new Info("Ort::Value::CreateTensor<int8_t>").javaNames("CreateTensorByte"))
               .put(new Info("Ort::Value::CreateTensor<int16_t>").javaNames("CreateTensorShort"))
               .put(new Info("Ort::Value::CreateTensor<int32_t>").javaNames("CreateTensorInt"))
               .put(new Info("Ort::Value::CreateTensor<int64_t>").javaNames("CreateTensorLong"))
               .put(new Info("Ort::Value::CreateTensor<uint8_t>").javaNames("CreateTensorUByte"))
               .put(new Info("Ort::Value::CreateTensor<uint16_t>").javaNames("CreateTensorUShort"))
               .put(new Info("Ort::Value::CreateTensor<uint32_t>").javaNames("CreateTensorUInt"))
               .put(new Info("Ort::Value::CreateTensor<uint64_t>").javaNames("CreateTensorULong"))
               .put(new Info("Ort::Value::CreateTensor<bool>").javaNames("CreateTensorBool"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<float>").javaNames("GetTensorMutableDataFloat"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<double>").javaNames("GetTensorMutableDataDouble"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<int8_t>").javaNames("GetTensorMutableDataByte"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<int16_t>").javaNames("GetTensorMutableDataShort"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<int32_t>").javaNames("GetTensorMutableDataInt"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<int64_t>").javaNames("GetTensorMutableDataLong"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<uint8_t>").javaNames("GetTensorMutableDataUByte"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<uint16_t>").javaNames("GetTensorMutableDataUShort"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<uint32_t>").javaNames("GetTensorMutableDataUInt"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<uint64_t>").javaNames("GetTensorMutableDataULong"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>::GetTensorMutableData<bool>").javaNames("GetTensorMutableDataBool"))
               .put(new Info("Ort::detail::Unowned<OrtAllocator>").pointerTypes("UnownedAllocator").purify())
               .put(new Info("Ort::detail::Unowned<const OrtEpAssignedNode>").pointerTypes("ConstEpAssignedNode").purify())
               .put(new Info("Ort::detail::Unowned<const OrtEpAssignedSubgraph>").pointerTypes("ConstEpAssignedSubgraph").purify())
               .put(new Info("Ort::detail::Unowned<const Ort::MemoryInfo>").pointerTypes("UnownedMemoryInfo").purify())
               .put(new Info("Ort::detail::Unowned<Ort::TensorTypeAndShapeInfo>").pointerTypes("UnownedTensorTypeAndShapeInfo").purify())
               .put(new Info("Ort::detail::Unowned<Ort::SequenceTypeInfo>").pointerTypes("UnownedSequenceTypeInfo").purify())
               .put(new Info("Ort::detail::Unowned<Ort::MapTypeInfo>").pointerTypes("UnownedMapTypeInfo").purify())
               .put(new Info("Ort::Exception").pointerTypes("OrtException").purify())
               .put(new Info("Ort::MemoryAllocation", "OrtApi").purify())
               .put(new Info("Ort::MemoryAllocation::operator =", "Ort::RunOptions::GetRunLogSeverityLevel", "ShapeInferFn").skip())

               .put(new Info("Ort::detail::ConstGraphImpl<Ort::detail::Unowned<const OrtGraph> >").pointerTypes("ConstGraph"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtGraph> >").pointerTypes("BaseConstGraph"))
               .put(new Info("Ort::detail::ConstValueImpl<detail::Unowned<const OrtValue> >").pointerTypes("ConstValue"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtValue> >").pointerTypes("BaseConstValue"))
               .put(new Info("Ort::detail::ConstSessionOptionsImpl<detail::Unowned<const OrtSessionOptions> >").pointerTypes("ConstSessionOptions"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtSessionOptions> >").pointerTypes("BaseConstSessionOptions"))
               .put(new Info("Ort::detail::ConstSessionImpl<detail::Unowned<const OrtSession> >").pointerTypes("ConstSession"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtSession> >").pointerTypes("BaseConstSession"))
               .put(new Info("Ort::detail::MapTypeInfoImpl<detail::Unowned<const OrtMapTypeInfo> >").pointerTypes("ConstMapTypeInfo"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtMapTypeInfo> >").pointerTypes("BaseConstMapTypeInfo"))
               .put(new Info("Ort::detail::ConstIoBindingImpl<detail::Unowned<const OrtIoBinding> >").pointerTypes("ConstIoBinding"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtIoBinding> >").pointerTypes("BaseConstIoBinding"))
               .put(new Info("Ort::detail::TensorTypeAndShapeInfoImpl<detail::Unowned<const OrtTensorTypeAndShapeInfo> >").pointerTypes("ConstTensorTypeAndShapeInfo"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtTensorTypeAndShapeInfo> >").pointerTypes("BaseConstTensorTypeAndShapeInfo"))
               .put(new Info("Ort::detail::AllocatorImpl<Ort::detail::Unowned<OrtAllocator> >").pointerTypes("AllocatorWithDefaultOptionsImpl"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<OrtAllocator> >").pointerTypes("BaseAllocatorWithDefaultOptions"))
               .put(new Info("Ort::detail::EpAssignedNodeImpl<Ort::detail::Unowned<const OrtEpAssignedNode> >").pointerTypes("EpAssignedNodeWithDefaultOptionsImpl"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtEpAssignedNode> >").pointerTypes("BaseEpAssignedNodeWithDefaultOptions"))
               .put(new Info("Ort::detail::EpAssignedSubgraphImpl<Ort::detail::Unowned<const OrtEpAssignedSubgraph> >").pointerTypes("EpAssignedSubgraphWithDefaultOptionsImpl"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtEpAssignedSubgraph> >").pointerTypes("BaseEpAssignedSubgraphWithDefaultOptions"))
               .put(new Info("Ort::detail::Base<Ort::detail::Unowned<const OrtGraph> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtValue> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtSessionOptions> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtSession> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtMapTypeInfo> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtIoBinding> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtTensorTypeAndShapeInfo> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtEpAssignedNode> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<const OrtEpAssignedSubgraph> >::release",
                             "Ort::detail::Base<Ort::detail::Unowned<OrtAllocator> >::release").skip())

               .put(new Info("Ort::detail::Base<OrtArenaCfg>", "Ort::detail::Base<OrtArenaCfg>").pointerTypes("BaseArenaCfg"))
               .put(new Info("Ort::detail::AllocatorImpl<OrtAllocator>").pointerTypes("AllocatorImpl"))
               .put(new Info("Ort::detail::Base<OrtAllocator>").pointerTypes("BaseAllocator"))
               .put(new Info("Ort::detail::ConstIoBindingImpl<OrtIoBinding>").pointerTypes("ConstIoBindingImpl"))
               .put(new Info("Ort::detail::IoBindingImpl<OrtIoBinding>").pointerTypes("IoBindingImpl"))
               .put(new Info("Ort::detail::Base<OrtIoBinding>").pointerTypes("BaseIoBinding"))
               .put(new Info("Ort::detail::MemoryInfoImpl<OrtMemoryInfo>").pointerTypes("MemoryInfoImpl"))
               .put(new Info("Ort::detail::Base<OrtMemoryInfo>",
                             "Ort::detail::BaseMemoryInfo<Ort::detail::Base<OrtMemoryInfo> >",
                             "Ort::detail::BaseMemoryInfo<Ort::detail::Base<const OrtMemoryInfo> >").pointerTypes("BaseMemoryInfo"))
               .put(new Info("Ort::detail::Base<OrtModelMetadata>", "Ort::detail::Base<OrtModelMetadata>").pointerTypes("BaseModelMetadata"))
               .put(new Info("Ort::detail::Base<OrtCustomOpDomain>", "Ort::detail::Base<OrtCustomOpDomain>").pointerTypes("BaseCustomOpDomain"))
               .put(new Info("Ort::detail::Base<OrtEnv>", "Ort::detail::Base<OrtEnv> ").pointerTypes("BaseEnv"))
               .put(new Info("Ort::detail::Base<OrtRunOptions>", "Ort::detail::Base<OrtRunOptions>").pointerTypes("BaseRunOptions"))
               .put(new Info("Ort::detail::ConstSessionImpl<OrtSession>").pointerTypes("ConstSessionImpl"))
               .put(new Info("Ort::detail::SessionImpl<OrtSession>").pointerTypes("SessionImpl"))
               .put(new Info("Ort::detail::Base<OrtSession>").pointerTypes("BaseSession"))
               .put(new Info("Ort::detail::SessionOptionsImpl<OrtSessionOptions>").pointerTypes("SessionOptionsImpl"))
               .put(new Info("Ort::detail::ConstSessionOptionsImpl<OrtSessionOptions>").pointerTypes("ConstSessionOptionsImpl"))
               .put(new Info("Ort::detail::Base<OrtSessionOptions>").pointerTypes("BaseSessionOptions"))
               .put(new Info("Ort::detail::TensorTypeAndShapeInfoImpl<OrtTensorTypeAndShapeInfo>").pointerTypes("TensorTypeAndShapeInfoImpl"))
               .put(new Info("Ort::detail::Base<OrtTensorTypeAndShapeInfo>").pointerTypes("BaseTensorTypeAndShapeInfo"))
               .put(new Info("Ort::detail::SequenceTypeInfoImpl<OrtSequenceTypeInfo>").pointerTypes("BaseSequenceTypeInfoImpl"))
               .put(new Info("Ort::detail::Base<OrtSequenceTypeInfo>").pointerTypes("BaseSequenceTypeInfo"))
               .put(new Info("Ort::detail::MapTypeInfoImpl<OrtMapTypeInfo>").pointerTypes("MapTypeInfoImpl"))
               .put(new Info("Ort::detail::Base<OrtMapTypeInfo>").pointerTypes("BaseMapTypeInfo"))
               .put(new Info("Ort::detail::TypeInfoImpl<OrtTypeInfo>").pointerTypes("TypeInfoImpl"))
               .put(new Info("Ort::detail::OptionalTypeInfoImpl<OrtTypeInfo>").pointerTypes("OptionalTypeInfoImpl"))
               .put(new Info("Ort::detail::Base<OrtTypeInfo>").pointerTypes("BaseTypeInfo"))
               .put(new Info("Ort::detail::ConstValueImpl<OrtValue>").pointerTypes("ConstValueImpl"))
               .put(new Info("Ort::detail::ValueImpl<OrtValue>").pointerTypes("ValueImpl"))
               .put(new Info("Ort::detail::Base<OrtValue>").pointerTypes("BaseValue"))
               .put(new Info("Ort::detail::Base<OrtOp>").pointerTypes("BaseOp"))
               .put(new Info("Ort::detail::Base<OrtOpAttr>").pointerTypes("BaseOpAttr"))
               .put(new Info("Ort::detail::ConstOpAttrImpl<OrtOpAttr>").pointerTypes("ConstOpAttrImpl"))
               .put(new Info("Ort::detail::Base<OrtStatus>").pointerTypes("BaseStatus"))
               .put(new Info("Ort::detail::ConstKernelDefImpl<OrtKernelDef>").pointerTypes("ConstKernelDefImpl"))
               .put(new Info("Ort::detail::Base<OrtKernelDef>").pointerTypes("BaseKernelDef"))
               .put(new Info("Ort::detail::Base<OrtKernelDefBuilder>").pointerTypes("BaseKernelDefBuilder"))
               .put(new Info("Ort::detail::Base<OrtKernelRegistry>").pointerTypes("BaseKernelRegistry"))
               .put(new Info("Ort::detail::KernelInfoImpl<OrtKernelInfo>").pointerTypes("KernelInfoImpl"))
               .put(new Info("Ort::detail::Base<OrtKernelInfo>").pointerTypes("BaseKernelInfo"))
               .put(new Info("Ort::detail::Base<OrtThreadingOptions>").pointerTypes("BaseThreadingOptions"))
               .put(new Info("Ort::detail::Base<OrtLoraAdapter>").pointerTypes("BaseLoraAdapter"))
               .put(new Info("Ort::detail::Base<OrtModelCompilationOptions>").pointerTypes("BaseModelCompilationOptions"))
               .put(new Info("Ort::detail::Base<OrtGraph>").pointerTypes("BaseGraph"))
               .put(new Info("Ort::detail::GraphImpl<OrtGraph>").pointerTypes("GraphImpl"))
               .put(new Info("Ort::detail::ConstGraphImpl<OrtGraph>").pointerTypes("ConstGraphImpl"))
               .put(new Info("Ort::detail::Base<OrtKeyValuePairs> ").pointerTypes("BaseKeyValuePairs"))
               .put(new Info("Ort::detail::KeyValuePairsImpl<OrtKeyValuePairs>").pointerTypes("KeyValuePairsImpl"))
               .put(new Info("Ort::detail::Base<OrtNode>").pointerTypes("BaseNode"))
               .put(new Info("Ort::detail::NodeImpl<OrtNode>").pointerTypes("NodeImpl"))
               .put(new Info("Ort::detail::ConstNodeImpl<OrtNode>").pointerTypes("ConstNodeImpl"))
               .put(new Info("Ort::detail::Base<OrtModel>").pointerTypes("BaseModel"))
               .put(new Info("Ort::detail::ModelImpl<OrtModel>").pointerTypes("ModelImpl"))
               .put(new Info("Ort::detail::Base<OrtValueInfo>").pointerTypes("BaseValueInfo"))
               .put(new Info("Ort::detail::ValueInfoImpl<OrtValueInfo>").pointerTypes("ValueInfoImpl"))
               .put(new Info("Ort::detail::ConstValueInfoImpl<OrtValueInfo>").pointerTypes("ConstValueInfoImpl"))
               .put(new Info("Ort::detail::Base<OrtEpDevice>").pointerTypes("BaseEpDevice"))
               .put(new Info("Ort::detail::EpDeviceImpl<OrtEpDevice>").pointerTypes("EpDeviceImpl"))
               .put(new Info("Ort::detail::Base<OrtCUDAProviderOptionsV2>").pointerTypes("BaseCUDAProviderOptionsV2"))
               .put(new Info("Ort::detail::Base<OrtPrepackedWeightsContainer>").pointerTypes("BasePrepackedWeightsContainer"))
               .put(new Info("Ort::detail::Base<OrtTensorRTProviderOptionsV2>").pointerTypes("BaseTensorRTProviderOptionsV2"))
               .put(new Info("Ort::detail::Base<OrtSyncStream>").pointerTypes("BaseSyncStream"))
               .put(new Info("Ort::detail::SyncStreamImpl<OrtSyncStream>").pointerTypes("SyncStreamImpl"))
               .put(new Info("Ort::detail::Base<OrtExternalInitializerInfo>").pointerTypes("BaseExternalInitializerInfo"))
               .put(new Info("Ort::detail::ConstExternalInitializerInfoImpl<OrtExternalInitializerInfo>").pointerTypes("ConstExternalInitializerInfoImpl"))

               .put(new Info("OrtSessionOptionsAppendExecutionProvider_MIGraphX", "OrtSessionOptionsAppendExecutionProvider_Tensorrt",
                             "OrtSessionOptionsAppendExecutionProvider_ROCM", "Ort::detail::OptionalTypeInfoImpl<OrtTypeInfo>::GetOptionalElementType").skip())
               .put(new Info("OrtSessionOptionsAppendExecutionProvider_CUDA").annotations("@Platform(extension=\"-gpu\")").javaNames("OrtSessionOptionsAppendExecutionProvider_CUDA"));
    }
}
