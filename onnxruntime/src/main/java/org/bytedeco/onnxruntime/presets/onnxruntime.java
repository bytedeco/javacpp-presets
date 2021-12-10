/*
 * Copyright (C) 2019-2021 Samuel Audet, Alexander Merritt
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
            compiler = "cpp11",
            define = {"GENERIC_EXCEPTION_CLASS Ort::Exception", "GENERIC_EXCEPTION_TOSTRING what()"},
            include = {
                "onnxruntime/core/session/onnxruntime_c_api.h",
                "onnxruntime/core/session/onnxruntime_cxx_api.h",
                "onnxruntime/core/providers/cpu/cpu_provider_factory.h",
//                "onnxruntime/core/providers/cuda/cuda_provider_factory.h",
                "onnxruntime/core/providers/dnnl/dnnl_provider_factory.h",
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
            },
            link = {"onnxruntime_providers_shared", "onnxruntime@.1.10.0"}
        ),
        @Platform(
            value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
            link = {"onnxruntime_providers_shared", "onnxruntime@.1.10.0", "onnxruntime_providers_dnnl"}
        ),
        @Platform(
            value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
            extension = "-gpu",
            link = {"onnxruntime_providers_shared", "onnxruntime@.1.10.0", "onnxruntime_providers_dnnl", "onnxruntime_providers_cuda"}
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
        String[] libs = {"cudart", "cublasLt", "cublas", "cufft", "curand", "cudnn",
                         "cudnn_ops_infer", "cudnn_ops_train", "cudnn_adv_infer",
                         "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.8" : lib.equals("cufft") || lib.equals("curand") ? "@.10" : lib.equals("cudart") ? "@.11.0" : "@.11";
            } else if (platform.startsWith("windows")) {
                lib += lib.startsWith("cudnn") ? "64_8" : lib.equals("cufft") || lib.equals("curand") ? "64_10" : lib.equals("cudart") ? "64_110" : "64_11";
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
        infoMap.put(new Info("ORTCHAR_T").cppText("").cppTypes().cast().pointerTypes("Pointer"))
               .put(new Info("ORT_EXPORT", "ORT_API_CALL", "NO_EXCEPTION", "ORT_ALL_ARGS_NONNULL", "OrtCustomOpApi").cppTypes().annotations())
               .put(new Info("ORT_API_MANUAL_INIT").define(false))
               .put(new Info("USE_CUDA", "USE_DNNL").define(true))
               .put(new Info("Ort::stub_api", "Ort::Global<T>::api_", "std::nullptr_t", "Ort::Env::s_api").skip())
               .put(new Info("std::string").annotations("@Cast({\"char*\", \"std::string&&\"}) @StdString").valueTypes("BytePointer", "String").pointerTypes("BytePointer"))
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<Ort::Value>").valueTypes("@StdMove ValueVector").pointerTypes("ValueVector").define())
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
               .put(new Info("Ort::Value::GetTensorMutableData<float>").javaNames("GetTensorMutableDataFloat"))
               .put(new Info("Ort::Value::GetTensorMutableData<double>").javaNames("GetTensorMutableDataDouble"))
               .put(new Info("Ort::Value::GetTensorMutableData<int8_t>").javaNames("GetTensorMutableDataByte"))
               .put(new Info("Ort::Value::GetTensorMutableData<int16_t>").javaNames("GetTensorMutableDataShort"))
               .put(new Info("Ort::Value::GetTensorMutableData<int32_t>").javaNames("GetTensorMutableDataInt"))
               .put(new Info("Ort::Value::GetTensorMutableData<int64_t>").javaNames("GetTensorMutableDataLong"))
               .put(new Info("Ort::Value::GetTensorMutableData<uint8_t>").javaNames("GetTensorMutableDataUByte"))
               .put(new Info("Ort::Value::GetTensorMutableData<uint16_t>").javaNames("GetTensorMutableDataUShort"))
               .put(new Info("Ort::Value::GetTensorMutableData<uint32_t>").javaNames("GetTensorMutableDataUInt"))
               .put(new Info("Ort::Value::GetTensorMutableData<uint64_t>").javaNames("GetTensorMutableDataULong"))
               .put(new Info("Ort::Value::GetTensorMutableData<bool>").javaNames("GetTensorMutableDataBool"))
               .put(new Info("Ort::Unowned<const Ort::MemoryInfo>").pointerTypes("UnownedMemoryInfo").purify())
               .put(new Info("Ort::Unowned<Ort::TensorTypeAndShapeInfo>").pointerTypes("UnownedTensorTypeAndShapeInfo").purify())
               .put(new Info("Ort::Unowned<Ort::SequenceTypeInfo>").pointerTypes("UnownedSequenceTypeInfo").purify())
               .put(new Info("Ort::Unowned<Ort::MapTypeInfo>").pointerTypes("UnownedMapTypeInfo").purify())
               .put(new Info("Ort::MemoryAllocation").purify())
               .put(new Info("Ort::MemoryAllocation::operator =").skip())
               .put(new Info("Ort::RunOptions::GetRunLogSeverityLevel").skip())
               .put(new Info("Ort::Exception").pointerTypes("OrtException"))
               .put(new Info("Ort::Base<OrtArenaCfg>").pointerTypes("BaseArenaCfg"))
               .put(new Info("Ort::Base<OrtAllocator>").pointerTypes("BaseAllocator"))
               .put(new Info("Ort::Base<OrtIoBinding>").pointerTypes("BaseIoBinding"))
               .put(new Info("Ort::Base<OrtMemoryInfo>", "Ort::BaseMemoryInfo<Ort::Base<OrtMemoryInfo> >",
                                                         "Ort::BaseMemoryInfo<Ort::Base<const OrtMemoryInfo> >").pointerTypes("BaseMemoryInfo"))
               .put(new Info("Ort::Base<OrtModelMetadata>").pointerTypes("BaseModelMetadata"))
               .put(new Info("Ort::Base<OrtCustomOpDomain>").pointerTypes("BaseCustomOpDomain"))
               .put(new Info("Ort::Base<OrtEnv>").pointerTypes("BaseEnv"))
               .put(new Info("Ort::Base<OrtRunOptions>").pointerTypes("BaseRunOptions"))
               .put(new Info("Ort::Base<OrtSession>").pointerTypes("BaseSession"))
               .put(new Info("Ort::Base<OrtSessionOptions>").pointerTypes("BaseSessionOptions"))
               .put(new Info("Ort::Base<OrtTensorTypeAndShapeInfo>").pointerTypes("BaseTensorTypeAndShapeInfo"))
               .put(new Info("Ort::Base<OrtSequenceTypeInfo>").pointerTypes("BaseSequenceTypeInfo"))
               .put(new Info("Ort::Base<OrtMapTypeInfo>").pointerTypes("BaseMapTypeInfo"))
               .put(new Info("Ort::Base<OrtTypeInfo>").pointerTypes("BaseTypeInfo"))
               .put(new Info("Ort::Base<OrtValue>").pointerTypes("BaseValue"))
               .put(new Info("OrtSessionOptionsAppendExecutionProvider_CUDA").annotations("@Platform(extension=\"-gpu\")").javaNames("OrtSessionOptionsAppendExecutionProvider_CUDA"));
    }
}
