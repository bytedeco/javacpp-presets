/*
 * Copyright (C) 2019-2020 Samuel Audet, Alexander Merritt
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
            include = {
                "onnxruntime/core/session/onnxruntime_c_api.h",
                "onnxruntime/core/session/onnxruntime_cxx_api.h",
                "onnxruntime/core/providers/dnnl/dnnl_provider_factory.h"
            },
            link = "onnxruntime@.1.1.1"
        ),
    },
    target = "org.bytedeco.onnxruntime",
    global = "org.bytedeco.onnxruntime.global.onnxruntime"
)
public class onnxruntime implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "onnxruntime"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("ORTCHAR_T").cppText("").cppTypes().cast().pointerTypes("Pointer"))
               .put(new Info("ORT_EXPORT", "ORT_API_CALL", "NO_EXCEPTION", "ORT_ALL_ARGS_NONNULL", "OrtCustomOpApi").cppTypes().annotations())
               .put(new Info("Ort::stub_api", "Ort::Global<T>::api_", "std::nullptr_t", "Ort::Env::s_api").skip())
               .put(new Info("std::string").annotations("@Cast({\"char*\", \"std::string&&\"}) @StdString").valueTypes("BytePointer", "String").pointerTypes("BytePointer"))
               .put(new Info("const std::vector<Ort::Value>", "std::vector<Ort::Value>").pointerTypes("ValueVector").define())
               .put(new Info("Ort::Exception").pointerTypes("OrtException"))
               .put(new Info("Ort::Value(Ort::Value)", "Ort::Value::operator =(Ort::Value)", "Ort::RunOptions::GetRunLogSeverityLevel").skip())
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
               .put(new Info("Ort::TensorTypeAndShapeInfo", "Ort::Unowned<Ort::TensorTypeAndShapeInfo>").pointerTypes("TensorTypeAndShapeInfo"))
               .put(new Info("Ort::Base<OrtMemoryInfo>").pointerTypes("BaseMemoryInfo"))
               .put(new Info("Ort::Base<OrtCustomOpDomain>").pointerTypes("BaseCustomOpDomain"))
               .put(new Info("Ort::Base<OrtEnv>").pointerTypes("BaseEnv"))
               .put(new Info("Ort::Base<OrtRunOptions>").pointerTypes("BaseRunOptions"))
               .put(new Info("Ort::Base<OrtSession>").pointerTypes("BaseSession"))
               .put(new Info("Ort::Base<OrtSessionOptions>").pointerTypes("BaseSessionOptions"))
               .put(new Info("Ort::Base<OrtTensorTypeAndShapeInfo>").pointerTypes("BaseTensorTypeAndShapeInfo"))
               .put(new Info("Ort::Base<OrtTypeInfo>").pointerTypes("BaseTypeInfo"))
               .put(new Info("Ort::Base<OrtValue>").pointerTypes("BaseValue"));
    }
}
