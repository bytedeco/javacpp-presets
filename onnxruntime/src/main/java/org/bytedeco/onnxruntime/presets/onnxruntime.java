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

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    value = {
        @Platform(
            value = {"linux", "macosx"},
            compiler = "cpp11",
            include = {
                "onnxruntime/core/session/onnxruntime_c_api.h",
                "onnxruntime/core/session/onnxruntime_cxx_api.h",
                "onnxruntime/core/providers/dnnl/dnnl_provider_factory.h"
            },
            link = "onnxruntime@.1.1.1",
            preload = {"gomp@.1##", "iomp5##", "dnnl@.1##"},
            preloadresource = "/org/bytedeco/dnnl/"
        ),
    },
    target = "org.bytedeco.onnxruntime",
    global = "org.bytedeco.onnxruntime.global.onnxruntime"
)
public class onnxruntime implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "onnxruntime"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("ORTCHAR_T", "ORT_EXPORT", "ORT_API_CALL", "NO_EXCEPTION", "ORT_ALL_ARGS_NONNULL", "OrtCustomOpApi").cppTypes().annotations())
               .put(new Info("Ort::stub_api", "Ort::Global<T>::api_", "std::nullptr_t", "Ort::Env::s_api").skip())
               .put(new Info("std::string").annotations("@Cast({\"char*\", \"std::string&&\"}) @StdString").valueTypes("BytePointer", "String").pointerTypes("BytePointer"))
               .put(new Info("const std::vector<Ort::Value>", "std::vector<Ort::Value>").pointerTypes("ValueVector").define())
               .put(new Info("Ort::Exception").pointerTypes("OrtException"))
               .put(new Info("Ort::Value(Ort::Value)", "Ort::Value::operator =(Ort::Value)").skip())
               .put(new Info("Ort::Value::CreateTensor<jbyte>").javaNames("CreateTensorByte"))
               .put(new Info("Ort::Value::CreateTensor<jshort>").javaNames("CreateTensorShort"))
               .put(new Info("Ort::Value::CreateTensor<jint>").javaNames("CreateTensorInt"))
               .put(new Info("Ort::Value::CreateTensor<float>").javaNames("CreateTensorFloat"))
               .put(new Info("Ort::Value::CreateTensor<double>").javaNames("CreateTensorDouble"))
               .put(new Info("Ort::Value::CreateTensor<jboolean>").javaNames("CreateTensorBoolean"))
               .put(new Info("Ort::Value::CreateTensor<jchar>").javaNames("CreateTensorChar"))
               .put(new Info("Ort::Value::GetTensorMutableData<jbyte>").javaNames("GetTensorMutableDataByte"))
               .put(new Info("Ort::Value::GetTensorMutableData<jshort>").javaNames("GetTensorMutableDataShort"))
               .put(new Info("Ort::Value::GetTensorMutableData<jint>").javaNames("GetTensorMutableDataInt"))
               .put(new Info("Ort::Value::GetTensorMutableData<float>").javaNames("GetTensorMutableDataFloat"))
               .put(new Info("Ort::Value::GetTensorMutableData<double>").javaNames("GetTensorMutableDataDouble"))
               .put(new Info("Ort::Value::GetTensorMutableData<jboolean>").javaNames("GetTensorMutableDataBoolean"))
               .put(new Info("Ort::Value::GetTensorMutableData<jchar>").javaNames("GetTensorMutableDataChar"))
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
