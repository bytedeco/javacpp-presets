/*
 * Copyright (C) 2019 Samuel Audet
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
            link = "onnxruntime@.1.1.0",
            preload = {"iomp5", "dnnl@.1"},
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
	.put(new Info("TypeInfo::GetTensorTypeAndShapeInfo()", "std::string&&", "Ort::Value::put").skip())
	.put(new Info("Ort::Value").purify(true).base("BasedValue"))
	.put(new Info("Ort::Env").base("BasedEnv"))
	.put(new Info("Ort::CustomOpDomain").base("BasedCustomOpDomain"))
	.put(new Info("Ort::RunOptions").base("BasedRunOptions"))
	.put(new Info("Ort::Session").base("BasedSession"))
	.put(new Info("Ort::SessionOptions").base("BasedSessionOptions"))
	.put(new Info("Ort::TypeInfo").base("BasedTypeInfo"))
	.put(new Info("Ort::MemoryInfo").base("BasedMemoryInfo"))
	.put(new Info("Ort::TensorTypeAndShapeInfo").base("BasedTensorTypeAndShapeInfo"))
	.put(new Info("Ort::Value::operator =").skip())
	.put(new Info("Ort::Base<OrtValue>", "Base<OrtValue>").pointerTypes("BasedValue"))
	.put(new Info("Ort::Base<OrtMemoryInfo>", "Base<OrtMemoryInfo>").pointerTypes("BasedMemoryInfo"))
	.put(new Info("Ort::Base<OrtEnv>", "Base<OrtEnv>").pointerTypes("BasedEnv"))
	.put(new Info("Ort::Base<OrtCustomOpDomain>", "Base<OrtCustomOpDomain>").pointerTypes("BasedCustomOpDomain"))
	.put(new Info("Ort::Base<OrtRunOptions>", "Base<OrtRunOptions>").pointerTypes("BasedRunOptions"))
	.put(new Info("Ort::Base<OrtSession>", "Base<OrtSession>").pointerTypes("BasedSession"))
	.put(new Info("Ort::Base<OrtSessionOptions>", "Base<OrtSessionOptions>").pointerTypes("BasedSessionOptions"))
	.put(new Info("Ort::Base<OrtTensorTypeAndShapeInfo>", "Base<OrtTensorTypeAndShapeInfo>").pointerTypes("BasedTensorTypeAndShapeInfo"))
	.put(new Info("Ort::Base<OrtTypeInfo>", "Base<OrtTypeInfo>").pointerTypes("BasedTypeInfo"))
	.put(new Info("Ort::Value::GetTensorMutableData<float>").javaNames("GetTensorMutableDataFloat"))
	.put(new Info("const std::vector<Ort::Value>", "std::vector<Ort::Value>").pointerTypes("ValueVector").define())
        .put(new Info("Ort::Exception").javaNames("OrtException"));
    }
}
