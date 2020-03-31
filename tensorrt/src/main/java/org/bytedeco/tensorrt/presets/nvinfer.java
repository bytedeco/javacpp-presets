/*
 * Copyright (C) 2018-2019 Samuel Audet
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

package org.bytedeco.tensorrt.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.cuda.presets.cudart;
import org.bytedeco.cuda.presets.cublas;
import org.bytedeco.cuda.presets.cudnn;
import org.bytedeco.cuda.presets.nvrtc;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = {cublas.class, cudnn.class, nvrtc.class},
    value = {
        @Platform(
            value = "linux-x86_64",
            compiler = "cpp11",
            include = {"NvInferVersion.h", "NvInferRuntimeCommon.h", "NvInferRuntime.h", "NvInfer.h", "NvUtils.h"},
            includepath = {"/usr/include/x86_64-linux-gnu/", "/usr/local/tensorrt/include/"},
            link = "nvinfer@.7",
            preload = "myelin@.1",
            linkpath = {"/usr/lib/x86_64-linux-gnu/", "/usr/local/tensorrt/lib/"}
        ),
        @Platform(
            value = "windows-x86_64",
            compiler = "cpp11",
            include = {"NvInferVersion.h", "NvInferRuntimeCommon.h", "NvInferRuntime.h", "NvInfer.h", "NvUtils.h"},
            includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-7.0.0.11/include",
            link = "nvinfer",
            preload = "myelin64_1",
            linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-7.0.0.11/lib/"
        )
    },
    target = "org.bytedeco.tensorrt.nvinfer",
    global = "org.bytedeco.tensorrt.global.nvinfer"
)
public class nvinfer implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "tensorrt"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || !(platform.equals("linux-x86_64") || platform.equals("windows-x86_64"))) {
            return;
        }
        int i = 0;
        String[] libs = {"cudart", "cublasLt", "cublas", "cudnn"};
        for (String lib : libs) {
            lib += lib.equals("cudnn") ? "@.7" : lib.equals("cudart") ? "@.10.2" : "@.10";
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info().enumerate())
               .put(new Info("NV_TENSORRT_FINAL", "_TENSORRT_FINAL", "_TENSORRT_OVERRIDE", "TENSORRTAPI").cppTypes().annotations())

               .put(new Info("TRT_DEPRECATED").cppText("#define TRT_DEPRECATED deprecated").cppTypes())
               .put(new Info("TRT_DEPRECATED_API").cppText("#define TRT_DEPRECATED_API deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("const char").pointerTypes("String", "@Cast(\"const char*\") BytePointer"))
               .put(new Info("nvinfer1::IErrorRecorder::ErrorDesc").valueTypes("String", "@Cast(\"const char*\") BytePointer"))
               .put(new Info("nvinfer1::PluginFormat").cast().valueTypes("TensorFormat", "int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("nvinfer1::EnumMax").skip())
               .put(new Info("nvinfer1::Weights::values").javaText("public native @Const Pointer values(); public native Weights values(Pointer values);"))
               .put(new Info("nvinfer1::IRaggedSoftMaxLayer", "nvinfer1::IIdentityLayer", "nvinfer1::ISoftMaxLayer",
                             "nvinfer1::IConcatenationLayer", "nvinfer1::IInt8EntropyCalibrator", "nvinfer1::IInt8EntropyCalibrator2",
                             "nvinfer1::IInt8MinMaxCalibrator", "nvinfer1::IParametricReLULayer", "nvinfer1::IShapeLayer", "nvinfer1::ISelectLayer").purify())
               .put(new Info("nvinfer1::IGpuAllocator::free").javaNames("_free"))
               .put(new Info("nvinfer1::IProfiler", "nvinfer1::ILogger").purify().virtualize())
               .put(new Info("nvinfer1::IPluginRegistry::getPluginCreatorList").javaText(
                             "public native @Cast(\"nvinfer1::IPluginCreator*const*\") PointerPointer getPluginCreatorList(IntPointer numCreators);"))
               .put(new Info("nvinfer1::IProfiler::~IProfiler").javaText("\n"
                     + "/** Default native constructor. */\n"
                     + "public IProfiler() { super((Pointer)null); allocate(); }\n"
                     + "private native void allocate();\n"))
               .put(new Info("nvinfer1::ILogger::~ILogger").javaText("\n"
                     + "/** Default native constructor. */\n"
                     + "public ILogger() { super((Pointer)null); allocate(); }\n"
                     + "private native void allocate();\n"));
    }
}
