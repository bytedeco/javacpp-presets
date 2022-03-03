/*
 * Copyright (C) 2018-2022 Samuel Audet
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
            value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "windows-x86_64"},
            compiler = "cpp11",
            include = {"NvInferVersion.h", "NvInferRuntimeCommon.h", "NvInferLegacyDims.h", "NvInferRuntime.h", "NvInfer.h", "NvInferImpl.h", "NvUtils.h"},
            link = "nvinfer@.8.2.3",
            preload = "nvinfer_builder_resource@.8.2.3"
        ),
        @Platform(
            value = "linux-arm64",
            includepath = {"/usr/include/aarch64-linux-gnu/", "/usr/local/tensorrt/include/"},
            linkpath = {"/usr/lib/aarch64-linux-gnu/", "/usr/local/tensorrt/lib/"}
        ),
        @Platform(
            value = "linux-ppc64le",
            includepath = {"/usr/include/powerpc64le-linux-gnu/", "/usr/local/tensorrt/include/"},
            linkpath = {"/usr/lib/powerpc64le-linux-gnu/", "/usr/local/tensorrt/lib/"}
        ),
        @Platform(
            value = "linux-x86_64",
            includepath = {"/usr/include/x86_64-linux-gnu/", "/usr/local/tensorrt/include/"},
            linkpath = {"/usr/lib/x86_64-linux-gnu/", "/usr/local/tensorrt/lib/"}
        ),
        @Platform(
            value = "windows-x86_64",
            includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/include",
            linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/lib/"
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
        if (!Loader.isLoadLibraries()) {
            return;
        }
        int i = 0;
        String[] libs = {"cudart", "cublasLt", "cublas", "cudnn", "nvrtc",
                         "cudnn_ops_infer", "cudnn_ops_train", "cudnn_adv_infer",
                         "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.8" : lib.equals("cudart") ? "@.11.0" : lib.equals("nvrtc") ? "@.11.2" : "@.11";
            } else if (platform.startsWith("windows")) {
                lib += lib.startsWith("cudnn") ? "64_8" : lib.equals("cudart") ? "64_110" : lib.equals("nvrtc") ? "64_112_0" : "64_11";
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
        infoMap.put(new Info().enumerate())
               .put(new Info("NV_TENSORRT_FINAL", "_TENSORRT_FINAL", "_TENSORRT_OVERRIDE", "TENSORRTAPI").cppTypes().annotations())
               .put(new Info("NV_TENSORRT_VERSION").translate(false))

               .put(new Info("TRT_DEPRECATED").cppText("#define TRT_DEPRECATED deprecated").cppTypes())
               .put(new Info("TRT_DEPRECATED_API").cppText("#define TRT_DEPRECATED_API deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("const char", "nvinfer1::AsciiChar").pointerTypes("String", "@Cast(\"const char*\") BytePointer"))
               .put(new Info("nvinfer1::IErrorRecorder::ErrorDesc").valueTypes("String", "@Cast(\"const char*\") BytePointer"))
               .put(new Info("nvinfer1::PluginFormat").cast().valueTypes("TensorFormat", "int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("nvinfer1::EnumMax").skip())
               .put(new Info("nvinfer1::Weights::values").javaText("public native @Const Pointer values(); public native Weights values(Pointer values);"))
               .put(new Info("nvinfer1::IDimensionExpr", "nvinfer1::IExprBuilder", "nvinfer1::IOptimizationProfile", "nvinfer1::ITensor", "nvinfer1::ILayer",
                             "nvinfer1::IConvolutionLayer", "nvinfer1::IFullyConnectedLayer", "nvinfer1::IActivationLayer", "nvinfer1::IPoolingLayer",
                             "nvinfer1::ILRNLayer", "nvinfer1::IScaleLayer", "nvinfer1::IPluginV2Layer", "nvinfer1::IUnaryLayer", "nvinfer1::IReduceLayer",
                             "nvinfer1::IPaddingLayer", "nvinfer1::IRaggedSoftMaxLayer", "nvinfer1::IIdentityLayer", "nvinfer1::ISoftMaxLayer",
                             "nvinfer1::IConcatenationLayer", "nvinfer1::IDeconvolutionLayer", "nvinfer1::IElementWiseLayer", "nvinfer1::IGatherLayer",
                             "nvinfer1::IRNNv2Layer", "nvinfer1::IIteratorLayer", "nvinfer1::IParametricReLULayer", "nvinfer1::IShuffleLayer",
                             "nvinfer1::ISliceLayer", "nvinfer1::IShapeLayer", "nvinfer1::ITopKLayer", "nvinfer1::IMatrixMultiplyLayer", "nvinfer1::ISelectLayer",
                             "nvinfer1::IConstantLayer", "nvinfer1::IResizeLayer", "nvinfer1::ILoop", "nvinfer1::ILoopBoundaryLayer", "nvinfer1::ILoopOutputLayer",
                             "nvinfer1::IRecurrenceLayer", "nvinfer1::ITripLimitLayer", "nvinfer1::IFillLayer", "nvinfer1::IQuantizeLayer", "nvinfer1::IDequantizeLayer",
                             "nvinfer1::IAssertionLayer", "nvinfer1::IConditionLayer", "nvinfer1::IEinsumLayer", "nvinfer1::IIfConditional",
                             "nvinfer1::IIfConditionalBoundaryLayer", "nvinfer1::IIfConditionalInputLayer", "nvinfer1::IIfConditionalOutputLayer", "nvinfer1::IScatterLayer",
                             "nvinfer1::IAlgorithmIOInfo", "nvinfer1::IAlgorithmVariant", "nvinfer1::IAlgorithmContext", "nvinfer1::IAlgorithm").purify())
               .put(new Info("nvinfer1::IGpuAllocator::free").javaNames("_free"))
               .put(new Info("nvinfer1::IProfiler", "nvinfer1::ILogger", "nvinfer1::IInt8Calibrator", "nvinfer1::IInt8EntropyCalibrator",
                             "nvinfer1::IInt8EntropyCalibrator2", "nvinfer1::IInt8MinMaxCalibrator", "nvinfer1::IInt8LegacyCalibrator").virtualize())
               .put(new Info("nvinfer1::IPluginRegistry::getPluginCreatorList").javaText(
                             "public native @Cast(\"nvinfer1::IPluginCreator*const*\") PointerPointer getPluginCreatorList(IntPointer numCreators);"))
        ;
    }
}
