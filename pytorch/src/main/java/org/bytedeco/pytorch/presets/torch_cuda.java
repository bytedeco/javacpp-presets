/*
 * Copyright (C) 2020-2023 Samuel Audet, Eduardo Gonzalez
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
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import java.util.List;

/**
 * @author Samuel Audet, Hervé Guillemet
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
            link = "cudart",
            linkpath = {
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/",
                "/usr/local/cuda-12.1/lib64/",
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

            .put(new Info("c10::optional<c10::cuda::CUDAStream>").pointerTypes("CUDAStreamOptional").define())

            //// Already defined in main torch
            .put(new Info("c10::Stream").pointerTypes("Stream"))
            .put(new Info("c10::optional<c10::Stream>").pointerTypes("StreamOptional"))
            .put(new Info("c10::optional<c10::Device>").pointerTypes("DeviceOptional"))
            .put(new Info("c10::Device").pointerTypes("Device"))
            .put(new Info("c10::impl::PyInterpreter").pointerTypes("PyInterpreter"))
            .put(new Info("std::tuple<int,int>").pointerTypes("T_IntInt_T"))
            .put(new Info("c10::optional<c10::DeviceIndex>").pointerTypes("ByteOptional"))
            .put(new Info("c10::IntArrayRef", "at::IntArrayRef").pointerTypes("LongArrayRef"))

            .put(new Info("c10::DeviceIndex").valueTypes("byte"))
            .put(new Info("c10::StreamId").valueTypes("long"))
            .put(new Info("c10::cuda::CaptureStatus").valueTypes("int").cast().skip()) // Enum doesn't parse
            .put(new Info("std::pair<std::vector<c10::cuda::DeviceAssertionsData>,std::vector<c10::cuda::CUDAKernelLaunchInfo> >").pointerTypes("DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair").define())
            .put(new Info("std::vector<c10::cuda::DeviceAssertionsData>").pointerTypes("DeviceAssertionsDataVector").define())
            .put(new Info("std::vector<c10::cuda::CUDAKernelLaunchInfo>").pointerTypes("CUDAKernelLaunchInfoVector").define())
            .put(new Info("std::atomic<const c10::impl::PyInterpreter*>").cast().pointerTypes("PyInterpreter"))
            .put(new Info("c10::CuDNNError").purify())
            .put(new Info("c10::impl::GPUTrace::gpuTraceState").skip())
            .put(new Info("at::native::RNNDescriptor::dropout_desc_").skip())
            .put(new Info("at::native::operator <<(std::ostream&, at::native::TensorDescriptor&)",
                "at::native::operator <<(std::ostream&, at::native::FilterDescriptor&)",
                "at::native::cudnnTypeToString", "at::native::getCudnnDataType", "at::native::cudnn_version",
                "c10::cuda::c10_retrieve_device_side_assertion_info").skip())

            .put(new Info(
                "at::native::Descriptor<cudnnActivationStruct,cudnnCreateActivationDescriptor&,cudnnDestroyActivationDescriptor&>",
                "at::native::Descriptor<cudnnConvolutionStruct,cudnnCreateConvolutionDescriptor&,cudnnDestroyConvolutionDescriptor&>",
                "at::native::Descriptor<cudnnCTCLossStruct,cudnnCreateCTCLossDescriptor&,cudnnDestroyCTCLossDescriptor&>",
                "at::native::Descriptor<cudnnDropoutStruct,cudnnCreateDropoutDescriptor&,cudnnDestroyDropoutDescriptor&>",
                "at::native::Descriptor<cudnnFilterStruct,cudnnCreateFilterDescriptor&,cudnnDestroyFilterDescriptor&>",
                "at::native::Descriptor<cudnnRNNStruct,cudnnCreateRNNDescriptor&,cudnnDestroyRNNDescriptor&>",
                "at::native::Descriptor<cudnnSpatialTransformerStruct,cudnnCreateSpatialTransformerDescriptor&,cudnnDestroySpatialTransformerDescriptor&>",
                "at::native::Descriptor<cudnnTensorStruct,cudnnCreateTensorDescriptor&,cudnnDestroyTensorDescriptor&>",

                "std::hash<c10::cuda::CUDAStream>"

            ).cast().pointerTypes("Pointer"))

            //// CUDA types
            .put(new Info( // Struct
                "cudaDeviceProp"
            ).pointerTypes("Pointer"))
            .put(new Info( // Pointers to opaque structs
                "cudaStream_t", "cusparseHandle_t", "cublasHandle_t", "cusolverDnHandle_t", "cudnnHandle_t"
            ).valueTypes("Pointer").cast())
            .put(new Info( // Enums
                "cudnnActivationMode_t", "cudnnLossNormalizationMode_t", "cudnnRNNInputMode_t",
                "cudnnDirectionMode_t", "cudnnRNNMode_t", "cudaStreamCaptureMode", "cudnnDataType_t", "cudnnNanPropagation_t",
                "cusparseStatus_t", "cusolverStatus_t", "cudnnRNNAlgo_t", "cudnnNanPropagation_t", "cublasStatus_t"
            ).valueTypes("int").cast())
        ;

        new torch.ArrayInfo("CUDAStream").elementTypes("c10::cuda::CUDAStream").mapArrayRef(infoMap);
    }
}