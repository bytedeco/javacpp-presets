/*
 * Copyright (C) 2021-2024 Samuel Audet
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

package org.bytedeco.tensorflowlite.presets;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
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
            value = {"android", "linux", "macosx", "windows"},
            compiler = "cpp17",
            define = "UNIQUE_PTR_NAMESPACE std",
            include = {
//                "flatbuffers/base.h",
//                "flatbuffers/flatbuffers.h",
//                "tensorflow/lite/schema/schema_generated.h",
                "tensorflow/compiler/mlir/lite/allocation.h",
                "tensorflow/compiler/mlir/lite/core/api/verifier.h",
                "tensorflow/compiler/mlir/lite/core/api/error_reporter.h",
                "tensorflow/compiler/mlir/lite/core/model_builder_base.h",
                "tensorflow/compiler/mlir/lite/utils/control_edges.h",
                "tensorflow/compiler/mlir/lite/experimental/remat/metadata_util.h",
                "tensorflow/compiler/mlir/lite/core/c/tflite_types.h",
                "tensorflow/lite/builtin_ops.h",
                "tensorflow/lite/c/c_api_types.h",
                "tensorflow/lite/core/c/c_api_types.h",
                "tensorflow/lite/c/c_api.h",
                "tensorflow/lite/core/c/c_api.h",
                "tensorflow/lite/core/c/operator.h",
                "tensorflow/lite/c/common.h",
                "tensorflow/lite/core/c/common.h",
                "tensorflow/lite/c/c_api_experimental.h",
                "tensorflow/lite/core/c/c_api_experimental.h",
                "tensorflow/lite/core/api/error_reporter.h",
                "tensorflow/lite/core/api/op_resolver.h",
                "tensorflow/lite/core/api/profiler.h",
                "tensorflow/lite/core/api/verifier.h",
                "tensorflow/lite/experimental/resource/initialization_status.h",
                "tensorflow/lite/experimental/resource/resource_base.h",
                "tensorflow/lite/allocation.h",
                "tensorflow/lite/stderr_reporter.h",
                "tensorflow/lite/graph_info.h",
                "tensorflow/lite/interpreter_options.h",
                "tensorflow/lite/memory_planner.h",
                "tensorflow/lite/util.h",
//                "tensorflow/lite/array.h",
                "tensorflow/lite/core/macros.h",
                "tensorflow/lite/core/subgraph.h",
                "tensorflow/lite/external_cpu_backend_context.h",
                "tensorflow/lite/portable_type_to_tflitetype.h",
                "tensorflow/lite/profiling/root_profiler.h",
                "tensorflow/lite/signature_runner.h",
                "tensorflow/lite/core/signature_runner.h",
                "tensorflow/lite/type_to_tflitetype.h",
                "tensorflow/lite/string_type.h",
                "tensorflow/lite/mutable_op_resolver.h",
                "tensorflow/lite/interpreter.h",
                "tensorflow/lite/core/interpreter.h",
                "tensorflow/lite/model_builder.h",
                "tensorflow/lite/core/model_builder.h",
                "tensorflow/lite/interpreter_builder.h",
                "tensorflow/lite/core/interpreter_builder.h",
                "tensorflow/lite/model.h",
                "tensorflow/lite/kernels/register.h",
                "tensorflow/lite/core/kernels/register.h",
                "tensorflow/lite/optional_debug_tools.h",
//                "tensorflow/lite/core/async/c/types.h",
//                "tensorflow/lite/core/async/interop/c/types.h",
//                "tensorflow/lite/core/async/interop/c/attribute_map.h",
//                "tensorflow/lite/core/async/async_signature_runner.h",
//                "tensorflow/lite/core/async/c/async_signature_runner.h",
                "tensorflow/lite/profiling/telemetry/c/profiler.h",
                "tensorflow/lite/profiling/telemetry/c/telemetry_setting.h",
                "tensorflow/lite/profiling/telemetry/telemetry_status.h",
                "tensorflow/lite/profiling/telemetry/profiler.h",
            }
//            link = "tensorflowlite_c"
        ),
    },
    target = "org.bytedeco.tensorflowlite",
    global = "org.bytedeco.tensorflowlite.global.tensorflowlite")
public class tensorflowlite implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "tensorflow-lite"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("TFLITE_ATTRIBUTE_WEAK", "TFL_CAPI_EXPORT", "TFLITE_NOINLINE").cppTypes().annotations())
               .put(new Info("DOYXGEN_SKIP").define(true))
               .put(new Info("FLATBUFFERS_LITTLEENDIAN == 0").define(false))
               .put(new Info("TfLiteIntArray", "TfLiteFloatArray").purify())
               .put(new Info("tflite::ops::builtin::BuiltinOpResolver").pointerTypes("BuiltinOpResolver"))
               .put(new Info("tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates").pointerTypes("BuiltinOpResolverWithoutDefaultDelegates"))
               .put(new Info("auto", "std::initializer_list", "tflite::BuildTfLiteTensor", "tflite::typeToTfLiteType", "TfLiteContext::ReportError", "tflite::MMAPAllocation",
                             "tflite::OpResolver::GetOpaqueDelegateCreators", "tflite::MutableOpResolver::GetOpaqueDelegateCreators", "IntArrayUniquePtr",
                             "tflite::InterpreterBuilder::PreserveAllTensorsExperimental", "tflite::async::AsyncSignatureRunner", "TfLiteAsyncKernel").skip())
               .put(new Info("kTfLiteInplaceOpMaxValue", "kTfLiteNoBufferIdentifier").translate(false))
               .put(new Info("tflite::Model", "tflite::ModelT", "tflite::OperatorCode", "tflite::OpResolver::TfLiteDelegateCreators",
                             "tflite::internal::SignatureDef").cast().pointerTypes("Pointer"))
               .put(new Info("tflite::Subgraph").valueTypes("@StdMove Subgraph").pointerTypes("Subgraph"))
               .put(new Info("std::int32_t", "std::uint32_t", "tflite::BuiltinOperator", "TfLiteBuiltinOperator",
                             "tflite::profiling::RootProfiler::EventType").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("std::string").annotations("@StdString").valueTypes("String", "BytePointer").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::vector<const std::string*>").pointerTypes("StringVector").define())
               .put(new Info("std::map<std::string,uint32_t>").pointerTypes("StringIntMap").define())
               .put(new Info("std::map<std::string,std::string>").pointerTypes("StringStringMap").define())
               .put(new Info("std::unordered_map<size_t,size_t>").pointerTypes("SizeTSizeTMap").define())
               .put(new Info("std::unique_ptr<TfLiteDelegate,void(*)(TfLiteDelegate*)>").annotations("@UniquePtr(\"TfLiteDelegate,void(*)(TfLiteDelegate*)\")").pointerTypes("TfLiteDelegate"))
               .put(new Info("std::unique_ptr<TfLiteIntArray,tflite::TfLiteArrayDeleter>").annotations("@UniquePtr(\"TfLiteIntArray,tflite::TfLiteArrayDeleter\")").pointerTypes("TfLiteIntArray"))
               .put(new Info("std::unique_ptr<tflite::Subgraph>").annotations("@UniquePtr").pointerTypes("Subgraph")
                                                                 .valueTypes("@Cast({\"\", \"std::unique_ptr<tflite::Subgraph>&&\"}) Subgraph"))
               .put(new Info("std::unique_ptr<tflite::resource::ResourceBase>").annotations("@UniquePtr").pointerTypes("ResourceBase")
                                                                               .valueTypes("@Cast({\"\", \"std::unique_ptr<tflite::resource::ResourceBase>&&\"}) ResourceBase"))
               .put(new Info("std::pair<int32_t,int32_t>", "tflite::ControlEdge").cast().pointerTypes("IntIntPair").define())
               .put(new Info("std::pair<TfLiteNode,TfLiteRegistration>").pointerTypes("RegistrationNodePair").define())
               .put(new Info("std::vector<tflite::NodeSubset>").pointerTypes("NodeSubsetVector").define())
               .put(new Info("std::vector<std::unique_ptr<tflite::Subgraph> >").valueTypes("@StdMove SubgraphVector").pointerTypes("SubgraphVector").define())
               .put(new Info("std::vector<std::pair<int32_t,int32_t> >", "tflite::ControlEdges").cast().pointerTypes("IntIntPairVector").define())
               .put(new Info("std::vector<std::vector<std::pair<int32_t,int32_t> > >", "tflite::ModelControlDependencies").cast().pointerTypes("IntIntPairVectorVector").define())
               .put(new Info("std::vector<std::pair<TfLiteNode,TfLiteRegistration> >").valueTypes("@StdMove RegistrationNodePairVector").pointerTypes("RegistrationNodePairVector").define())
               .put(new Info("const std::vector<std::unique_ptr<TfLiteDelegate,void(*)(TfLiteDelegate*)> >", "tflite::OpResolver::TfLiteDelegatePtrVector").pointerTypes("TfLiteDelegatePtrVector").define())
               .put(new Info("std::unordered_map<std::int32_t,std::unique_ptr<tflite::resource::ResourceBase> >").valueTypes("@StdMove IntResourceBaseMap").pointerTypes("IntResourceBaseMap").define())
               .put(new Info("const TfLiteOperator* (*)(void*, int, int)").pointerTypes("Find_builtin_op_external_Pointer_int_int"))
               .put(new Info("const TfLiteOperator* (*)(void*, const char*, int)").pointerTypes("Find_custom_op_external_Pointer_String_int"))

               .put(new Info("tflite::impl::Interpreter::typed_tensor<int8_t>").javaNames("typed_tensor_byte"))
               .put(new Info("tflite::impl::Interpreter::typed_tensor<int16_t>").javaNames("typed_tensor_short"))
               .put(new Info("tflite::impl::Interpreter::typed_tensor<int32_t>").javaNames("typed_tensor_int"))
               .put(new Info("tflite::impl::Interpreter::typed_tensor<int64_t>").javaNames("typed_tensor_long"))
               .put(new Info("tflite::impl::Interpreter::typed_tensor<float>").javaNames("typed_tensor_float"))
               .put(new Info("tflite::impl::Interpreter::typed_tensor<double>").javaNames("typed_tensor_double"))
               .put(new Info("tflite::impl::Interpreter::typed_tensor<bool>").javaNames("typed_tensor_bool"))
               .put(new Info("tflite::impl::Interpreter::typed_tensor<TfLiteFloat16>").javaNames("typed_tensor_float16"))

               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<int8_t>").javaNames("typed_input_tensor_byte"))
               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<int16_t>").javaNames("typed_input_tensor_short"))
               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<int32_t>").javaNames("typed_input_tensor_int"))
               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<int64_t>").javaNames("typed_input_tensor_long"))
               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<float>").javaNames("typed_input_tensor_float"))
               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<double>").javaNames("typed_input_tensor_double"))
               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<bool>").javaNames("typed_input_tensor_bool"))
               .put(new Info("tflite::impl::Interpreter::typed_input_tensor<TfLiteFloat16>").javaNames("typed_input_tensor_float16"))

               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<int8_t>").javaNames("typed_output_tensor_byte"))
               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<int16_t>").javaNames("typed_output_tensor_short"))
               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<int32_t>").javaNames("typed_output_tensor_int"))
               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<int64_t>").javaNames("typed_output_tensor_long"))
               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<float>").javaNames("typed_output_tensor_float"))
               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<double>").javaNames("typed_output_tensor_double"))
               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<bool>").javaNames("typed_output_tensor_bool"))
               .put(new Info("tflite::impl::Interpreter::typed_output_tensor<TfLiteFloat16>").javaNames("typed_input_tensor_float16"))

               .put(new Info("tflite::impl::FlatBufferModel").purify())
               .put(new Info("tflite::impl::FlatBufferModelBase<tflite::impl::FlatBufferModel>::VerifyAndBuildFromFileDescriptor").skip())
               .put(new Info("tflite::impl::FlatBufferModelBase<tflite::impl::FlatBufferModel>").javaNames("FlatBufferModelBase"))

                // Classes passed to some native functions as unique_ptr and that can be allocated Java-side
               .put(new Info("tflite::impl::Interpreter::Interpreter").annotations("@UniquePtr", "@Name(\"std::make_unique<tflite::impl::Interpreter>\")"))
               .put(new Info("tflite::Subgraph::Subgraph").annotations("@UniquePtr", "@Name(\"std::make_unique<tflite::Subgraph>\")"))
        ;
    }
}
