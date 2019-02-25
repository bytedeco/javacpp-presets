/*
 * Copyright (C) 2018 Alexander Merritt
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

package org.bytedeco.ngraph.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Properties(target = "org.bytedeco.ngraph", global = "org.bytedeco.ngraph.global.ngraph", value = {@Platform(
    value = {"linux", "macosx"},
    define = {"SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std"},
    compiler = "cpp11",
    include = {
        "ngraph/frontend/onnxifi/backend.hpp",
        "ngraph/frontend/onnxifi/backend_manager.hpp",
        "ngraph/descriptor/tensor.hpp",
        "ngraph/runtime/tensor.hpp",
        "ngraph/runtime/backend.hpp",
        "ngraph/runtime/cpu/cpu_backend.hpp",
//        "ngraph/runtime/cpu/cpu_external_function.hpp",
        "ngraph/runtime/performance_counter.hpp",
        "ngraph/type/element_type.hpp",
        "ngraph/shape.hpp",
        "ngraph/function.hpp",
        "ngraph/node_vector.hpp",
        "ngraph/assertion.hpp",
        "ngraph/except.hpp",
        "ngraph/placement.hpp",
        "ngraph/coordinate.hpp",
        "ngraph/strides.hpp",
        "ngraph/descriptor/input.hpp",
        "ngraph/descriptor/output.hpp",
        "ngraph/op/op.hpp",
        "ngraph/parameter_vector.hpp",
        "ngraph/op/parameter.hpp",
        "ngraph/op/constant.hpp",
        "ngraph/op/util/binary_elementwise_arithmetic.hpp",
        "ngraph/op/add.hpp",
        "ngraph/op/multiply.hpp",
        "ngraph/result_vector.hpp",
        "ngraph/op/util/op_annotations.hpp",
        "ngraph/autodiff/adjoints.hpp",
        "ngraph/dimension.hpp",
        "ngraph/rank.hpp",
        "ngraph/partial_shape.hpp",
        "ngraph/node.hpp",
        "ngraph/frontend/onnxifi/onnxifi.h",
//        "core/node.hpp",
        "ngraph/frontend/onnx_import/core/weight.hpp",
//        "core/model.hpp",
//        "core/value_info.hpp",
//        "core/tensor.hpp",
        "ngraph/frontend/onnx_import/onnx.hpp"
    },
    preload = {"iomp5", "mklml", "mklml_intel"},
    link = {"mkldnn", "ncurses@.6", "onnxifi", "ngraph", "onnxifi-ngraph", "cpu_backend", "codegen", "tbb@.2"}
//@Platform(value = "macosx", link = {"onnx_proto", "onnx"})}) // "onnxifi" not available on Mac
)})
public class ngraph implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.ngraph.Function;"))
//               .put(new Info("string", "std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("ngraph::runtime::cpu::CPU_Backend::Property").cast().valueTypes("int"))
               .put(new Info("ngraph::element::Type_t::boolean").javaNames("boolean_type"))
               .put(new Info("ngraph::descriptor::Tensor").purify(true).pointerTypes("DescriptorTensor"))
               .put(new Info("std::shared_ptr<descriptor::Tensor>", "std::shared_ptr<ngraph::descriptor::Tensor>").annotations("@SharedPtr").pointerTypes("DescriptorTensor"))
               .put(new Info("ngraph::runtime::Tensor").purify(true)) //.purify(false).virtualize())
               .put(new Info("runtime::Handle").annotations("@SharedPtr").pointerTypes("Function"))
               .put(new Info("ngraph::onnxifi::Backend").purify(true).pointerTypes("NgraphONNXIFIBackend"))
               .put(new Info("ngraph::onnxifi::Backend::operator =").skip())
               .put(new Info("ngraph::element::from<char>").javaNames("fromChar"))
               .put(new Info("ngraph::element::from<bool>").javaNames("fromBool"))
               .put(new Info("ngraph::element::from<float>").javaNames("fromFloat"))
               .put(new Info("ngraph::element::from<double>").javaNames("fromDouble"))
               .put(new Info("ngraph::element::from<int8_t>").javaNames("fromInt8t"))
               .put(new Info("ngraph::element::from<int16_t>").javaNames("fromInt16t"))
               .put(new Info("ngraph::element::from<int32_t>").javaNames("fromInt32t"))
               .put(new Info("ngraph::element::from<int64_t>").javaNames("fromInt64t"))
               .put(new Info("ngraph::element::from<uint8_t>").javaNames("fromUInt8t"))
               .put(new Info("ngraph::element::from<uint16_t>").javaNames("fromUInt16t"))
               .put(new Info("ngraph::element::from<uint32_t>").javaNames("fromUInt32t"))
               .put(new Info("ngraph::element::from<uint64_t>").javaNames("fromUInt64t"))
               .put(new Info("ngraph::element::from<ngraph::bfloat16>").javaNames("fromNGraphBFloat16"))

               .put(new Info("ngraph::op::util::BinaryElementwiseArithmetic", "ngraph::op::ScalarConstantLike").purify(true))
               .put(new Info("std::shared_ptr<ngraph::op::Result>","std::shared_ptr<op::Result>").annotations("@SharedPtr").pointerTypes("Result"))
               .put(new Info("std::shared_ptr<ngraph::runtime::Tensor>").annotations("@SharedPtr").pointerTypes("Tensor"))
//               .put(new Info("ngraph::Node").purify(false).virtualize())
               .put(new Info("std::shared_ptr<const Node>", "std::shared_ptr<ngraph::Node>").annotations("@SharedPtr").pointerTypes("Node"))
               .put(new Info("std::shared_ptr<ngraph::op::Parameter>", "std::shared_ptr<op::Parameter>").annotations("@SharedPtr").pointerTypes("Parameter"))
               .put(new Info("std::shared_ptr<ngraph::Function>", "std::shared_ptr<Function>").annotations("@SharedPtr").pointerTypes("Function"))
               .put(new Info("std::shared_ptr<const ngraph::Function>", "std::shared_ptr<const Function>").annotations("@Cast(\"const ngraph::Function*\") @SharedPtr").pointerTypes("Function"))
               .put(new Info("std::enable_shared_from_this<ngraph::Node>", "std::enable_shared_from_this<Node>", "std::enable_shared_from_this<ngraph::runtime::cpu::CPU_ExternalFunction>").pointerTypes("Pointer"))
               .put(new Info("std::runtime_error").cast().pointerTypes("Pointer"))
               .put(new Info("std::list<std::shared_ptr<Node> >", "std::pair<std::shared_ptr<ngraph::op::Result>,std::shared_ptr<ngraph::op::Parameter> >", "std::deque<ngraph::Node::descriptor::Input>", "std::deque<descriptor::Output>", "std::set<ngraph::Node::descriptor::Input*>", "std::unordered_set<descriptor::Tensor*>", "std::stringstream", "ngraph::Node::has_same_type", "ngraph::descriptor::Tensor::set_tensor_layout", "ngraph::runtime::cpu::CPU_ExternalFunction::get_executor", "ngraph::runtime::cpu::CPU_ExternalFunction::get_callees", "ngraph::runtime::cpu::CPU_ExternalFunction::get_halide_functions", "ngraph::runtime::cpu::CPU_ExternalFunction::get_subgraph_params", "ngraph::runtime::cpu::CPU_ExternalFunction::get_subgraph_param_sizes", "ngraph::runtime::cpu::CPU_ExternalFunction::get_subgraph_param_ptrs", "ngraph::runtime::cpu::CPU_ExternalFunction::get_parameter_layout_descriptors", "ngraph::runtime::cpu::CPU_ExternalFunction::get_result_layout_descriptors", "ngraph::runtime::cpu::CPU_ExternalFunction::get_mkldnn_emitter", "ngraph::runtime::cpu::CPU_ExternalFunction::add_state", "ngraph::runtime::cpu::CPU_ExternalFunction::add_state", "ngraph::runtime::cpu::CPU_ExternalFunction::get_functors", "ngraph::runtime::cpu::CPU_Backend::make_call_frame", "ngraph::onnxifi::BackendManager::unregister", "ngraph::onnxifi::BackendManager::get", "ngraph::onnx_import::register_operator").skip())
               .put(new Info("ONNXIFI_ABI", "ONNXIFI_PUBLIC", "ONNXIFI_CHECK_RESULT", "NGRAPH_API").cppTypes().annotations())
               .put(new Info("std::initializer_list", "from<char>", "from<bool>", "from<float>", "from<double>", "from<int8_t>", "from<int16_t>", "from<int32_t>",
                             "from<int64_t>", "from<uint8_t>", "from<uint16_t>", "from<uint32_t>", "from<uint64_t>", "from<ngraph::bfloat16>").skip())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<size_t>").pointerTypes("SizeTVector").define())
               .put(new Info("std::vector<std::shared_ptr<ngraph::op::Result> >", "std::vector<std::shared_ptr<op::Result> >").pointerTypes("NgraphResultVector").define())
               .put(new Info("std::size_t", "size_t", "std::int64_t", "int64_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
//               .put(new Info("std::vector<std::shared_ptr<ngraph::op::Result> >").pointerTypes("Pointer"))
//               .put(new Info("std::vector<std::shared_ptr<op::Parameter> >", "std::vector<std::shared_ptr<ngraph::op::Parameter> >").pointerTypes("Pointer"))
               .put(new Info("std::unordered_map<std::string,void*>").pointerTypes("StringVoidMap").define())
               .put(new Info("std::vector<std::shared_ptr<ngraph::op::Parameter> >", "std::vector<std::shared_ptr<op::Parameter> >").pointerTypes("NgraphParameterVector").define())
               .put(new Info("std::vector<std::shared_ptr<ngraph::Node> >").pointerTypes("NgraphNodeVector").define())
               .put(new Info("std::vector<std::shared_ptr<ngraph::runtime::Tensor> >", "std::vector<std::shared_ptr<runtime::Tensor> >").pointerTypes("NgraphTensorVector").define())
               .put(new Info("std::vector<std::shared_ptr<ngraph::Function> >").pointerTypes("NgraphFunctionVector").define());

       //TODO: std::list<std::shared_ptr<Node> >   ?
    }
}
