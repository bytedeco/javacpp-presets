/*
 * Copyright (C) 2018-2021 Alexander Merritt, Samuel Audet
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

package org.bytedeco.onnx.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.*;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;

@Properties(inherit = javacpp.class, target = "org.bytedeco.onnx", global = "org.bytedeco.onnx.global.onnx", value = {
@Platform(
    value = {"linux", "macosx", "windows"},
    define = {"ONNX_NAMESPACE onnx", "ONNX_USE_LITE_PROTO", "ONNX_ML 1", "SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std", "NO_WINDOWS_H"},
    compiler = "cpp11",
    exclude = "google/protobuf/port_def.inc",
    include = {
        "onnx/defs/schema.h",
        "onnx/defs/operator_sets.h",
        "onnx/defs/operator_sets_ml.h",
        "onnx/defs/operator_sets_training.h",
        "onnx/defs/data_type_utils.h",
        "onnx/defs/shape_inference.h",
        "google/protobuf/port_def.inc",
        "google/protobuf/arena.h",
        "google/protobuf/message_lite.h",
        "google/protobuf/unknown_field_set.h",
        "google/protobuf/descriptor.h",
        "onnx/onnx-data.pb.h",
        "onnx/onnx-operators-ml.pb.h",
        "onnx/onnx-ml.pb.h",
        "onnx/proto_utils.h",
//        "onnx/string_utils.h",
        "onnx/defs/function.h",
        "onnx/checker.h",
        "onnx/shape_inference/implementation.h",
        "onnx/onnxifi.h",
        "onnx/common/tensor.h",
        "onnx/common/array_ref.h",
        "onnx/common/status.h",
//        "onnx/common/graph_node_list.h",
        "onnx/common/stl_backports.h",
        "onnx/common/ir.h",
        "onnx/common/ir_pb_converter.h",
        "onnx/version_converter/adapters/adapter.h",
        "onnx/version_converter/helper.h",
        "onnx/version_converter/BaseConverter.h",
        "onnx/version_converter/convert.h",
//        "onnx/optimizer/pass_registry.h",
//        "onnx/optimizer/pass.h",
//        "onnx/optimizer/optimize.h",
    },
    link = {"onnx_proto", "onnx", "onnxifi"}),
// "onnxifi" not available on Mac
@Platform(value = "macosx", link = {"onnx_proto", "onnx"}),
@Platform(value = "windows", link = {"libprotobuf#", "onnx_proto", "onnx", "onnxifi"})
})
public class onnx implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "onnx"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("ONNX_NAMESPACE").cppText("#define ONNX_NAMESPACE onnx"))
               .put(new Info("alignas", "PROTOC_EXPORT", "LIBPROTOBUF_EXPORT", "PROTOBUF_EXPORT", "PROTOBUF_CONSTEXPR", "PROTOBUF_FINAL",
                             "PROTOBUF_ATTRIBUTE_REINITIALIZES", "PROTOBUF_ALWAYS_INLINE", "PROTOBUF_NOINLINE", "PROTOBUF_RETURNS_NONNULL",
                             "PROTOBUF_NAMESPACE_ID", "PROTOBUF_NAMESPACE_OPEN", "PROTOBUF_NAMESPACE_CLOSE", "PROTOBUF_FALLTHROUGH_INTENDED", "PROTOBUF_UNUSED",
                             "PROTOBUF_EXPORT_TEMPLATE_DEFINE", "GOOGLE_ATTRIBUTE_ALWAYS_INLINE", "GOOGLE_ATTRIBUTE_FUNC_ALIGN", "GOOGLE_ATTRIBUTE_NOINLINE",
                             "GOOGLE_PREDICT_TRUE", "GOOGLE_PREDICT_FALSE", "GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE", "GOOGLE_PROTOBUF_ATTRIBUTE_RETURNS_NONNULL",
                             "ONNX_UNUSED", "ONNX_API", "ONNXIFI_ABI", "ONNXIFI_CHECK_RESULT", "ONNXIFI_PUBLIC", "ONNX_IMPORT", "ONNX_EXPORT").cppTypes().annotations())
               .put(new Info("GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER").define(false))

               .put(new Info("PROTOBUF_DEPRECATED").cppText("#define PROTOBUF_DEPRECATED DEPRECATED").cppTypes())
               .put(new Info("PROTOBUF_DEPRECATED_ENUM").cppText("#define PROTOBUF_DEPRECATED_ENUM DEPRECATED").cppTypes())
               .put(new Info("PROTOBUF_RUNTIME_DEPRECATED").cppText("#define PROTOBUF_RUNTIME_DEPRECATED() DEPRECATED").cppTypes())
               .put(new Info("DEPRECATED").annotations("@Deprecated"))

               .put(new Info("onnx::AttributeProto::AttributeType", "onnx::TensorProto::DataType", "onnx::TensorProto_DataType",
                             "onnx::SequenceProto::DataType", "onnx::OptionalProto::DataType", "onnx::TypeProto::ValueCase",
                             "onnx::OpSchema::UseType", "onnx::OperatorStatus").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int..."))
               .put(new Info("onnx::OpSchema::SinceVersion").annotations("@Function"))
               .put(new Info("string", "std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("onnx::TensorShapeProto_Dimension", "onnx::TensorShapeProto::Dimension", "TensorShapeProto_Dimension").pointerTypes("Dimension"))
               .put(new Info("std::vector<float>").pointerTypes("FloatVector").define())
               .put(new Info("std::vector<int64_t>").pointerTypes("LongVector").define())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("onnx::Dimension").pointerTypes("DimensionIR").define())
               .put(new Info("std::initializer_list", "std::function<void(OpSchema&&)>", "NodeKind", "graph_node_list",
                             "graph_node_list_iterator", "std::vector<onnx::Tensor>::const_iterator",
                             "onnx::Attributes<onnx::Node>", "Symbol", "std::reverse_iterator<onnx::ArrayRef<onnx::Node::Value*>::iterator>",
                             "const_graph_node_list_iterator", "const_graph_node_list", "onnx::toString", "onnx::ResourceGuard", "onnx::GraphInferencer",
                             "onnx::shape_inference::GraphInferenceContext", "onnx::optimization::FullGraphBasedPass", "onnx::optimization::ImmutablePass",
                             "PROTOBUF_INTERNAL_EXPORT_protobuf_onnx_2fonnx_2doperators_2dml_2eproto", "PROTOBUF_INTERNAL_EXPORT_onnx_2fonnx_2doperators_2dml_2eproto",
                             "PROTOBUF_INTERNAL_EXPORT_protobuf_onnx_2fonnx_2dml_2eproto", "PROTOBUF_INTERNAL_EXPORT_onnx_2fonnx_2dml_2eproto",
                             "PROTOBUF_INTERNAL_EXPORT_onnx_2fonnx_2ddata_2eproto").skip())
               .put(new Info("onnx::shape_inference::InferenceContextImpl").skip())
               .put(new Info("std::set<int>").pointerTypes("IntSet").define())
               .put(new Info("onnx::optimization::Pass").purify(true))
               .put(new Info("std::shared_ptr<onnx::optimization::Pass>").annotations("@SharedPtr").pointerTypes("Pass"))
//               .put(new Info("std::map<std::string,std::shared_ptr<onnx::optimization::Pass> >").pointerTypes("StringPassMap").define())
               .put(new Info("std::unordered_set<std::string>").pointerTypes("UnorderedStringSet").define())
               .put(new Info("std::multimap<std::string,const onnx::FunctionProto*>").skip())
               .put(new Info("std::runtime_error").cast().pointerTypes("Pointer"))
               .put(new Info("onnx::version_conversion::BaseVersionConverter::registerAdapter").skip())
//               .put(new Info("onnx::Node::Value", "onnx::Graph::Value", "onnx::AttributeValue", "onnx::Value").skip())
               .put(new Info("std::iterator_traits<onnx::generic_graph_node_list_iterator<T> >").skip())
               .put(new Info("onnx::ArrayRef<onnx::Node::Value*>", "onnx::ArrayRef<onnx::Graph::Value*>", "std::vector<onnx::Node::Value*>").skip())
//               .put(new Info("std::vector<onnx::Tensor>").pointerTypes("VectorTensor").define())
               .put(new Info("google::protobuf::int8", "google::protobuf::uint8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("google::protobuf::int16", "google::protobuf::uint16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("google::protobuf::int32", "google::protobuf::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("google::protobuf::int64", "google::protobuf::uint64").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("std::pair<google::protobuf::uint64,google::protobuf::uint64>").pointerTypes("LongLongPair").define())
               .put(new Info("google::protobuf::Message").cast().pointerTypes("MessageLite"))

               .put(new Info("google::protobuf::Any", "google::protobuf::Descriptor", "google::protobuf::Metadata").cast().pointerTypes("Pointer"))
               .put(new Info("google::protobuf::FindAllExtensions", "google::protobuf::Map", "google::protobuf::RepeatedField", "google::protobuf::RepeatedPtrField",
                             "google::protobuf::arena_metrics::EnableArenaMetrics", "google::protobuf::io::EpsCopyOutputStream", "google::protobuf::internal::DescriptorTable",
                             "google::protobuf::internal::ExplicitlyConstructed", "google::protobuf::internal::MapEntry", "google::protobuf::internal::MapField",
                             "google::protobuf::internal::AuxillaryParseTableField", "google::protobuf::internal::ParseTableField", "google::protobuf::internal::ParseTable",
                             "google::protobuf::internal::FieldMetadata", "google::protobuf::internal::SerializationTable", "google::protobuf::internal::proto3_preserve_unknown_",
                             "google::protobuf::internal::MergePartialFromImpl", "google::protobuf::internal::UnknownFieldParse", "google::protobuf::internal::WriteLengthDelimited",
                             "google::protobuf::is_proto_enum", "google::protobuf::GetEnumDescriptor", "google::protobuf::RepeatedField", "google::protobuf::UnknownField::LengthDelimited",
                             "google::protobuf::internal::empty_string_once_init_", "google::protobuf::SourceLocation::leading_detached_comments", "onnx::_TypeProto_default_instance_",
                             "onnx::_TypeProto_Map_default_instance_", "onnx::_TypeProto_Sequence_default_instance_", "onnx::_SparseTensorProto_default_instance_",
                             "onnx::_TypeProto_Opaque_default_instance_", "onnx::_TypeProto_SparseTensor_default_instance_", "onnx::_TrainingInfoProto_default_instance_",
                             "onnx::_TypeProto_Tensor_default_instance_",  "onnx::_ValueInfoProto_default_instance_", "onnx::_TensorShapeProto_Dimension_default_instance_",
                             "onnx::_TensorShapeProto_default_instance_", "onnx::_TensorProto_Segment_default_instance_","onnx::_TensorProto_default_instance_",
                             "onnx::_NodeProto_default_instance_", "onnx::_GraphProto_default_instance_", "onnx::_FunctionProto_default_instance_",
                             "onnx::_ModelProto_default_instance_", "onnx::_OperatorSetProto_default_instance_", "onnx::RegisterOneFunctionBuilder", "BuildFunction",
                             "onnx::_OperatorSetIdProto_default_instance_", "onnx::_StringStringEntryProto_default_instance_", "onnx::_OperatorProto_default_instance_",
                             "onnx::_AttributeProto_default_instance_", "onnx::_TensorAnnotation_default_instance_", "onnx::_MapProto_default_instance_",
                             "onnx::_SequenceProto_default_instance_", "onnx::_OptionalProto_default_instance_", "onnx::_TypeProto_Optional_default_instance_").skip())

               .put(new Info("onnx::DataType").annotations("@StdString").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("onnx::OpSchema::Attribute").pointerTypes("OpSchema.Attribute"))
               .put(new Info("onnx::OpSchema::FormalParameter").pointerTypes("OpSchema.FormalParameter"))
               .put(new Info("onnx::OpSchema::TypeConstraintParam").pointerTypes("OpSchema.TypeConstraintParam"))

               .put(new Info("std::pair<int,int>", "std::pair<onnx::OpSchema::UseType,int>").pointerTypes("UseTypeIntPair").define())
               .put(new Info("const std::map<std::string,onnx::OpSchema::Attribute>").pointerTypes("StringAttributeMap").define())
               .put(new Info("std::unordered_map<size_t,std::string>").pointerTypes("SizeTStringMap").define())
               .put(new Info("std::unordered_map<std::string,int>").pointerTypes("StringIntMap").define())
               .put(new Info("std::unordered_map<std::string,onnx::TypeProto*>").pointerTypes("StringTypeProtoMap").define())
               .put(new Info("std::unordered_map<std::string,onnx::TensorShapeProto>").pointerTypes("StringTensorShapeProtoMap").define())
               .put(new Info("std::unordered_map<std::string,const onnx::TensorProto*>").pointerTypes("StringTensorProtoMap").define())
               .put(new Info("std::unordered_map<std::string,const onnx::AttributeProto*>",
                             "std::unordered_map<std::string,const AttributeProto*>").pointerTypes("StringAttributeProtoMap").define())
               .put(new Info("std::unordered_map<std::string,std::pair<int,int> >").pointerTypes("StringIntIntPairMap").define())
               .put(new Info("std::unordered_map<int,int>").pointerTypes("IntIntMap").define())
               .put(new Info("std::unordered_set<onnx::DataType>").pointerTypes("DataTypeSet").define())
               .put(new Info("std::set<std::string>").pointerTypes("StringSet").define())
               .put(new Info("std::vector<onnx::OpSchema>").pointerTypes("OpSchemaVector").define())
               .put(new Info("std::vector<onnx::OpSchema::FormalParameter>").pointerTypes("FormalParameterVector").define())
               .put(new Info("std::vector<onnx::TypeProto*>").pointerTypes("TypeProtoVector").define())
               .put(new Info("const std::vector<onnx::OpSchema::TypeConstraintParam>").pointerTypes("TypeConstraintParamVector").define())
               .put(new Info("onnx::TensorShapeProto").pointerTypes("TensorShapeProto"))
               .put(new Info("std::vector<const onnx::TensorShapeProto*>").pointerTypes("TensorShapeProtoVector").define())

               .put(new Info("onnx::OpSchema::GetTypeAndShapeInferenceFunction", "onnx::OpSchema::SetContextDependentFunctionBodyBuilder",
                             "onnx::OpSchema::GetDataPropagationFunction", "onnx::RegisterSchema", "onnx::ReplaceAll", "onnx::DbgOperatorSetTracker::Instance",
                             "onnx::shape_inference::checkShapesAndTypes(const onnx::TypeProto_Sequence&, const onnx::TypeProto_Sequence&)",
                             "onnx::shape_inference::mergeShapesAndTypes(const onnx::TypeProto_Sequence&, onnx::TypeProto_Tensor*)").skip())

               .put(new Info("onnx::RetrieveValues<int64_t>").javaNames("RetrieveValuesLong"))
               .put(new Info("onnx::RetrieveValues<std::string>").javaNames("RetrieveValuesString"))
               .put(new Info("onnx::ParseProtoFromBytes<google::protobuf::MessageLite>").javaNames("ParseProtoFromBytes"))

               .put(new Info("onnx::checker::ValidationError", "onnx::version_conversion::GenericAdapter").purify())

               .put(new Info("std::function<bool(int)>").pointerTypes("BoolIntFn"))
               .put(new Info("std::function<bool(int,int)>").pointerTypes("BoolIntIntFn"))
               .put(new Info("std::function<int(int)>").pointerTypes("IntIntFn"))
               .put(new Info("std::function<void(DataPropagationContext&)>").pointerTypes("DataPropagationFunction"))
               .put(new Info("std::function<void(InferenceContext&)>").pointerTypes("InferenceFunction"))
               .put(new Info("std::function<void(OpSchema&)>").pointerTypes("VoidOpSchemaFn"))
               .put(new Info("std::function<std::pair<bool,int>(int)>").pointerTypes("PairBoolIntIntFn"));
    }

    public static class BoolIntFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    BoolIntFn(Pointer p) { super(p); }
        protected BoolIntFn() { allocate(); }
        private native void allocate();
        public native boolean call(int a);
    }

    public static class BoolIntIntFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    BoolIntIntFn(Pointer p) { super(p); }
        protected BoolIntIntFn() { allocate(); }
        private native void allocate();
        public native boolean call(int a, int b);
    }

    public static class IntIntFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    IntIntFn(Pointer p) { super(p); }
        protected IntIntFn() { allocate(); }
        private native void allocate();
        public native int call(int a);
    }

    public static class DataPropagationFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    DataPropagationFunction(Pointer p) { super(p); }
        protected DataPropagationFunction() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("onnx::DataPropagationContext*") Pointer a);
    }

    public static class InferenceFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    InferenceFunction(Pointer p) { super(p); }
        protected InferenceFunction() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("onnx::InferenceContext*") Pointer a);
    }

    public static class VoidOpSchemaFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    VoidOpSchemaFn(Pointer p) { super(p); }
        protected VoidOpSchemaFn() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("onnx::OpSchema*") Pointer a);
    }

//    public static class PairBoolIntIntFn extends FunctionPointer {
//        static { Loader.load(); }
//        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
//        public    PairBoolIntIntFn(Pointer p) { super(p); }
//        protected PairBoolIntIntFn() { allocate(); }
//        private native void allocate();
//        public @ByVal native @Cast("std::pair<bool,int>*") Pointer call(int a);
//    }
}
