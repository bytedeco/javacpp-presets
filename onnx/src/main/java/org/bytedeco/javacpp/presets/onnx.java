package org.bytedeco.javacpp.presets;

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


@Properties(target="org.bytedeco.javacpp.onnx", value={@Platform(compiler="cpp11",
include={
"onnx/proto_utils.h",
"defs/schema.h",
"defs/data_type_utils.h",
"onnx/onnx-operators.pb.h",
"onnx/onnx.pb.h",
"/usr/include/google/protobuf/message_lite.h",
"/usr/include/google/protobuf/unknown_field_set.h",
//"onnx/checker.h",
//"onnx_pb.h",
//"onnx/string_utils.h"
}, 
link="onnx")})
public class onnx implements InfoMapper {
    public void map(InfoMap infoMap) {
infoMap.put(new Info("string", "std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
              .put(new Info("int", "int32").valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int..."))
              .put(new Info("onnx::TensorProto::DataType").valueTypes("int"))
              .put(new Info("onnx::TensorProto_DataType").valueTypes("int"))
//              .put(new Info("onnx::OpSchema::UseType").valueTypes("int"))
              .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
              .put(new Info("std::set<int>").pointerTypes("IntSet").define())
              .put(new Info("std::runtime_error").cast().pointerTypes("Pointer"))
//              .put(new Info("onnx::checker::ValidationError").cast().pointerTypes("ValidationError").define())
//               .put(new Info("long long", "std::size_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long..."))
//               .put(new Info("float").valueTypes("float").pointerTypes("FloatPointer", "FloatBuffer", "float..."))
//               .put(new Info("double").valueTypes("double").pointerTypes("DoublePointer", "DoubleBuffer", "double..."))
//               .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("BoolPointer", "boolean..."))
//               .put(new Info("std::complex<float>").cast().pointerTypes("FloatPointer", "FloatBuffer", "float..."))
               .put(new Info("google::protobuf::int8", "google::protobuf::uint8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("google::protobuf::int16", "google::protobuf::uint16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("google::protobuf::int32", "google::protobuf::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("google::protobuf::int64", "google::protobuf::uint64").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("google::protobuf::Message").cast().pointerTypes("MessageLite"))
               .put(new Info("google::protobuf::Any", "google::protobuf::Descriptor", "google::protobuf::EnumDescriptor", "google::protobuf::Metadata").cast().pointerTypes("Pointer"))
               .put(new Info("google::protobuf::Map", "google::protobuf::RepeatedField", "google::protobuf::RepeatedPtrField", "protobuf::RepeatedPtrField",
                             "google::protobuf::internal::ExplicitlyConstructed", "google::protobuf::internal::MapEntry", "google::protobuf::internal::MapField",
                             "google::protobuf::internal::AuxillaryParseTableField", "google::protobuf::internal::ParseTableField", "google::protobuf::internal::ParseTable",
                             "google::protobuf::internal::FieldMetadata", "google::protobuf::internal::SerializationTable", "google::protobuf::internal::proto3_preserve_unknown_",
                             "google::protobuf::is_proto_enum", "google::protobuf::GetEnumDescriptor").skip()) 
          
               .put(new Info("std::vector<onnx::OpSchema::FormalParameter>").pointerTypes("FormalParameterVector").define())
               .put(new Info("std::vector<onnx::OpSchema>").pointerTypes("OpSchemaVector").define())
               .put(new Info("onnx::OpSchema::FormalParameter").pointerTypes("OpSchema.FormalParameter"))
               .put(new Info("onnx::OpSchema::TypeConstraintParam").pointerTypes("OpSchema.TypeConstraintParam"))
       
               .put(new Info("onnx::OpSchema::Attribute").pointerTypes("OpSchema.Attribute"))
               .put(new Info("const std::map<std::string,onnx::OpSchema::Attribute>").pointerTypes("StringAttributeMap").define())
//               .put(new Info("std::map<std::string,onnx::OpSchema::Attribute>").pointerTypes("StringAttributeMap").define())

               .put(new Info("std::function<bool(int)>").pointerTypes("BoolIntFn"))
               .put(new Info("std::function<bool(int,int)>").pointerTypes("BoolIntIntFn"))
               .put(new Info("std::function<int(int)>").pointerTypes("IntIntFn"))
//               .put(new Info("std::function<void(OpSchema&)>").pointerTypes("VoidOpSchemaFn"))
               .put(new Info("onnx::OpSchema::UseType").valueTypes("int")) 
//               .put(new Info("onnx::OpSchema::UseType::DEFAULT").pointerTypes("DEFAULTUseType").define())
//               .put(new Info("onnx::OpSchema::UseType::CONSUME_ALLOWED").pointerTypes("CONSUME_ALLOWEDUseType").define())
//               .put(new Info("onnx::OpSchema::UseType::CONSUME_ENFORCED").pointerTypes("CONSUME_ENFORCEDUseType").define())
               .put(new Info("std::pair<int,int>", "std::pair<onnx::OpSchema::UseType,int>").pointerTypes("UseTypeIntPair").define())
//               .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
               .put(new Info("std::unordered_map<std::string,std::pair<int,int> >").pointerTypes("StringIntIntPairMap").define())   
               .put(new Info("std::unordered_map<int,int>").pointerTypes("IntIntMap").define())
               .put(new Info("std::function<std::pair<bool,int>(int)>").pointerTypes("PairBoolIntIntFn"))
               .put(new Info("onnx::DataType").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::unordered_set<onnx::DataType>").pointerTypes("DataTypeSet").define())
               .put(new Info("const std::vector<onnx::OpSchema::TypeConstraintParam>").pointerTypes("TypeConstraintParamVector").define());
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
/*
    public static class VoidOpSchemaFn extends FunctionPointer {
        static { Loader.load(); }
 
        public    VoidOpSchemaFn(Pointer p) { super(p); }
        protected VoidOpSchemaFn() { allocate(); }
        private native void allocate();
        public native void call(@Cast("onnx::OpSchema") Pointer a);
    }
*/

    public static class PairBoolIntIntFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    PairBoolIntIntFn(Pointer p) { super(p); }
        protected PairBoolIntIntFn() { allocate(); }
        private native void allocate();
        public @ByVal native @Cast("std::pair<bool,int>*") Pointer call(int a);
    }

//    @Documented @Retention(RetentionPolicy.RUNTIME)
//    @Target({ElementType.METHOD, ElementType.PARAMETER})
//    @Cast("onnx::OpSchema&") @Adapter("OpSchemaAdapter")
//    public @interface OpSchema { }
}
