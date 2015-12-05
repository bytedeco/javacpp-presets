/*
 * Copyright (C) 2015 Samuel Audet
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

package org.bytedeco.javacpp.presets;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Adapter;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(value = @Platform(value = {"linux", "macosx"}, compiler = "cpp11", define = "NDEBUG", include = {
        "tensorflow/core/platform/default/integral_types.h", "tensorflow/core/framework/numeric_types.h", "tensorflow/core/platform/init_main.h",
        /* "tensorflow/core/platform/default/mutex.h", "tensorflow/core/lib/core/refcount.h", "tensorflow/core/lib/gtl/array_slice.h",
        "tensorflow/core/lib/core/stringpiece.h", */ "tensorflow/core/platform/port.h", "tensorflow/core/lib/core/error_codes.pb.h",
        "tensorflow/core/platform/logging.h", "tensorflow/core/public/status.h", "tensorflow/core/platform/protobuf.h",
        "tensorflow/core/public/env.h", "tensorflow/core/framework/config.pb.h", "tensorflow/core/public/session_options.h",
        "tensorflow/core/lib/core/threadpool.h", "tensorflow/core/framework/allocation_description.pb.h", "tensorflow/core/framework/allocator.h",
        "tensorflow/core/framework/tensor_shape.pb.h", "tensorflow/core/framework/types.pb.h", "tensorflow/core/framework/tensor.pb.h",
        "tensorflow/core/framework/tensor_description.pb.h", "tensorflow/core/framework/tensor_types.h", "tensorflow/core/public/tensor_shape.h",
        "tensorflow/core/public/tensor.h", "tensorflow/core/framework/attr_value.pb.h", "tensorflow/core/framework/op_def.pb.h",
        "tensorflow/core/framework/function.pb.h", "tensorflow/core/framework/graph.pb.h", "tensorflow/core/public/session.h",
        "tensorflow/core/public/tensor_c_api.h", "tensorflow/core/framework/op_def.pb.h", "tensorflow/core/framework/op_def_builder.h",
        "tensorflow/core/framework/op_def_util.h", "tensorflow/core/framework/op.h", "tensorflow/core/framework/types.h",
        "tensorflow/core/graph/edgeset.h", "tensorflow/core/lib/gtl/iterator_range.h", "tensorflow/core/graph/graph.h",
        "tensorflow/core/graph/node_builder.h", "tensorflow/core/graph/graph_def_builder.h", "tensorflow/core/graph/default_device.h",
        "tensorflow/cc/ops/standard_ops.h", "tensorflow/cc/ops/const_op.h", "tensorflow/cc/ops/cc_op_gen.h",
        "tensorflow/cc/ops/array_ops.h", "tensorflow/cc/ops/attention_ops.h", "tensorflow/cc/ops/const_op.h",
        "tensorflow/cc/ops/data_flow_ops.h", "tensorflow/cc/ops/image_ops.h", "tensorflow/cc/ops/io_ops.h",
        "tensorflow/cc/ops/linalg_ops.h", "tensorflow/cc/ops/logging_ops.h", "tensorflow/cc/ops/math_ops.h",
        "tensorflow/cc/ops/nn_ops.h", "tensorflow/cc/ops/parsing_ops.h", "tensorflow/cc/ops/random_ops.h",
        "tensorflow/cc/ops/sparse_ops.h", "tensorflow/cc/ops/state_ops.h", "tensorflow/cc/ops/string_ops.h",
        "tensorflow/cc/ops/summary_ops.h", "tensorflow/cc/ops/training_ops.h", "tensorflow/cc/ops/user_ops.h",
        "tensorflow_adapters.h"}, link = "tensorflow"),
            target = "org.bytedeco.javacpp.tensorflow", helper = "org.bytedeco.javacpp.helper.tensorflow")
public class tensorflow implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("tensorflow_adapters.h").skip())
               .put(new Info("TF_FALLTHROUGH_INTENDED", "TF_ATTRIBUTE_NORETURN", "TF_ATTRIBUTE_NOINLINE",
                             "TF_ATTRIBUTE_UNUSED", "TF_ATTRIBUTE_COLD", "TF_PACKED", "TF_MUST_USE_RESULT").cppTypes().annotations())
               .put(new Info("TF_CHECK_OK", "TF_QCHECK_OK").cppTypes("void", "tensorflow::Status"))
               .put(new Info("TF_DISALLOW_COPY_AND_ASSIGN").cppText("#define TF_DISALLOW_COPY_AND_ASSIGN(TypeName)"))
               .put(new Info("SWIG").define())
               .put(new Info("long long").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long..."))
               .put(new Info("std::initializer_list").skip())
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::condition_variable", "std::mutex", "std::unique_lock<std::mutex>", "tensorflow::condition_variable", "tensorflow::mutex_lock").cast().pointerTypes("Pointer"))

               .put(new Info("google::protobuf::int8", "google::protobuf::uint8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("google::protobuf::int16", "google::protobuf::uint16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("google::protobuf::int32", "google::protobuf::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("google::protobuf::int64", "google::protobuf::uint64").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("google::protobuf::Arena", "google::protobuf::Descriptor", "google::protobuf::EnumDescriptor", "google::protobuf::Message",
                             "google::protobuf::Metadata", "google::protobuf::io::CodedInputStream", "google::protobuf::io::CodedOutputStream").cast().pointerTypes("Pointer"))
               .put(new Info("google::protobuf::Map", "google::protobuf::RepeatedField", "google::protobuf::RepeatedPtrField").skip())

               .put(new Info("tensorflow::core::RefCounted").cast().pointerTypes("Pointer"))
               .put(new Info("tensorflow::ConditionResult").cast().valueTypes("int"))
               .put(new Info("tensorflow::protobuf::Message", "tensorflow::protobuf::MessageLite").cast().pointerTypes("Pointer"))

               .put(new Info("basic/containers").cppTypes("tensorflow::gtl::InlinedVector"))
               .put(new Info("tensorflow::DataType").cast().valueTypes("int").pointerTypes("IntPointer"))
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::DataType,4>").pointerTypes("DataTypeVector").define())
               .put(new Info("tensorflow::DataTypeSlice")/*.cast()*/.pointerTypes("DataTypeVector"))

               .put(new Info("tensorflow::Tensor").base("AbstractTensor"))
               .put(new Info("tensorflow::Session").base("AbstractSession"))
               .put(new Info("tensorflow::Session::~Session()").javaText("/** Calls {@link tensorflow#NewSession(SessionOptions)} and registers a deallocator. */\n"
                                                                       + "public Session(SessionOptions options) { super(options); }"))
               .put(new Info("std::vector<tensorflow::Tensor>").pointerTypes("TensorVector").define())
               .put(new Info("std::vector<tensorflow::TensorShape>").pointerTypes("TensorShapeVector").define())
               .put(new Info("std::vector<tensorflow::NodeBuilder::NodeOut>").pointerTypes("NodeOutVector").define())
               .put(new Info("std::vector<tensorflow::Node*>").pointerTypes("NodeVector").define())
               .put(new Info("tensorflow::ops::NodeOut").valueTypes("@ByVal NodeBuilder.NodeOut", "Node"))
               .put(new Info("tensorflow::NodeBuilder::NodeOut").pointerTypes("NodeBuilder.NodeOut"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::TensorShape>")/*.cast()*/.pointerTypes("TensorShapeVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::ops::NodeOut>")/*.cast()*/.pointerTypes("NodeOutVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::Node*>")/*.cast()*/.pointerTypes("NodeVector"))
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NeighborIter>").pointerTypes("NeighborIterRange").define())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NeighborIter>()").skip())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NodeIter>").pointerTypes("NodeIterRange").define())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NodeIter>()").skip())

               .put(new Info("std::vector<std::pair<std::string,tensorflow::Tensor> >").pointerTypes("StringTensorPairVector").define())
               .put(new Info("std::pair<tensorflow::EdgeSet::const_iterator,bool>").pointerTypes("EdgeSetBoolPair").define())
               .put(new Info("tensorflow::EdgeSet::const_iterator").pointerTypes("EdgeSetIterator"))

               .put(new Info("std::function<void()>").pointerTypes("Fn"))
               .put(new Info("std::function<tensorflow::OpDef(void)>").pointerTypes("OpDefFunc"))

               .put(new Info("tensorflow::gtl::ArraySlice").annotations("@ArraySlice"))
               .put(new Info("tensorflow::StringPiece").annotations("@StringPiece").valueTypes("BytePointer", "String").pointerTypes("BytePointer"))
               .put(new Info("tensorflow::ops::Const(tensorflow::StringPiece, tensorflow::GraphDefBuilder::Options&)")
                       .javaText("@Namespace(\"tensorflow::ops\") public static native Node Const("
                               + "@Cast({\"\", \"tensorflow::StringPiece&\"}) @StringPiece String s, @Const @ByRef GraphDefBuilder.Options options);"));
    }

    public static class Fn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Fn(Pointer p) { super(p); }
        protected Fn() { allocate(); }
        private native void allocate();
        public native void call();
    }

    public static class OpDefFunc extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    OpDefFunc(Pointer p) { super(p); }
        protected OpDefFunc() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("tensorflow::OpDef*") Pointer call();
    }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"tensorflow::gtl::ArraySlice", "&"}) @Adapter("ArraySliceAdapter")
    public @interface ArraySlice { String value() default ""; }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Adapter("StringPieceAdapter") public @interface StringPiece { }
}
