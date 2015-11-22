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

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
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
@Properties(target = "org.bytedeco.javacpp.tensorflow", value =
    @Platform(value = {"linux", "macosx"}, compiler = "cpp11", include = {"tensorflow/core/platform/default/integral_types.h",
        /* "tensorflow/core/platform/default/mutex.h", "tensorflow/core/lib/core/refcount.h", "tensorflow/core/lib/gtl/array_slice.h", */
        "tensorflow/core/platform/default/dynamic_annotations.h", "tensorflow/core/platform/port.h",
        "tensorflow/core/lib/core/stringpiece.h", "tensorflow/core/lib/core/error_codes.pb.h",
        "tensorflow/core/platform/logging.h", "tensorflow/core/public/status.h", "tensorflow/core/platform/protobuf.h",
        "tensorflow/core/public/env.h", "tensorflow/core/framework/config.pb.h", "tensorflow/core/public/session_options.h",
        "tensorflow/core/framework/allocation_description.pb.h", "tensorflow/core/framework/allocator.h",
        "tensorflow/core/framework/tensor_shape.pb.h", "tensorflow/core/framework/types.pb.h", "tensorflow/core/framework/tensor.pb.h",
        "tensorflow/core/framework/tensor_description.pb.h", "tensorflow/core/framework/tensor_types.h",
        "tensorflow/core/public/tensor_shape.h", "tensorflow/core/public/tensor.h", "tensorflow/core/framework/attr_value.pb.h",
        "tensorflow/core/framework/op_def.pb.h", "tensorflow/core/framework/function.pb.h", "tensorflow/core/framework/graph.pb.h",
        "tensorflow/core/public/session.h", "tensorflow/core/public/tensor_c_api.h", "tensorflow/core/framework/op_def.pb.h",
        "tensorflow/core/framework/op_def_builder.h", "tensorflow/core/framework/op_def_util.h", "tensorflow/core/framework/op.h",
        "tensorflow/core/framework/types.h", "tensorflow/core/graph/edgeset.h", "tensorflow/core/lib/gtl/iterator_range.h",
        "tensorflow/core/graph/graph.h", "tensorflow/core/graph/node_builder.h", "tensorflow/core/graph/graph_def_builder.h"}, link = "tensorflow"))
public class tensorflow implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("TF_FALLTHROUGH_INTENDED", "TF_ATTRIBUTE_NORETURN", "TF_ATTRIBUTE_NOINLINE",
                             "TF_ATTRIBUTE_UNUSED", "TF_ATTRIBUTE_COLD", "TF_PACKED", "TF_MUST_USE_RESULT").cppTypes().annotations())
               .put(new Info("TF_DISALLOW_COPY_AND_ASSIGN").cppText("#define TF_DISALLOW_COPY_AND_ASSIGN(TypeName)"))
               .put(new Info("SWIG").define())
               .put(new Info("long long").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
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
               .put(new Info("tensorflow::StringPiece::iterator").cast().valueTypes("BytePointer").pointerTypes("@ByPtrPtr BytePointer"))

               .put(new Info("basic/containers").cppTypes("tensorflow::gtl::InlinedVector"))
               .put(new Info("tensorflow::DataType").cast().valueTypes("int").pointerTypes("IntPointer"))
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::DataType,4>").pointerTypes("DataTypeVector").define())
               .put(new Info("tensorflow::DataTypeSlice")./*cast().*/pointerTypes("DataTypeVector"))

               .put(new Info("std::initializer_list<long long>", "tensorflow::TensorShape::dim_sizes()").skip())
               .put(new Info("std::vector<tensorflow::Tensor>").pointerTypes("TensorVector").define())
               .put(new Info("std::vector<tensorflow::TensorShape>").pointerTypes("TensorShapeVector").define())
               .put(new Info("std::vector<tensorflow::NodeBuilder::NodeOut>").pointerTypes("NodeOutVector").define())
               .put(new Info("std::vector<tensorflow::Node*>").pointerTypes("NodeVector").define())
               .put(new Info("tensorflow::NodeBuilder::NodeOut").pointerTypes("NodeBuilder.NodeOut"))
               .put(new Info("tensorflow::gtl::ArraySlice<long long>").annotations("@Cast({\"\", \"const std::vector<long long>&\"}) @StdVector")
                                                                      ./*cast().*/pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::TensorShape>")./*cast().*/pointerTypes("TensorShapeVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::NodeBuilder::NodeOut>")./*cast().*/pointerTypes("NodeOutVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::Node*>")./*cast().*/pointerTypes("NodeVector"))
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NeighborIter>").pointerTypes("NeighborIterRange").define())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NeighborIter>()").skip())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NodeIter>").pointerTypes("NodeIterRange").define())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NodeIter>()").skip())

               .put(new Info("std::vector<std::pair<std::string,tensorflow::Tensor> >").pointerTypes("StringTensorPairVector").define())
               .put(new Info("std::pair<tensorflow::EdgeSet::const_iterator,bool>").pointerTypes("EdgeSetBoolPair").define())
               .put(new Info("tensorflow::EdgeSet::const_iterator").pointerTypes("EdgeSetIterator"))

               .put(new Info("std::function<void()>").pointerTypes("Fn"))
               .put(new Info("std::function<tensorflow::OpDef(void)>").pointerTypes("OpDefFunc"));
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
}
