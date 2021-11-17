/*
 * Copyright (C) 2020-2021 Samuel Audet
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
package org.bytedeco.tvm.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.dnnl.presets.*;
import org.bytedeco.llvm.presets.*;
import org.bytedeco.mkl.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = {dnnl.class, LLVM.class, mkl_rt.class},
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            compiler = "cpp14",
            define = {"GENERIC_EXCEPTION_CLASS std::exception", "GENERIC_EXCEPTION_TOSTRING what()", "DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>"},
            exclude = {"<polly/LinkAllPasses.h>", "<FullOptimization.h>", "<NamedMetadataOperations.h>"},
            include = {
                "dlpack/dlpack.h",
                "dmlc/base.h",
                "dmlc/logging.h",
                "dmlc/io.h",
                "dmlc/type_traits.h",
                "dmlc/endian.h",
                "dmlc/serializer.h",
                "tvm/runtime/c_runtime_api.h",
                "tvm/runtime/data_type.h",
                "tvm/runtime/object.h",
                "tvm/runtime/memory.h",
//                "tvm/runtime/container.h",
                "tvm/runtime/container/base.h",
                "tvm/runtime/container/adt.h",
                "tvm/runtime/container/array.h",
                "tvm/runtime/container/closure.h",
                "tvm/runtime/container/optional.h",
                "tvm/runtime/container/map.h",
                "tvm/runtime/container/shape_tuple.h",
                "tvm/runtime/container/string.h",
                "tvm/runtime/ndarray.h",
                "tvm/runtime/serializer.h",
                "tvm/runtime/module.h",
                "tvm/runtime/packed_func.h",
                "tvm/runtime/registry.h",
                "org_apache_tvm_native_c_api.cc",
            },
            link = "tvm_runtime#"
        ),
        @Platform(
            value = "linux",
            preload = {"tvm_runtime:python/tvm/libtvm.so", "tvm_runtime:libtvm_runtime.so"}
        ),
        @Platform(
            value = "macosx",
            preload = {"tvm_runtime:python/tvm/libtvm.dylib", "tvm_runtime:libtvm_runtime.dylib"}
        ),
        @Platform(
            value = "windows",
            preload = {"tvm_runtime:python/tvm/tvm.dll#tvm_runtime.dll", "tvm_runtime:tvm_runtime.dll"}
        ),
        @Platform(
            value = {"linux", "macosx", "windows"},
            extension = "-gpu"
        ),
    },
    target = "org.bytedeco.tvm",
    global = "org.bytedeco.tvm.global.tvm_runtime"
)
public class tvm_runtime implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "tvm"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || extension == null || !extension.equals("-gpu")) {
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
        infoMap.put(new Info("org_apache_tvm_native_c_api.cc").skip())
               .put(new Info("DLPACK_EXTERN_C", "DLPACK_DLL", "DMLC_STRICT_CXX11", "DMLC_CXX11_THREAD_LOCAL",
                             "DMLC_ATTRIBUTE_UNUSED", "DMLC_SUPPRESS_UBSAN", "DMLC_NO_INLINE", "DMLC_ALWAYS_INLINE", "TVM_WEAK", "TVM_DLL",
                             "TVM_ATTRIBUTE_UNUSED", "TVM_OBJECT_REG_VAR_DEF", "TVM_ADD_FILELINE", "TVM_ALWAYS_INLINE",
                             "TVM_FUNC_REG_VAR_DEF").cppTypes().annotations())
               .put(new Info("__APPLE__", "_MSC_VER", "defined(_MSC_VER)", "defined(_MSC_VER) && _MSC_VER < 1900").define(false))
               .put(new Info("defined DMLC_USE_LOGGING_LIBRARY", "DMLC_LOG_STACK_TRACE", "DMLC_CMAKE_LITTLE_ENDIAN").define(true))
               .put(new Info("DMLC_LITTLE_ENDIAN", "DMLC_IO_NO_ENDIAN_SWAP").translate(false))
               .put(new Info("auto", "std::equal_to", "std::initializer_list", "std::hash", "std::nullptr_t", "dmlc::Demangle",
                             "dmlc::DummyOStream", "dmlc::InputSplit::Create", "dmlc::SeekStream::CreateForRead", "dmlc::Stream::Create",
                             "dmlc::io::FileSystem::GetInstance", "tvm::runtime::NDArray::reset").skip())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::runtime_error", "std::basic_istream<char>", "std::basic_ostream<char>",
                             "tvm::runtime::MapNode::iterator", "tvm::runtime::MapNode::KVType").cast().pointerTypes("Pointer"))
               .put(new Info("tvm::runtime::MapNode::key_type", "tvm::runtime::MapNode::mapped_type",
                             "key_type", "mapped_type").pointerTypes("ObjectRef"))

               .put(new Info("tvm::runtime::DataType::TypeCode").enumerate().cppTypes("long long"))
               .put(new Info("tvm::runtime::ShapeTuple::index_type").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long..."))
               .put(new Info("TVMArgTypeCode::kTVMOpaqueHandle").javaNames("kTVMOpaqueHandle"))
               .put(new Info("TVMArrayHandle").valueTypes("@Cast(\"TVMArrayHandle\") DLTensor")
                                              .pointerTypes("@Cast(\"TVMArrayHandle*\") PointerPointer", "@ByPtrPtr @Cast(\"TVMArrayHandle*\") DLTensor"))

               .put(new Info("tvm::runtime::ObjectPtr<tvm::runtime::Object>").pointerTypes("ObjectPtr"))
               .put(new Info("tvm::runtime::ObjAllocatorBase<tvm::runtime::SimpleObjAllocator>",
                             "tvm::runtime::ObjAllocatorBase<SimpleObjAllocator>").pointerTypes("SimpleObjAllocatorBase"))
               .put(new Info("tvm::runtime::ObjAllocatorBase<tvm::runtime::SimpleObjAllocator>::make_object").skip())

               .put(new Info("tvm::runtime::ADTObj").pointerTypes("ADTObj"))
               .put(new Info("tvm::runtime::InplaceArrayBase<tvm::runtime::ADTObj,tvm::runtime::ObjectRef>",
                             "tvm::runtime::InplaceArrayBase<ADTObj,tvm::runtime::ObjectRef>").pointerTypes("ADTObjBase"))
               .put(new Info("tvm::runtime::ArrayNode").pointerTypes("ArrayNode"))
               .put(new Info("tvm::runtime::InplaceArrayBase<tvm::runtime::ArrayNode,tvm::runtime::ObjectRef>",
                             "tvm::runtime::InplaceArrayBase<ArrayNode,tvm::runtime::ObjectRef>").pointerTypes("ArrayNodeBase"))
               .put(new Info("tvm::runtime::SmallMapNode").pointerTypes("SmallMapNode"))
               .put(new Info("tvm::runtime::InplaceArrayBase<tvm::runtime::SmallMapNode,tvm::runtime::MapNode::KVType>",
                             "tvm::runtime::InplaceArrayBase<SmallMapNode,tvm::runtime::MapNode::KVType>").pointerTypes("SmallMapNodeBase"))

               .put(new Info("tvm::runtime::make_object<tvm::runtime::ArrayNode>").javaNames("makeArrayNode"))
               .put(new Info("tvm::runtime::ObjectPtr<tvm::runtime::ArrayNode>").pointerTypes("ArrayNodePtr"))
               .put(new Info("tvm::runtime::ObjectPtr<tvm::runtime::MapNode>").pointerTypes("MapNodePtr"))
               .put(new Info("tvm::runtime::Optional<tvm::runtime::String>").pointerTypes("TVMStringOptional"))
               .put(new Info("tvm::runtime::Object").pointerTypes("TVMObject"))
               .put(new Info("tvm::runtime::String").pointerTypes("TVMString"))
               .put(new Info("tvm::runtime::StringObj::FromStd", "tvm::runtime::ShapeTupleObj::FromStd",
                             "tvm::runtime::TVMMovableArgValueWithContext_", "llvm::StringRef").skip())

               .put(new Info("FDeleter").valueTypes("FDeleter"))
               .put(new Info("tvm::runtime::NDArray::operator ->").javaNames("accessDLTensor"))
               .put(new Info("tvm::runtime::NDArray::Container").pointerTypes("Container"))
               .put(new Info("tvm::runtime::NDArray::ContainerBase").pointerTypes("ContainerBase"))

               .put(new Info("tvm::runtime::PackedFunc::FType").pointerTypes("PackedFuncFType"))
               .put(new Info("tvm::runtime::PackedFunc::body()").javaText(
                        "public native @ByPtrPtr @Name(\"body().target<void(*)(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>\") PackedFuncFType body();"))
               .put(new Info("tvm::runtime::TVMArgsSetter::operator ()(size_t, const TObjectRef&)").javaText(
                        "public native @Name(\"operator ()\") void apply(@Cast(\"size_t\") long i, @Const @ByRef ObjectRef value);"));
    }

    public static class PackedFuncFType extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    PackedFuncFType(Pointer p) { super(p); }
        protected PackedFuncFType() { allocate(); }
        private native void allocate();
        public native void call(@ByVal @Cast("tvm::runtime::TVMArgs*") Pointer args, @Cast("tvm::runtime::TVMRetValue*") Pointer rv);
    }
}
