/*
 * Copyright (C) 2018-2021 Samuel Audet, Alexander Merritt
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

package org.bytedeco.dnnl.presets;

import java.util.List;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.opencl.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = OpenCL.class,
    value = {
        @Platform(
            value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "macosx-x86_64", "windows-x86_64"},
            compiler = "cpp11",
            define = {"GENERIC_EXCEPTION_CLASS dnnl::error", "GENERIC_EXCEPTION_TOSTRING toStdString().c_str()"},
            include = {"oneapi/dnnl/dnnl_types.h", "oneapi/dnnl/dnnl_config.h", /*"oneapi/dnnl/dnnl_debug.h",*/ "oneapi/dnnl/dnnl_version.h", "oneapi/dnnl/dnnl.h", "oneapi/dnnl/dnnl.hpp"},
            link = "dnnl@.2", preload = {"gomp@.1", "iomp5", "omp", "tbb@.2"}, resource = {"include", "lib"}
        ),
    },
    target = "org.bytedeco.dnnl",
    global = "org.bytedeco.dnnl.global.dnnl"
)
public class dnnl implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "dnnl"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info().enumerate())
               .put(new Info("DNNL_HELPER_DLL_IMPORT", "DNNL_HELPER_DLL_EXPORT", "DNNL_API",
                             "DNNL_MEMORY_NONE", "DNNL_MEMORY_ALLOCATE").cppTypes().annotations())
               .put(new Info("DNNL_VERSION_HASH", "DNNL_RUNTIME_F32_VAL").translate(false))
               .put(new Info("DNNL_RUNTIME_DIM_VAL").cppTypes("long long").translate(false))
               .put(new Info("DNNL_DEPRECATED").cppText("#define DNNL_DEPRECATED deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))
               .put(new Info("DOXYGEN_SHOULD_SKIP_THIS").define())
               .put(new Info("DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL",
                             "DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL",
                             "DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL").define(false))
               .put(new Info("DNNL_RUNTIME_F32_VAL_REP").skip())

               .put(new Info("dnnl_dims_t").cppTypes("int64_t* const"))
               .put(new Info("dnnl_memory_t").valueTypes("dnnl_memory").pointerTypes("@ByPtrPtr dnnl_memory", "@Cast(\"dnnl_memory_t*\") PointerPointer"))
               .put(new Info("const_dnnl_memory_t").valueTypes("@Const dnnl_memory").pointerTypes("@Const @ByPtrPtr dnnl_memory", "@Cast(\"const_dnnl_memory_t*\") PointerPointer"))
               .put(new Info("dnnl_engine_t").valueTypes("dnnl_engine").pointerTypes("@ByPtrPtr dnnl_engine", "@Cast(\"dnnl_engine_t*\") PointerPointer"))
               .put(new Info("const_dnnl_engine_t").valueTypes("@Const dnnl_engine").pointerTypes("@Const @ByPtrPtr dnnl_engine", "@Cast(\"const_dnnl_engine_t*\") PointerPointer"))
               .put(new Info("dnnl_primitive_desc_iterator_t").valueTypes("dnnl_primitive_desc_iterator").pointerTypes("@ByPtrPtr dnnl_primitive_desc_iterator", "@Cast(\"dnnl_primitive_desc_iterator_t*\") PointerPointer"))
               .put(new Info("const_dnnl_primitive_desc_iterator_t").valueTypes("@Const dnnl_primitive_desc_iterator").pointerTypes("@Const @ByPtrPtr dnnl_primitive_desc_iterator", "@Cast(\"const_dnnl_primitive_desc_iterator_t*\") PointerPointer"))
               .put(new Info("dnnl_primitive_desc_t").valueTypes("dnnl_primitive_desc").pointerTypes("@ByPtrPtr dnnl_primitive_desc", "@Cast(\"dnnl_primitive_desc_t*\") PointerPointer"))
               .put(new Info("const_dnnl_primitive_desc_t").valueTypes("@Const dnnl_primitive_desc").pointerTypes("@Const @ByPtrPtr dnnl_primitive_desc", "@Cast(\"const_dnnl_primitive_desc_t*\") PointerPointer"))
               .put(new Info("dnnl_primitive_attr_t").valueTypes("dnnl_primitive_attr").pointerTypes("@ByPtrPtr dnnl_primitive_attr", "@Cast(\"dnnl_primitive_attr_t*\") PointerPointer"))
               .put(new Info("const_dnnl_primitive_attr_t").valueTypes("@Const dnnl_primitive_attr").pointerTypes("@Const @ByPtrPtr dnnl_primitive_attr", "@Cast(\"const_dnnl_primitive_attr_t*\") PointerPointer"))
               .put(new Info("dnnl_post_ops_t").valueTypes("dnnl_post_ops").pointerTypes("@ByPtrPtr dnnl_post_ops", "@Cast(\"dnnl_post_ops_t*\") PointerPointer"))
               .put(new Info("const_dnnl_post_ops_t").valueTypes("@Const dnnl_post_ops").pointerTypes("@Const @ByPtrPtr dnnl_post_ops", "@Cast(\"const_dnnl_post_ops_t*\") PointerPointer"))
               .put(new Info("dnnl_primitive_t").valueTypes("dnnl_primitive").pointerTypes("@ByPtrPtr dnnl_primitive", "@Cast(\"dnnl_primitive_t*\") PointerPointer"))
               .put(new Info("const_dnnl_primitive_t").valueTypes("@Const dnnl_primitive").pointerTypes("@Const @ByPtrPtr dnnl_primitive", "@Cast(\"const_dnnl_primitive_t*\") PointerPointer"))
               .put(new Info("dnnl_stream_t").valueTypes("dnnl_stream").pointerTypes("@ByPtrPtr dnnl_stream", "@Cast(\"dnnl_stream_t*\") PointerPointer"))
               .put(new Info("const_dnnl_stream_t").valueTypes("@Const dnnl_stream").pointerTypes("@Const @ByPtrPtr dnnl_stream", "@Cast(\"const_dnnl_stream_t*\") PointerPointer"))
//               .put(new Info("dnnl_stream_attr_t").valueTypes("dnnl_stream_attr").pointerTypes("@ByPtrPtr dnnl_stream_attr", "@Cast(\"dnnl_stream_attr_t*\") PointerPointer"))
//               .put(new Info("const_dnnl_stream_attr_t").valueTypes("@Const dnnl_stream_attr").pointerTypes("@Const @ByPtrPtr dnnl_stream_attr", "@Cast(\"const_dnnl_stream_attr_t*\") PointerPointer"))

               .put(new Info("dnnl::primitive_desc").pointerTypes("org.bytedeco.dnnl.primitive_desc"))
               .put(new Info("dnnl::memory::dims").annotations("@Cast({\"dnnl_dim_t*\", \"std::vector<dnnl_dim_t>&\"}) @StdVector(\"dnnl_dim_t\")").pointerTypes("LongPointer", "LongBuffer", "long[]"))
//               .put(new Info("std::vector<const_dnnl_primitive_desc_t>").annotations("@StdVector @Cast(\"const_dnnl_primitive_desc_t*\")").pointerTypes("PointerPointer"))
//               .put(new Info("dnnl::primitive::at").pointerTypes("primitive.at").define())
//               .put(new Info("dnnl::memory::primitive_desc").pointerTypes("memory.primitive_desc").define())
//               .put(new Info("std::vector<int64_t>", "std::vector<dnnl_dim_t>", "dnnl::memory::dims").pointerTypes("memory_dims").define())
               .put(new Info("std::vector<dnnl_primitive_desc_t>",
                             "std::vector<const_dnnl_primitive_desc_t>").cast().pointerTypes("dnnl_primitive_desc_vector").define())
               .put(new Info("std::vector<dnnl::primitive>").pointerTypes("primitive_vector").define())
//               .put(new Info("std::vector<dnnl::primitive::at>").pointerTypes("primitive_at_vector").define())
//               .put(new Info("std::vector<dnnl::memory::primitive_desc>").pointerTypes("memory_primitive_desc_vector").define())

               .put(new Info("dnnl::memory::data_type").valueTypes("memory.data_type").enumerate())
               .put(new Info("dnnl::memory::format_tag").valueTypes("memory.format_tag").enumerate())
               .put(new Info("dnnl::stream::flags").valueTypes("stream.flags").enumerate())
               .put(new Info("dnnl::primitive::kind").valueTypes("primitive.kind").enumerate())

               .put(new Info("dnnl::handle_traits", "dnnl::primitive_attr(dnnl_primitive_attr_t)",
                             "dnnl::reorder::primitive_desc(dnnl_primitive_desc_t)",
                             "dnnl::concat::primitive_desc(dnnl_primitive_desc_t)",
                             "dnnl::sum::primitive_desc(dnnl_primitive_desc_t)").skip())
               .put(new Info("dnnl::handle<dnnl_engine_t>", "dnnl::handle<dnnl_engine_t,traits>").pointerTypes("dnnl_engine_handle"))
               .put(new Info("dnnl::handle<dnnl_memory_t>", "dnnl::handle<dnnl_memory_t,traits>").pointerTypes("dnnl_memory_handle"))
               .put(new Info("dnnl::handle<dnnl_primitive_desc_t>", "dnnl::handle<dnnl_primitive_desc_t,traits>").pointerTypes("dnnl_primitive_desc_handle"))
               .put(new Info("dnnl::handle<dnnl_primitive_attr_t>", "dnnl::handle<dnnl_primitive_attr_t,traits>").pointerTypes("dnnl_primitive_attr_handle"))
               .put(new Info("dnnl::handle<dnnl_post_ops_t>", "dnnl::handle<dnnl_post_ops_t,traits>").pointerTypes("dnnl_post_ops_handle"))
               .put(new Info("dnnl::handle<dnnl_primitive_t>", "dnnl::handle<dnnl_primitive_t,traits>").pointerTypes("dnnl_primitive_handle"))
               .put(new Info("dnnl::handle<dnnl_stream_t>", "dnnl::handle<dnnl_stream_t,traits>").pointerTypes("dnnl_stream_handle"))
//               .put(new Info("dnnl::handle<dnnl_stream_attr_t>", "dnnl::handle<dnnl_stream_attr_t,traits>").pointerTypes("dnnl_stream_attr_handle"))

               .put(new Info("std::unordered_map<int,dnnl::memory>").pointerTypes("IntMemoryMap").define())
               .put(new Info("dnnl::primitive::get_primitive_desc").javaNames("get_dnnl_primitive_desc"))
               .put(new Info("dnnl::eltwise_forward::desc<float>",
                             "dnnl::eltwise_backward::desc<float>",
                             "dnnl::batch_normalization_forward::desc<float>",
                             "dnnl::batch_normalization_backward::desc<float>").javaNames("desc"))

               .put(new Info("dnnl::rnn_cell::desc::operator const dnnl_rnn_cell_desc_t*()").javaText(
                         "public native @Name(\"operator const dnnl_rnn_cell_desc_t*\") @Const dnnl_rnn_cell_desc_t as_dnnl_rnn_cell_desc_t();\n"))

               .put(new Info("dnnl_stream_kind_t::dnnl_any_stream").javaNames("dnnl_any_stream"))
               .put(new Info("dnnl_stream_kind_t::dnnl_eager").javaNames("dnnl_eager"))
               .put(new Info("dnnl_stream_kind_t::dnnl_lazy").javaNames("dnnl_lazy"));
    }
}
