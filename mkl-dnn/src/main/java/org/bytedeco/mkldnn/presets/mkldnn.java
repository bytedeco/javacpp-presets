/*
 * Copyright (C) 2018 Samuel Audet
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

package org.bytedeco.mkldnn.presets;

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
            value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
            compiler = "cpp11",
            define = {"GENERIC_EXCEPTION_CLASS mkldnn::error"},
            include = {"mkldnn_types.h", /*"mkldnn_debug.h",*/ "mkldnn.h", "mkldnn.hpp"},
            link = "mkldnn@.0", preload = "libmkldnn")},
    target = "org.bytedeco.mkldnn", global = "org.bytedeco.mkldnn.global.mkldnn")
public class mkldnn implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap//.put(new Info().enumerate())
               .put(new Info("MKLDNN_HELPER_DLL_IMPORT", "MKLDNN_HELPER_DLL_EXPORT", "MKLDNN_API").cppTypes().annotations())
               .put(new Info("MKLDNN_DEPRECATED").cppText("#define MKLDNN_DEPRECATED deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))
               .put(new Info("DOXYGEN_SHOULD_SKIP_THIS").define())

	       .put(new Info("MKLDNN_MEMORY_ALLOCATE", "MKLDNN_MEMORY_NONE", "mkldnn::rnn_flags", "mkldnn::query", "mkldnn::memory::format_kind", "mkldnn::memory::format_tag", "cl_mem", "cl_device_id", "cl_command_queue", "cl_context", "cl_device_id", "undef").skip())
  
	       .put(new Info("mkldnn_dims_t").cppTypes("int64_t * const"))
	       .put(new Info("mkldnn_dim_t").cppTypes("int64_t"))
	       .put(new Info("int64_t const**", "const mkldnn_dims_t").cast().pointerTypes("CLongPointer"))
               .put(new Info("mkldnn_engine_t").valueTypes("mkldnn_engine").pointerTypes("@ByPtrPtr mkldnn_engine", "@Cast(\"mkldnn_engine_t*\") PointerPointer"))
               .put(new Info("const_mkldnn_engine_t").valueTypes("@Const mkldnn_engine").pointerTypes("@Const @ByPtrPtr mkldnn_engine", "@Cast(\"const_mkldnn_engine_t*\") PointerPointer"))
               .put(new Info("mkldnn_primitive_desc_iterator_t").valueTypes("mkldnn_primitive_desc_iterator").pointerTypes("@ByPtrPtr mkldnn_primitive_desc_iterator", "@Cast(\"mkldnn_primitive_desc_iterator_t*\") PointerPointer"))
               .put(new Info("const_mkldnn_primitive_desc_iterator_t").valueTypes("@Const mkldnn_primitive_desc_iterator").pointerTypes("@Const @ByPtrPtr mkldnn_primitive_desc_iterator", "@Cast(\"const_mkldnn_primitive_desc_iterator_t*\") PointerPointer"))
               .put(new Info("mkldnn_primitive_desc_t", "memory::desc").valueTypes("mkldnn_primitive_desc").pointerTypes("@ByPtrPtr mkldnn_primitive_desc", "@Cast(\"mkldnn_primitive_desc_t*\") PointerPointer"))
               .put(new Info("const_mkldnn_primitive_desc_t").valueTypes("@Const mkldnn_primitive_desc").pointerTypes("@Const @ByPtrPtr mkldnn_primitive_desc", "@Cast(\"const_mkldnn_primitive_desc_t*\") PointerPointer"))
               .put(new Info("mkldnn_primitive_attr_t").valueTypes("mkldnn_primitive_attr").pointerTypes("@ByPtrPtr mkldnn_primitive_attr", "@Cast(\"mkldnn_primitive_attr_t*\") PointerPointer"))
               .put(new Info("const_mkldnn_primitive_attr_t").valueTypes("@Const mkldnn_primitive_attr").pointerTypes("@Const @ByPtrPtr mkldnn_primitive_attr", "@Cast(\"const_mkldnn_primitive_attr_t*\") PointerPointer"))
               .put(new Info("mkldnn_post_ops_t").valueTypes("mkldnn_post_ops").pointerTypes("@ByPtrPtr mkldnn_post_ops", "@Cast(\"mkldnn_post_ops_t*\") PointerPointer"))
               .put(new Info("const_mkldnn_post_ops_t").valueTypes("@Const mkldnn_post_ops").pointerTypes("@Const @ByPtrPtr mkldnn_post_ops", "@Cast(\"const_mkldnn_post_ops_t*\") PointerPointer"))

	       .put(new Info("mkldnn_primitive_t").valueTypes("mkldnn_primitive").pointerTypes("@ByPtrPtr mkldnn_primitive", "@Cast(\"mkldnn_primitive_t*\") PointerPointer"))
	       .put(new Info("mkldnn_memory_t").valueTypes("mkldnn_memory").pointerTypes("@ByPtrPtr mkldnn_memory", "@Cast(\"mkldnn_memory_t*\") PointerPointer"))

	       .put(new Info("const_mkldnn_primitive_t").valueTypes("@Const mkldnn_primitive").pointerTypes("@Const @ByPtrPtr mkldnn_primitive", "@Cast(\"const_mkldnn_primitive_t*\") PointerPointer"))
	       .put(new Info("const_mkldnn_memory_t").valueTypes("@Const mkldnn_memory").pointerTypes("@Const @ByPtrPtr mkldnn_memory", "@Cast(\"const_mkldnn_memory_t*\") PointerPointer"))
               .put(new Info("mkldnn_stream_t").valueTypes("mkldnn_stream").pointerTypes("@ByPtrPtr mkldnn_stream", "@Cast(\"mkldnn_stream_t*\") PointerPointer"))
               .put(new Info("const_mkldnn_stream_t").valueTypes("@Const mkldnn_stream").pointerTypes("@Const @ByPtrPtr mkldnn_stream", "@Cast(\"const_mkldnn_stream_t*\") PointerPointer"))


	      //.put(new Info("long long").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]")) 
//               .put(new Info("long long * const", "long * const").pointerTypes("@Cast(\"long long * const\") CLongPointer"))
 
	       .put(new Info("mkldnn::primitive_desc").pointerTypes("org.bytedeco.mkldnn.primitive_desc"))
               .put(new Info("std::vector<const_mkldnn_primitive_desc_t>").annotations("@StdVector @Cast(\"const_mkldnn_primitive_desc_t*\")").pointerTypes("PointerPointer"))
//               .put(new Info("mkldnn::primitive::at").pointerTypes("primitive.at").define())
//               .put(new Info("mkldnn::memory::primitive_desc").pointerTypes("memoryPrimitive_desc").define())

               .put(new Info("std::vector<mkldnn_primitive_desc_t>",
                             "std::vector<const_mkldnn_primitive_desc_t>").cast().pointerTypes("mkldnn_primitive_desc_vector").define())
               .put(new Info("std::vector<mkldnn::primitive>").pointerTypes("primitive_vector").define())
//               .put(new Info("std::vector<mkldnn::primitive::at>").pointerTypes("primitive_at_vector").define())
//               .put(new Info("std::vector<mkldnn::memory::primitive_desc>").pointerTypes("memory_primitive_desc_vector").define())

               .put(new Info("mkldnn::handle<mkldnn_engine_t>").pointerTypes("mkldnn_engine_handle"))
	       .put(new Info("mkldnn::handle<mkldnn_memory_t>").pointerTypes("mkldnn_memory_handle"))
               .put(new Info("mkldnn::handle<mkldnn_primitive_desc_t>").pointerTypes("mkldnn_primitive_desc_handle"))
               .put(new Info("mkldnn::handle<mkldnn_primitive_attr_t>").pointerTypes("mkldnn_primitive_attr_handle"))
               .put(new Info("mkldnn::handle<mkldnn_post_ops_t>").pointerTypes("mkldnn_post_ops_handle"))
               .put(new Info("mkldnn::handle<mkldnn_primitive_t>").pointerTypes("mkldnn_primitive_handle"))
               .put(new Info("mkldnn::handle<mkldnn_stream_t>").pointerTypes("mkldnn_stream_handle"))

	       .put(new Info("std::unordered_map<int,mkldnn::memory>").pointerTypes("IntMemoryMap").define())
               .put(new Info("mkldnn::primitive::get_primitive_desc").javaNames("get_mkldnn_primitive_desc"))
               .put(new Info("mkldnn::eltwise_forward::desc<float>",
                             "mkldnn::eltwise_backward::desc<float>",
                             "mkldnn::batch_normalization_forward::desc<float>",
                             "mkldnn::batch_normalization_backward::desc<float>").javaNames("desc").skipDefaults())

               .put(new Info("mkldnn::rnn_cell::desc::operator const mkldnn_rnn_cell_desc_t*()").javaText(
                         "public native @Name(\"operator const mkldnn_rnn_cell_desc_t*\") @Const mkldnn_rnn_cell_desc_t as_mkldnn_rnn_cell_desc_t();\n"))

               .put(new Info("mkldnn_stream_kind_t::mkldnn_any_stream").javaNames("mkldnn_any_stream"))
               .put(new Info("mkldnn_stream_kind_t::mkldnn_eager").javaNames("mkldnn_eager"))
               .put(new Info("mkldnn_stream_kind_t::mkldnn_lazy").javaNames("mkldnn_lazy"));
    }
}
