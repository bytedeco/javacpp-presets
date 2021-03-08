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

package org.bytedeco.opencl.presets;

import java.util.List;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            include = {"CL/opencl.h", "CL/cl_version.h", "CL/cl_platform.h", "CL/cl.h", /*"CL/cl_gl.h", "CL/cl_gl_ext.h", "CL/cl_ext.h"*/},
            resource = {"include", "lib"}
        ),
        @Platform(
            value = {"linux", "windows"},
            link = "OpenCL@.1"
        ),
        @Platform(
            value = "macosx",
            framework = "OpenCL"
        ),
    },
    target = "org.bytedeco.opencl",
    global = "org.bytedeco.opencl.global.OpenCL"
)
public class OpenCL implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "opencl"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CL_API_ENTRY", "CL_API_CALL", "CL_CALLBACK",
                             "CL_PROGRAM_STRING_DEBUG_INFO", "__CL_ANON_STRUCT__",
                             "CL_API_SUFFIX_COMMON", "CL_API_PREFIX_COMMON",
                             "CL_API_SUFFIX__VERSION_1_0", "CL_API_SUFFIX__VERSION_1_1",
                             "CL_API_SUFFIX__VERSION_1_2", "CL_API_SUFFIX__VERSION_2_0",
                             "CL_API_SUFFIX__VERSION_2_1", "CL_API_SUFFIX__VERSION_2_2",
                             "CL_API_SUFFIX__VERSION_3_0", "CL_API_SUFFIX__EXPERIMENTAL").cppTypes().annotations())
               .put(new Info("CL_NAN", "CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM").translate(false))
               .put(new Info("CL_HUGE_VAL", "nanf").skip())

               .put(new Info("CL_EXT_SUFFIX_DEPRECATED").cppText("#define CL_EXT_SUFFIX_DEPRECATED").cppTypes())
               .put(new Info("CL_EXT_PREFIX_DEPRECATED").cppText("#define CL_EXT_PREFIX_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED").cppText("#define CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED").cppTypes())
               .put(new Info("CL_EXT_PREFIX__VERSION_1_0_DEPRECATED").cppText("#define CL_EXT_PREFIX__VERSION_1_0_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED").cppText("#define CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED").cppTypes())
               .put(new Info("CL_EXT_PREFIX__VERSION_1_1_DEPRECATED").cppText("#define CL_EXT_PREFIX__VERSION_1_1_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED").cppText("#define CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED").cppTypes())
               .put(new Info("CL_EXT_PREFIX__VERSION_1_2_DEPRECATED").cppText("#define CL_EXT_PREFIX__VERSION_1_2_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED").cppText("#define CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED").cppTypes())
               .put(new Info("CL_EXT_PREFIX__VERSION_2_0_DEPRECATED").cppText("#define CL_EXT_PREFIX__VERSION_2_0_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_EXT_SUFFIX__VERSION_2_1_DEPRECATED").cppText("#define CL_EXT_SUFFIX__VERSION_2_1_DEPRECATED").cppTypes())
               .put(new Info("CL_EXT_PREFIX__VERSION_2_1_DEPRECATED").cppText("#define CL_EXT_PREFIX__VERSION_2_1_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_EXT_SUFFIX__VERSION_2_2_DEPRECATED").cppText("#define CL_EXT_SUFFIX__VERSION_2_2_DEPRECATED").cppTypes())
               .put(new Info("CL_EXT_PREFIX__VERSION_2_2_DEPRECATED").cppText("#define CL_EXT_PREFIX__VERSION_2_2_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_API_SUFFIX_DEPRECATED").cppText("#define CL_API_SUFFIX_DEPRECATED").cppTypes())
               .put(new Info("CL_API_PREFIX_DEPRECATED").cppText("#define CL_API_PREFIX_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_API_SUFFIX__VERSION_1_0_DEPRECATED").cppText("#define CL_API_SUFFIX__VERSION_1_0_DEPRECATED").cppTypes())
               .put(new Info("CL_API_PREFIX__VERSION_1_0_DEPRECATED").cppText("#define CL_API_PREFIX__VERSION_1_0_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_API_SUFFIX__VERSION_1_1_DEPRECATED").cppText("#define CL_API_SUFFIX__VERSION_1_1_DEPRECATED").cppTypes())
               .put(new Info("CL_API_PREFIX__VERSION_1_1_DEPRECATED").cppText("#define CL_API_PREFIX__VERSION_1_1_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_API_SUFFIX__VERSION_1_2_DEPRECATED").cppText("#define CL_API_SUFFIX__VERSION_1_2_DEPRECATED").cppTypes())
               .put(new Info("CL_API_PREFIX__VERSION_1_2_DEPRECATED").cppText("#define CL_API_PREFIX__VERSION_1_2_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_API_SUFFIX__VERSION_2_0_DEPRECATED").cppText("#define CL_API_SUFFIX__VERSION_2_0_DEPRECATED").cppTypes())
               .put(new Info("CL_API_PREFIX__VERSION_2_0_DEPRECATED").cppText("#define CL_API_PREFIX__VERSION_2_0_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_API_SUFFIX__VERSION_2_1_DEPRECATED").cppText("#define CL_API_SUFFIX__VERSION_2_1_DEPRECATED").cppTypes())
               .put(new Info("CL_API_PREFIX__VERSION_2_1_DEPRECATED").cppText("#define CL_API_PREFIX__VERSION_2_1_DEPRECATED deprecated").cppTypes())
               .put(new Info("CL_API_SUFFIX__VERSION_2_2_DEPRECATED").cppText("#define CL_API_SUFFIX__VERSION_2_2_DEPRECATED").cppTypes())
               .put(new Info("CL_API_PREFIX__VERSION_2_2_DEPRECATED").cppText("#define CL_API_PREFIX__VERSION_2_2_DEPRECATED deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("__CL_HAS_ANON_STRUCT__", "defined( __cl_uchar2__)",
                             "defined( __CL_CHAR2__)", "defined( __CL_CHAR4__)", "defined( __CL_CHAR8__ )", "defined( __CL_CHAR16__ )",
                             "defined( __CL_UCHAR2__)", "defined( __CL_UCHAR4__)", "defined( __CL_UCHAR8__ )", "defined( __CL_UCHAR16__ )",
                             "defined( __CL_SHORT2__)", "defined( __CL_SHORT4__)", "defined( __CL_SHORT8__ )", "defined( __CL_SHORT16__ )",
                             "defined( __CL_USHORT2__)", "defined( __CL_USHORT4__)", "defined( __CL_USHORT8__ )", "defined( __CL_USHORT16__ )",
                             "defined( __CL_HALF2__)", "defined( __CL_HALF4__)", "defined( __CL_HALF8__ )", "defined( __CL_HALF16__ )",
                             "defined( __CL_INT2__)", "defined( __CL_INT4__)", "defined( __CL_INT8__ )", "defined( __CL_INT16__ )",
                             "defined( __CL_UINT2__)", "defined( __CL_UINT4__)", "defined( __CL_UINT8__ )", "defined( __CL_UINT16__ )",
                             "defined( __CL_LONG2__)", "defined( __CL_LONG4__)", "defined( __CL_LONG8__ )", "defined( __CL_LONG16__ )",
                             "defined( __CL_ULONG2__)", "defined( __CL_ULONG4__)", "defined( __CL_ULONG8__ )", "defined( __CL_ULONG16__ )",
                             "defined( __CL_FLOAT2__)", "defined( __CL_FLOAT4__)", "defined( __CL_FLOAT8__ )", "defined( __CL_FLOAT16__ )",
                             "defined( __CL_DOUBLE2__)", "defined( __CL_DOUBLE4__)", "defined( __CL_DOUBLE8__ )", "defined( __CL_DOUBLE16__ )").define(false))

               .put(new Info("cl_platform_id").valueTypes("_cl_platform_id").pointerTypes("@Cast(\"cl_platform_id*\") PointerPointer", "@ByPtrPtr _cl_platform_id"))
               .put(new Info("const cl_platform_id").valueTypes("@Const _cl_platform_id").pointerTypes("@Cast(\"const cl_platform_id*\") PointerPointer", "@Cast(\"const cl_platform_id*\") @ByPtrPtr _cl_platform_id"))
               .put(new Info("cl_device_id").valueTypes("_cl_device_id").pointerTypes("@Cast(\"cl_device_id*\") PointerPointer", "@ByPtrPtr _cl_device_id"))
               .put(new Info("const cl_device_id").valueTypes("@Const _cl_device_id").pointerTypes("@Cast(\"const cl_device_id*\") PointerPointer", "@Cast(\"const cl_device_id*\") @ByPtrPtr _cl_device_id"))
               .put(new Info("cl_context").valueTypes("_cl_context").pointerTypes("@Cast(\"cl_context*\") PointerPointer", "@ByPtrPtr _cl_context"))
               .put(new Info("const cl_context").valueTypes("@Const _cl_context").pointerTypes("@Cast(\"const cl_context*\") PointerPointer", "@Cast(\"const cl_context*\") @ByPtrPtr _cl_context"))
               .put(new Info("cl_command_queue").valueTypes("_cl_command_queue").pointerTypes("@Cast(\"cl_command_queue*\") PointerPointer", "@ByPtrPtr _cl_command_queue"))
               .put(new Info("const cl_command_queue").valueTypes("@Const _cl_command_queue").pointerTypes("@Cast(\"const cl_command_queue*\") PointerPointer", "@Cast(\"const cl_command_queue*\") @ByPtrPtr _cl_command_queue"))
               .put(new Info("cl_mem").valueTypes("_cl_mem").pointerTypes("@Cast(\"cl_mem*\") PointerPointer", "@ByPtrPtr _cl_mem"))
               .put(new Info("const cl_mem").valueTypes("@Const _cl_mem").pointerTypes("@Cast(\"const cl_mem*\") PointerPointer", "@Cast(\"const cl_mem*\") @ByPtrPtr _cl_mem"))
               .put(new Info("cl_program").valueTypes("_cl_program").pointerTypes("@Cast(\"cl_program*\") PointerPointer", "@ByPtrPtr _cl_program"))
               .put(new Info("const cl_program").valueTypes("@Const _cl_program").pointerTypes("@Cast(\"const cl_program*\") PointerPointer", "@Cast(\"const cl_program*\") @ByPtrPtr _cl_program"))
               .put(new Info("cl_kernel").valueTypes("_cl_kernel").pointerTypes("@Cast(\"cl_kernel*\") PointerPointer", "@ByPtrPtr _cl_kernel"))
               .put(new Info("const cl_kernel").valueTypes("@Const cl_kernel").pointerTypes("@Cast(\"const cl_kernel*\") PointerPointer", "@Cast(\"const cl_kernel*\") @ByPtrPtr cl_kernel"))
               .put(new Info("cl_event").valueTypes("_cl_event").pointerTypes("@Cast(\"cl_event*\") PointerPointer", "@ByPtrPtr _cl_event"))
               .put(new Info("const cl_event").valueTypes("@Const _cl_event").pointerTypes("@Cast(\"const cl_event*\") PointerPointer", "@Cast(\"const cl_event*\") @ByPtrPtr _cl_event"))
               .put(new Info("cl_sampler").valueTypes("_cl_sampler").pointerTypes("@Cast(\"cl_sampler*\") PointerPointer", "@ByPtrPtr _cl_sampler"))
               .put(new Info("const cl_sampler").valueTypes("@Const _cl_sampler").pointerTypes("@Cast(\"const cl_sampler*\") PointerPointer", "@Cast(\"const cl_sampler*\") @ByPtrPtr _cl_sampler"))
               .put(new Info("cl_GLsync").valueTypes("__GLsync").pointerTypes("@Cast(\"cl_GLsync*\") PointerPointer", "@ByPtrPtr __GLsync"))
               .put(new Info("const cl_GLsync").valueTypes("@Const __GLsync").pointerTypes("@Cast(\"const cl_GLsync*\") PointerPointer", "@Cast(\"const cl_GLsync*\") @ByPtrPtr __GLsync"));
    }
}
