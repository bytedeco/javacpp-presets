/*
 * Copyright (C) 2022 Andrey Krainyak
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

package org.bytedeco.bullet.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Andrey Krainyak
 */
@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            include = {
                "clew/clew.h",
                "clew_stubs.h",
            },
            link = "Bullet3OpenCL_clew@.3.20"
        )
    },
    target = "org.bytedeco.bullet.clew"
)
public class clew implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("cl_bool").cppTypes("bool"))
            .put(new Info("cl_int").cppTypes("int"))
            .put(new Info("cl_long").cppTypes("long"))
            .put(new Info("cl_uint").cppTypes("unsigned int"))
            .put(new Info("cl_ulong").cppTypes("unsigned long"))

            .put(new Info(
                    "CL_HUGE_VAL",
                    "CL_PROGRAM_STRING_DEBUG_INFO"
                ).cppTypes().translate(false))

            .put(new Info(
                    "(defined(_WIN32) && defined(_MSC_VER))",
                    "__APPLE1__",
                    "defined(CL_NAMED_STRUCT_SUPPORTED) && defined(_MSC_VER)",
                    "defined(CL_NAMED_STRUCT_SUPPORTED)",
                    "defined(_WIN32)",
                    "defined(__AVX__)",
                    "defined(__GNUC__)",
                    "defined(__MMX__)",
                    "defined(__SSE2__)",
                    "defined(__SSE__)",
                    "defined(__VEC__)",
                    "defined(__cl_uchar2__)"
                ).define(false))

            .put(new Info("cl_command_queue").pointerTypes("cl_command_queue"))
            .put(new Info("cl_context").pointerTypes("cl_context"))
            .put(new Info("cl_device_id").pointerTypes("cl_device_id"))
            .put(new Info("cl_event").pointerTypes("cl_event"))
            .put(new Info("cl_kernel").pointerTypes("cl_kernel"))
            .put(new Info("cl_mem").pointerTypes("cl_mem"))
            .put(new Info("cl_platform_id").pointerTypes("cl_platform_id"))
            .put(new Info("cl_program").pointerTypes("cl_program"))
            .put(new Info("cl_sampler").pointerTypes("cl_sampler"))

            .put(new Info("clew.h").linePatterns(
                    ".*typedef.*CL_API_ENTRY.*",
                    "#define clGetExtensionFunctionAddress.*"
                ).skip())

            .put(new Info(
                    "CL_NAN",
                    "cl_long2",
                    "cl_long4",
                    "cl_long8",
                    "cl_long16",
                    "nanf"
                ).skip())
            ;

        String[] types = new String[] {
            "CHAR",
            "UCHAR",
            "SHORT",
            "USHORT",
            "INT",
            "UINT",
            "LONG",
            "ULONG",
            "FLOAT",
            "DOUBLE",
        };

        for (String type: types) {
            for (String size: new String[] { "2", "4", "8", "16" }) {
                infoMap.put(new Info(
                    "defined(__CL_" + type + size + "__)").define(false));
            }
        }
    }

	public static class cl_command_queue extends Pointer {
	    public cl_command_queue() { super((Pointer)null); }
	    public cl_command_queue(Pointer p) { super(p); }
	}

	public static class cl_context extends Pointer {
	    public cl_context() { super((Pointer)null); }
	    public cl_context(Pointer p) { super(p); }
	}

	public static class cl_device_id extends Pointer {
	    public cl_device_id() { super((Pointer)null); }
	    public cl_device_id(Pointer p) { super(p); }
	}

	public static class cl_event extends Pointer {
	    public cl_event() { super((Pointer)null); }
	    public cl_event(Pointer p) { super(p); }
	}

	public static class cl_kernel extends Pointer {
	    public cl_kernel() { super((Pointer)null); }
	    public cl_kernel(Pointer p) { super(p); }
	}

	public static class cl_mem extends Pointer {
	    public cl_mem() { super((Pointer)null); }
	    public cl_mem(Pointer p) { super(p); }
	}

	public static class cl_platform_id extends Pointer {
	    public cl_platform_id() { super((Pointer)null); }
	    public cl_platform_id(Pointer p) { super(p); }
	}

	public static class cl_program extends Pointer {
	    public cl_program() { super((Pointer)null); }
	    public cl_program(Pointer p) { super(p); }
	}

	public static class cl_sampler extends Pointer {
	    public cl_sampler() { super((Pointer)null); }
	    public cl_sampler(Pointer p) { super(p); }
	}
}
