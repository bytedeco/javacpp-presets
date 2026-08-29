/*
 * Copyright (C) 2026 Barry Pitman
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

package org.bytedeco.openvino.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.opencl.presets.OpenCL;

/**
 *
 * @author Barry Pitman
 */
@Properties(
    inherit = OpenCL.class,
    value = {
        @Platform(
            value = {"linux-x86_64"},
            include = {
                "openvino/c/openvino.h",
                "openvino/c/auto/properties.h",
                "openvino/c/ov_common.h",
                "openvino/c/ov_compiled_model.h",
                "openvino/c/ov_core.h",
                "openvino/c/ov_dimension.h",
                "openvino/c/ov_infer_request.h",
                "openvino/c/ov_layout.h",
                "openvino/c/ov_model.h",
                "openvino/c/ov_node.h",
                "openvino/c/ov_partial_shape.h",
                "openvino/c/ov_prepostprocess.h",
                "openvino/c/ov_property.h",
                "openvino/c/ov_rank.h",
                "openvino/c/ov_remote_context.h",
                "openvino/c/ov_shape.h",
                "openvino/c/ov_tensor.h",
                "openvino/c/ov_util.h",
                "openvino/c/gpu/gpu_plugin_properties.h"
            },
            link = {"openvino_c@.2621#", "openvino@.2621#"},
            preloadresource = {"runtime/lib/intel64/"},
            preload = {
                "tbb:runtime/3rdparty/tbb/lib/libtbb.so.12",
                "openvino:runtime/lib/intel64/libopenvino.so.2621",
                "openvino_c:runtime/lib/intel64/libopenvino_c.so.2621",
                "openvino_auto_batch_plugin:runtime/lib/intel64/libopenvino_auto_batch_plugin.so",
                "openvino_auto_plugin:runtime/lib/intel64/libopenvino_auto_plugin.so",
                "openvino_hetero_plugin:runtime/lib/intel64/libopenvino_hetero_plugin.so",
                "openvino_intel_cpu_plugin:runtime/lib/intel64/libopenvino_intel_cpu_plugin.so",
                "openvino_intel_gpu_plugin:runtime/lib/intel64/libopenvino_intel_gpu_plugin.so",
                "openvino_intel_npu_plugin:runtime/lib/intel64/libopenvino_intel_npu_plugin.so",
                "openvino_ir_frontend:runtime/lib/intel64/libopenvino_ir_frontend.so.2621",
                "openvino_onnx_frontend:runtime/lib/intel64/libopenvino_onnx_frontend.so.2621",
                "openvino_paddle_frontend:runtime/lib/intel64/libopenvino_paddle_frontend.so.2621",
                "openvino_pytorch_frontend:runtime/lib/intel64/libopenvino_pytorch_frontend.so.2621",
                "openvino_tensorflow_frontend:runtime/lib/intel64/libopenvino_tensorflow_frontend.so.2621",
                "openvino_tensorflow_lite_frontend:runtime/lib/intel64/libopenvino_tensorflow_lite_frontend.so.2621"
            },
            resource = {"runtime"}
        ),
        @Platform(
            value = {"macosx-arm64"},
            include = {
                "openvino/c/openvino.h",
                "openvino/c/auto/properties.h",
                "openvino/c/ov_common.h",
                "openvino/c/ov_compiled_model.h",
                "openvino/c/ov_core.h",
                "openvino/c/ov_dimension.h",
                "openvino/c/ov_infer_request.h",
                "openvino/c/ov_layout.h",
                "openvino/c/ov_model.h",
                "openvino/c/ov_node.h",
                "openvino/c/ov_partial_shape.h",
                "openvino/c/ov_prepostprocess.h",
                "openvino/c/ov_property.h",
                "openvino/c/ov_rank.h",
                "openvino/c/ov_remote_context.h",
                "openvino/c/ov_shape.h",
                "openvino/c/ov_tensor.h",
                "openvino/c/ov_util.h",
                "openvino/c/gpu/gpu_plugin_properties.h"
            },
            link = {"openvino_c@.2610", "openvino@.2610"},
            preloadresource = {
                "runtime/lib/arm64/Release/",
                "runtime/3rdparty/tbb/lib/"
            },
            preload = {
                "openvino_auto_batch_plugin",
                "openvino_auto_plugin",
                "openvino_hetero_plugin",
                "openvino_arm_cpu_plugin",
                "openvino_ir_frontend@.2610",
                "openvino_onnx_frontend@.2610",
                "openvino_paddle_frontend@.2610",
                "openvino_pytorch_frontend@.2610",
                "openvino_tensorflow_frontend@.2610",
                "openvino_tensorflow_lite_frontend@.2610",
                "tbb@.12"
            },
            resource = {"runtime"}
        ),
        @Platform(
            value = {"windows-x86_64"},
            include = {
                "openvino/c/openvino.h",
                "openvino/c/auto/properties.h",
                "openvino/c/ov_common.h",
                "openvino/c/ov_compiled_model.h",
                "openvino/c/ov_core.h",
                "openvino/c/ov_dimension.h",
                "openvino/c/ov_infer_request.h",
                "openvino/c/ov_layout.h",
                "openvino/c/ov_model.h",
                "openvino/c/ov_node.h",
                "openvino/c/ov_partial_shape.h",
                "openvino/c/ov_prepostprocess.h",
                "openvino/c/ov_property.h",
                "openvino/c/ov_rank.h",
                "openvino/c/ov_remote_context.h",
                "openvino/c/ov_shape.h",
                "openvino/c/ov_tensor.h",
                "openvino/c/ov_util.h",
                "openvino/c/gpu/gpu_plugin_properties.h"
            },
            link = {"openvino_c", "openvino"},
            preloadresource = {
                "runtime/bin/intel64/Release/",
                "runtime/3rdparty/tbb/bin/"
            },
            preload = {
                "tbb12",
                "openvino",
                "openvino_c",
                "openvino_auto_batch_plugin",
                "openvino_auto_plugin",
                "openvino_hetero_plugin",
                "openvino_intel_cpu_plugin",
                "openvino_intel_gpu_plugin",
                "openvino_intel_npu_plugin",
                "openvino_ir_frontend",
                "openvino_onnx_frontend",
                "openvino_paddle_frontend",
                "openvino_pytorch_frontend",
                "openvino_tensorflow_frontend",
                "openvino_tensorflow_lite_frontend"
            },
            resource = {"runtime"}
        ),
    },
    target = "org.bytedeco.openvino",
    global = "org.bytedeco.openvino.global.openvino"
)
public class openvino implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "openvino"); }

    @Override public void map(InfoMap infoMap) {
        infoMap.put(new org.bytedeco.javacpp.tools.Info("extern", "__cdecl").cppTypes().annotations())
               // JavaCPP's lightweight preprocessor does not evaluate OpenVINO's compiler/platform tests.
               // Expand the public declaration macros directly, while the generated native code retains the
               // original header definitions (including __cdecl on Windows).
               .put(new org.bytedeco.javacpp.tools.Info("OPENVINO_C_API").cppText("#define OPENVINO_C_API(...) __VA_ARGS__"))
               .put(new org.bytedeco.javacpp.tools.Info("OPENVINO_C_VAR").cppText("#define OPENVINO_C_VAR(...) __VA_ARGS__"))
               .put(new org.bytedeco.javacpp.tools.Info("OPENVINO_C_API_EXTERN", "OPENVINO_C_API_CALLBACK").skip())
               // JavaCPP cannot expand C varargs. These overloads cover the property-pair form used by
               // OpenVINO's C API while retaining the generated zero-property overloads.
               .put(new org.bytedeco.javacpp.tools.Info("ov_core_compile_model").javaText(
                       "public static native @Cast(\"ov_status_e\") int ov_core_compile_model(@Const ov_core_t core, @Const ov_model_t model, String device_name, @Cast(\"const size_t\") long property_args_size, @ByPtrPtr ov_compiled_model_t compiled_model);\n"
                     + "public static native @Cast(\"ov_status_e\") int ov_core_compile_model(@Const ov_core_t core, @Const ov_model_t model, String device_name, @Cast(\"const size_t\") long property_args_size, @ByPtrPtr ov_compiled_model_t compiled_model, String property_key, String property_value);\n"))
               .put(new org.bytedeco.javacpp.tools.Info("ov_core_create_context").javaText(
                       "public static native @Cast(\"ov_status_e\") int ov_core_create_context(@Const ov_core_t core, String device_name, @Cast(\"const size_t\") long context_args_size, @ByPtrPtr ov_remote_context_t context);\n"
                     + "public static native @Cast(\"ov_status_e\") int ov_core_create_context(@Const ov_core_t core, String device_name, @Cast(\"const size_t\") long context_args_size, @ByPtrPtr ov_remote_context_t context, String property_key, Pointer property_value);\n"
                     + "public static native @Cast(\"ov_status_e\") int ov_core_create_context(@Const ov_core_t core, String device_name, @Cast(\"const size_t\") long context_args_size, @ByPtrPtr ov_remote_context_t context, String property_key1, String property_value1, String property_key2, Pointer property_value2, String property_key3, Pointer property_value3);\n"))
               .put(new org.bytedeco.javacpp.tools.Info("ov_core_compile_model_with_context").javaText(
                       "public static native @Cast(\"ov_status_e\") int ov_core_compile_model_with_context(@Const ov_core_t core, @Const ov_model_t model, @Const ov_remote_context_t context, @Cast(\"const size_t\") long property_args_size, @ByPtrPtr ov_compiled_model_t compiled_model);\n"
                     + "public static native @Cast(\"ov_status_e\") int ov_core_compile_model_with_context(@Const ov_core_t core, @Const ov_model_t model, @Const ov_remote_context_t context, @Cast(\"const size_t\") long property_args_size, @ByPtrPtr ov_compiled_model_t compiled_model, String property_key, String property_value);\n"))
               .put(new org.bytedeco.javacpp.tools.Info("ov_remote_context_create_tensor").javaText(
                       "public static native @Cast(\"ov_status_e\") int ov_remote_context_create_tensor(@Const ov_remote_context_t context, @Cast(\"const ov_element_type_e\") int type, @Const @ByVal ov_shape_t shape, @Cast(\"const size_t\") long object_args_size, @ByPtrPtr ov_tensor_t remote_tensor);\n"
                     + "public static native @Cast(\"ov_status_e\") int ov_remote_context_create_tensor(@Const ov_remote_context_t context, @Cast(\"const ov_element_type_e\") int type, @Const @ByVal ov_shape_t shape, @Cast(\"const size_t\") long object_args_size, @ByPtrPtr ov_tensor_t remote_tensor, String property_key, Pointer property_value);\n"
                     + "public static native @Cast(\"ov_status_e\") int ov_remote_context_create_tensor(@Const ov_remote_context_t context, @Cast(\"const ov_element_type_e\") int type, @Const @ByVal ov_shape_t shape, @Cast(\"const size_t\") long object_args_size, @ByPtrPtr ov_tensor_t remote_tensor, String property_key1, String property_value1, String property_key2, Pointer property_value2);\n"))
               .put(new org.bytedeco.javacpp.tools.Info("OV_BOOLEAN", "BOOLEAN").skip())
               .put(new org.bytedeco.javacpp.tools.Info("ov_dimension_t", "ov_rank_t").skip());
    }
}
