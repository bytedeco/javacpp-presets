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
                "openvino/c/openvino.h"
            },
            link = {"openvino_c@.2610", "openvino@.2610"},
            preloadresource = {
                "runtime/lib/intel64/",
                "runtime/3rdparty/tbb/lib/"
            },
            preload = {
                "openvino_auto_batch_plugin",
                "openvino_auto_plugin",
                "openvino_hetero_plugin",
                "openvino_intel_cpu_plugin",
                "openvino_intel_gpu_plugin",
                "openvino_intel_npu_plugin",
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
            value = {"macosx-arm64"},
            include = {
                "openvino/c/openvino.h"
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
                "openvino/c/openvino.h"
            },
            link = {"openvino_c", "openvino"},
            preloadresource = {
                "runtime/bin/intel64/Release/",
                "runtime/3rdparty/tbb/bin/"
            },
            preload = {
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
                "openvino_tensorflow_lite_frontend",
                "tbb12"
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
               .put(new org.bytedeco.javacpp.tools.Info("OPENVINO_C_API_EXTERN", "OPENVINO_C_API_CALLBACK").skip())
               .put(new org.bytedeco.javacpp.tools.Info("OV_BOOLEAN", "BOOLEAN").skip())
               .put(new org.bytedeco.javacpp.tools.Info("ov_rank_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new org.bytedeco.javacpp.tools.Info("ov_dimension_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"));
    }
}
