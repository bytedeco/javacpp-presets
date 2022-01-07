/*
 * Copyright (C) 2021 Jack He, Samuel Audet
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

package org.bytedeco.tritonserver.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.cuda.presets.cudart;
import org.bytedeco.cuda.presets.cublas;
import org.bytedeco.cuda.presets.cudnn;
import org.bytedeco.cuda.presets.nvrtc;
import org.bytedeco.tensorrt.presets.nvinfer;
import org.bytedeco.tensorrt.presets.nvinfer_plugin;
import org.bytedeco.tensorrt.presets.nvonnxparser;
import org.bytedeco.tensorrt.presets.nvparsers;

/**
 *
 * @author Jack He
 */
@Properties(
    inherit = {cublas.class, cudnn.class, nvrtc.class, nvinfer.class, nvinfer_plugin.class, nvonnxparser.class, nvparsers.class},
    value = {
        @Platform(
            value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "windows-x86_64"},
            include = {"tritonserver.h", "tritonbackend.h", "tritonrepoagent.h"},
            exclude = {"<cudaGL.h>", "<cuda_gl_interop.h>"},
            link = "tritonserver",
            includepath = {"/opt/tritonserver/include/triton/core/", "/opt/tritonserver/include/", "/usr/local/cuda/include/", "/usr/include"},
            linkpath = {"/usr/local/cuda/lib64/", "/opt/tritonserver/lib/"}
        ),
        @Platform(
            value = "windows-x86_64",
            includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TritonServer/include/triton/core/",
            linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TritonServer/lib/",
            preloadpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TritonServer/bin/"
        )
    },
    target = "org.bytedeco.tritonserver.tritonserver",
    global = "org.bytedeco.tritonserver.global.tritonserver"
)
public class tritonserver implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "tritonserver"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries()) {
            return;
        }
        int i = 0;
        String[] libs = {"cudart", "cublasLt", "cublas", "cudnn", "nvrtc",
                         "cudnn_ops_infer", "cudnn_ops_train", "cudnn_adv_infer",
                         "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train",
                         "nvinfer", "nvinfer_plugin", "nvonnxparser", "nvparsers"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.8" : lib.equals("cudart") ? "@.11.0" : lib.equals("nvrtc") ? "@.11.2" : "@.11";
                lib += lib.startsWith("nvinfer") ? "@.8" : lib.equals("nvonnxparser") ? "@.8" : lib.equals("nvparsers") ? "@.8" :"@.8";
            } else if (platform.startsWith("windows")) {
                lib += lib.startsWith("cudnn") ? "64_8" : lib.equals("cudart") ? "64_110" : lib.equals("nvrtc") ? "64_112_0" : "64_11";
                lib += lib.startsWith("nvinfer") ? "64_8" : lib.equals("nvonnxparser") ? "64_8" : lib.equals("nvparsers") ? "64_8" :"64_8";
            } else {
                continue; // no CUDA
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
            resources.add("/org/bytedeco/tensorrt/");
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.putFirst(new Info().enumerate(false))
               .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("boolean[]", "BoolPointer"))
               .put(new Info("TRITONSERVER_EXPORT", "TRITONSERVER_DECLSPEC",
                             "TRITONBACKEND_DECLSPEC", "TRITONBACKEND_ISPEC",
                             "TRITONREPOAGENT_DECLSPEC", "TRITONREPOAGENT_ISPEC").cppTypes().annotations())
        ;
    }
}
