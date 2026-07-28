/*
 * Copyright (C) 2025-2026 Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.presets;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * JavaCPP preset for {@code c10d::ProcessGroupUCC}.
 *
 * <p>UCC (Unified Collective Communication) is primarily a Linux feature;
 * UCX/UCC do not build cleanly on macOS (no librt / CPU affinity APIs).
 * This preset always parses headers (with {@code USE_C10D_UCC}) so Java glue
 * is generated everywhere. Native {@code jnitorch_ucc} is only linked on Linux
 * when libtorch was built with {@code USE_UCC=1} and UCC is installed.
 *
 * <p>For header-only parse on Mac, point include path at a UCC source checkout
 * (e.g. {@code cppbuild/deps/install/include}) via {@code UCC_HOME}.
 */
@Properties(
    inherit = torch.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            define = {"USE_C10D_UCC"},
            include = {
                "torch/csrc/distributed/c10d/UCCUtils.hpp",
                "torch/csrc/distributed/c10d/ProcessGroupUCC.hpp",
            }
        ),
        @Platform(
            value = "linux",
            link = { "c10", "torch", "torch_cpu", "ucc", "ucp", "ucs" }
        )
    },
    target = "org.bytedeco.pytorch.distributed",
    global = "org.bytedeco.pytorch.global.torch_ucc"
)
public class torch_ucc implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        torch.initIncludes(getClass(), properties);

        // UCC public headers (parse-only on Mac; full install on Linux).
        String[] uccIncCandidates = {
            System.getenv("UCC_HOME") != null ? System.getenv("UCC_HOME") + "/include" : null,
            System.getProperty("user.dir") + "/cppbuild/deps/install/include",
            "/Users/muller/Documents/code/rust/javacpp-presets/pytorch/cppbuild/deps/install/include",
            "/usr/local/include",
            "/opt/ucc/include",
        };
        String[] uccLibCandidates = {
            System.getenv("UCC_HOME") != null ? System.getenv("UCC_HOME") + "/lib" : null,
            System.getProperty("user.dir") + "/cppbuild/deps/install/lib",
            "/usr/local/lib",
            "/opt/ucc/lib",
        };
        for (String p : uccIncCandidates) {
            if (p != null && new java.io.File(p, "ucc/api/ucc.h").isFile()) {
                properties.addAll("platform.includepath", p);
                break;
            }
        }
        for (String p : uccLibCandidates) {
            if (p != null && new java.io.File(p).isDirectory()) {
                properties.addAll("platform.linkpath", p);
            }
        }

        // Native jnitorch_ucc only on Linux with explicit opt-in (needs rebuilt libtorch).
        String platform = properties.getProperty("platform");
        boolean nativeUcc = platform != null && platform.startsWith("linux")
                && ("1".equals(System.getenv("JAVACPP_ENABLE_UCC_NATIVE"))
                    || "1".equals(System.getenv("USE_UCC")));
        if (!nativeUcc) {
            properties.setProperty("platform.library", "");
            properties.put("platform.link", new java.util.ArrayList<String>());
            java.util.ArrayList<String> skipNative = new java.util.ArrayList<String>();
            skipNative.add("__skip_native_library__");
            properties.put("platform.executable", skipNative);
        }
    }

    @Override
    public void map(InfoMap infoMap) {
        for (torch.PointerInfo pi : new torch.PointerInfo[]{
            new torch.PointerInfo("c10d::ProcessGroupUCC"),
            new torch.PointerInfo("c10d::ProcessGroupUCCLogger"),
        }) {
            pi.makeIntrusive(infoMap);
        }

        infoMap
            .put(new Info().javaText("import org.bytedeco.pytorch.distributed.Backend;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.distributed.Work;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.distributed.Store;"))
            // UCC C API opaque handles
            .put(new Info("ucc_coll_args_t", "ucc_coll_req_h", "ucc_team_h", "ucc_ee_h",
                          "ucc_lib_h", "ucc_context_h", "ucc_status_t",
                          "ucc_datatype_t", "ucc_memory_type_t",
                          "ucc_coll_type_t", "ucc_reduction_op_t").cast().valueTypes("long").pointerTypes("LongPointer", "Pointer"))
            // Internal helpers / CUDA-only pieces
            .put(new Info("c10d::Comm", "c10d::CommBase", "c10d::CommUCC",
                          "c10d::WorkData", "c10d::AlltoallWorkData",
                          "c10d::ProgressEntry", "c10d::event_pool_t",
                          "c10d::torch_ucc_oob_coll_info_t",
                          "c10d::torch_ucc_phase_t",
                          "c10d::ucc_phase_map",
                          "std::map<c10d::torch_ucc_phase_t,std::string>",
                          "c10d::ProcessGroupUCC::WorkData",
                          "c10d::ProcessGroupUCC::AlltoallWorkData",
                          "c10d::ProcessGroupUCC::ProgressEntry",
                          "c10d::ProcessGroupUCC::set_timeout").skip())
            // CUDA event / stream bits when headers see USE_CUDA
            .put(new Info("at::cuda::CUDAEvent", "at::cuda::CUDAStream",
                          "std::unique_ptr<at::cuda::CUDAEvent>",
                          "std::unique_ptr<at::cuda::CUDAStream>",
                          "std::queue<std::unique_ptr<at::cuda::CUDAEvent> >").skip())
        ;
    }
}
