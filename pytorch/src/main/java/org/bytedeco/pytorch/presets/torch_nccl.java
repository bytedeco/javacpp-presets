/*
 * Copyright (C) 2025 Samuel Audet
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

import org.bytedeco.cuda.presets.nccl;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;

/**
 * @author Samuel Audet
 */
@Properties(
    inherit = {nccl.class, torch_cuda.class},
    value = {
        // Java glue for NCCL is generated on all platforms (shared sources).
        // Native jnitorch_nccl is only produced on linux-gpu via library/link below.
        @Platform(
            value = {"linux", "macosx", "windows"},
            define = "USE_C10D_NCCL",
            include = {
                //"torch/csrc/distributed/c10d/cuda/CUDAEventCache.hpp",
                "torch/csrc/distributed/c10d/NCCLUtils.hpp",
                "torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp",
            }
        ),
        @Platform(
            value = "linux",
            extension = "-gpu",
            link = { "c10", "torch", "c10_cuda", "torch_cuda", "nccl" }
        )
    },
    target = "org.bytedeco.pytorch.distributed",
    global = "org.bytedeco.pytorch.global.torch_nccl"
)
public class torch_nccl implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        // Always parse NCCL headers so Java glue is generated on every OS.
        // Clear platform.library on non linux-gpu builds so javacpp does NOT
        // compile jnitorch_nccl.dylib/.so on macOS/Windows CPU.
        torch.initIncludes(getClass(), properties);
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        boolean nativeNccl = platform != null && platform.startsWith("linux")
                && extension != null && extension.endsWith("-gpu");
        if (!nativeNccl) {
            // Clearing platform.library alone is insufficient: generated NCCL peer
            // classes re-set it from the last global target. A dummy executable
            // makes Builder skip native library generation entirely.
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
            new torch.PointerInfo("c10d::NCCLComm"),
        }) {
            pi.makeShared(infoMap);
        }

        for (torch.PointerInfo pi : new torch.PointerInfo[]{
            new torch.PointerInfo("c10d::ProcessGroupNCCL"),
            new torch.PointerInfo("c10d::ProcessGroupNCCL::Options"),
            new torch.PointerInfo("c10d::ProcessGroupNCCL::WorkNCCL"),
        }) {
            pi.makeIntrusive(infoMap);
        }

        infoMap
            .put(new Info().javaText("import org.bytedeco.pytorch.Allocator;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.distributed.Backend;"))
            .put(new Info("(defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)").define(false))
            .put(new Info("std::map<at::ScalarType,ncclDataType_t>").pointerTypes("ScalaTypeDataTypeMap").define())
            .put(new Info("std::unordered_map<std::string,std::shared_ptr<c10d::NCCLComm> >").pointerTypes("StringNCCLCommMap").define())
        ;

        infoMap
            .put(new Info("c10d::ProcessGroupNCCL::registerOnCompletionHook").javaText(
                "public native void registerOnCompletionHook(\n" +
                "      WorkInfoConsumer hook);\n"))
            .put(new Info("c10d::ProcessGroupNCCL::Options::split_from").javaText(
                "public native @IntrusivePtr ProcessGroupNCCL split_from(); public native Options split_from(ProcessGroupNCCL setter);\n"))
            .put(new Info("c10d::ProcessGroupNCCL::HeartbeatMonitor::HeartbeatMonitor").javaText(
                "public HeartbeatMonitor(ProcessGroupNCCL pg) { super((Pointer)null); allocate(pg); }\n" +
                "private native void allocate(ProcessGroupNCCL pg);\n"))
            .put(new Info("c10d::ProcessGroupNCCL::Watchdog::Watchdog").javaText(
                "public Watchdog(ProcessGroupNCCL pg) { super((Pointer)null); allocate(pg); }\n" +
                "private native void allocate(ProcessGroupNCCL pg);\n"))
        ;

        infoMap
            .put(new Info(
                "std::enable_shared_from_this<CUDAEventCache>",
                "std::enable_shared_from_this<WorkNCCL>").cast().pointerTypes("Pointer"))
        ;

        //// No way to map
        infoMap
            .put(new Info("c10d::CUDAEventCache", "at::cuda::CUDAEvent",
                          "std::shared_ptr<at::cuda::CUDAEvent>").skip())
            .put(new Info("c10d::ProcessGroupNCCL::initIntraNodeComm",
                          "std::optional<std::function<std::string()> >",
                          "std::optional<std::function<void(std::function<void(const std::string&)>)> >").skip())
        ;
    }
}
