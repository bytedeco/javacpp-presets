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
    value = @Platform(
        value = {"linux"}, //// Not on Mac or Windows
        extension = "-gpu",
        define = "USE_C10D_NCCL",
        include = {
            //"torch/csrc/distributed/c10d/cuda/CUDAEventCache.hpp",
            "torch/csrc/distributed/c10d/NCCLUtils.hpp",
            "torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp",
        }
    ),
    target = "org.bytedeco.pytorch.nccl",
    global = "org.bytedeco.pytorch.global.torch_nccl"
)
public class torch_nccl implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        torch.initIncludes(getClass(), properties);
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
            .put(new Info().javaText("import org.bytedeco.pytorch.Backend;"))
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
            .put(new Info("c10d::ProcessGroupNCCL::initIntraNodeComm",
                          "std::optional<std::function<std::string()> >",
                          "std::optional<std::function<void(std::function<void(const std::string&)>)> >").skip())
        ;
    }
}
