/*
 * Copyright (C) 2024-2025 Hervé Guillemet, Samuel Audet
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

package org.bytedeco.cuda.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Hervé Guillemet
 */
@Properties(inherit = cudart.class, value = {
    @Platform(include = {"cupti_result.h", "cupti_version.h", "cupti_activity.h", "cupti_callbacks.h", "cupti_events.h", "cupti_metrics.h", "cupti_driver_cbid.h", "cupti_runtime_cbid.h", "cupti_nvtx_cbid.h"},
              link = "cupti@.13"),
    @Platform(value = "windows-x86_64", includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/extras/CUPTI/include/", linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/extras/CUPTI/lib64/",
              preload = "cupti64_2025.4.1", preloadpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib64/"),
    @Platform(value = {"linux-x86_64", "linux-arm64", "linux-ppc64le"}, includepath = {"/usr/local/cuda-13.1/extras/CUPTI/include/", "/usr/local/cuda/extras/CUPTI/include/"}, linkpath = {"/usr/local/cuda-13.1/extras/CUPTI/lib64/", "/usr/local/cuda/extras/CUPTI/lib64/"}),
    @Platform(value = "macosx-x86_64", includepath = "/Developer/NVIDIA/CUDA-13.1/extras/CUPTI/include/", linkpath = "/Developer/NVIDIA/CUDA-13.1/extras/CUPTI/lib64/"),
},
    target = "org.bytedeco.cuda.cupti", global = "org.bytedeco.cuda.global.cupti")
@NoException
public class cupti implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("CUPTIAPI").cppTypes().annotations().cppText(""))
            .put(new Info("CUPTILP64").define())
            .put(new Info("CUpti_ActivityConfidentialComputeRotation", "cuptiActivityEnableAllocationSource").skip())
            .put(new Info("CUpti_EventID", "CUpti_EventDomainID", "CUpti_MetricID", "CUpti_CallbackId", "CUpti_DeviceAttribute", "CUpti_MetricValueKind").valueTypes("int").cast().pointerTypes("IntPointer", "int[]")) // enum or uint32
            .put(new Info("CUpti_SubscriberHandle").valueTypes("@ByPtr CUpti_Subscriber_st").pointerTypes("@ByPtrPtr CUpti_Subscriber_st"))
        ;
    }
}
