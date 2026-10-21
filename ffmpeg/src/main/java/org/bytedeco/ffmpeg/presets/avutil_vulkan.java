/*
 * Copyright (C) 2026 Jorge Raul Arroyo Perez
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

package org.bytedeco.ffmpeg.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Bindings for the Vulkan hardware context of libavutil, {@code <libavutil/hwcontext_vulkan.h>}:
 * {@code AVVulkanDeviceContext}, {@code AVVulkanFramesContext}, {@code AVVulkanDeviceQueueFamily} and
 * {@code AVVkFrame}, plus {@code av_vk_frame_alloc()}, {@code av_vkfmt_from_pixfmt()} and the
 * {@code av_vk_get_optional_*_extensions()} functions. With these, an application can hand its own
 * {@code VkDevice} to FFmpeg through {@code AVHWDeviceContext.hwctx} and read the {@code VkImage},
 * {@code VkDeviceMemory} and timeline {@code VkSemaphore} of each decoded frame from
 * {@code AVFrame.data[0]}, without any copy through host memory.
 * <p>
 * The structs are generated in the {@code org.bytedeco.ffmpeg.avutil} package next to the other
 * libavutil types, the functions and the {@code AVVkFrameFlags} enum in
 * {@link org.bytedeco.ffmpeg.global.avutil_vulkan}, and everything is compiled into the same
 * {@code jniavutil} library as the rest of libavutil. Vulkan handles ({@code VkInstance},
 * {@code VkPhysicalDevice}, {@code VkDevice}, {@code VkImage}, {@code VkDeviceMemory},
 * {@code VkSemaphore}) and {@code PFN_vkGetInstanceProcAddr} are mapped to {@code long}, Vulkan
 * enums and flag bits to {@code int}, which is also how LWJGL represents them, so values can be
 * passed back and forth as is. {@code VkPhysicalDeviceFeatures2} and {@code VkAllocationCallbacks}
 * are exposed as plain {@link org.bytedeco.javacpp.Pointer} addresses.
 * <p>
 * FFmpeg is not built with Vulkan support on Android, so this API is not available on that platform.
 *
 * @author Jorge Raul Arroyo Perez
 */
@Properties(
    inherit = avutil.class,
    target = "org.bytedeco.ffmpeg.avutil",
    global = "org.bytedeco.ffmpeg.global.avutil_vulkan",
    value = {
        @Platform(not = "android", cinclude = "<libavutil/hwcontext_vulkan.h>", library = "jniavutil")
    }
)
public class avutil_vulkan implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "ffmpeg"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("VkInstance", "VkPhysicalDevice", "VkDevice", "VkImage", "VkDeviceMemory", "VkSemaphore",
                             "PFN_vkGetInstanceProcAddr").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("VkFormat", "VkImageTiling", "VkImageLayout", "VkImageCreateFlags", "VkImageUsageFlagBits",
                             "VkMemoryPropertyFlagBits", "VkAccessFlagBits", "VkQueueFlagBits", "VkVideoCodecOperationFlagBitsKHR")
                             .cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("VkAllocationCallbacks", "VkPhysicalDeviceFeatures2", "AVVkFrameInternal").cast().pointerTypes("Pointer"))
               // "const char * const *" members get no setter by default, but applications supplying their own
               // VkDevice have to tell FFmpeg which instance and device extensions were enabled on it
               .put(new Info("AVVulkanDeviceContext::enabled_inst_extensions").javaText(
                             "@MemberGetter public native @Cast(\"const char*\") BytePointer enabled_inst_extensions(int i);\n"
                           + "public native @Cast(\"const char*const*\") PointerPointer enabled_inst_extensions(); "
                           + "public native AVVulkanDeviceContext enabled_inst_extensions(@Cast(\"const char*const*\") PointerPointer setter);"))
               .put(new Info("AVVulkanDeviceContext::enabled_dev_extensions").javaText(
                             "@MemberGetter public native @Cast(\"const char*\") BytePointer enabled_dev_extensions(int i);\n"
                           + "public native @Cast(\"const char*const*\") PointerPointer enabled_dev_extensions(); "
                           + "public native AVVulkanDeviceContext enabled_dev_extensions(@Cast(\"const char*const*\") PointerPointer setter);"));
    }
}
