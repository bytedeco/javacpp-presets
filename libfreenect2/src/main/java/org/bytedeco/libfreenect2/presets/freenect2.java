/*
 * Copyright (C) 2016 Jérémy Laviole
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
package org.bytedeco.libfreenect2.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Jeremy Laviole
 */
@Properties(target = "org.bytedeco.libfreenect2.freenect2", global = "org.bytedeco.libfreenect2.global.freenect2", value = {
    @Platform(value = {"linux-x86", "macosx-x86_64", "windows-x86_64"}, include = {"<libfreenect2/libfreenect2.hpp>",
                "<libfreenect2/frame_listener.hpp>", "<libfreenect2/frame_listener_impl.h>", "<libfreenect2/logger.h>",
                "<libfreenect2/packet_pipeline.h>", "<libfreenect2/registration.h>", "<libfreenect2/config.h>"},
            link = "freenect2@.0.2"),
    @Platform(value = "macosx-x86_64", preload = "usb-1.0@.0", preloadpath = "/usr/local/lib/"),
    @Platform(value = "windows-x86_64", preload = {
                "api-ms-win-crt-locale-l1-1-0", "api-ms-win-crt-string-l1-1-0", "api-ms-win-crt-stdio-l1-1-0", "api-ms-win-crt-math-l1-1-0",
                "api-ms-win-crt-heap-l1-1-0", "api-ms-win-crt-runtime-l1-1-0", "api-ms-win-crt-convert-l1-1-0", "api-ms-win-crt-environment-l1-1-0",
                "api-ms-win-crt-time-l1-1-0", "api-ms-win-crt-filesystem-l1-1-0", "api-ms-win-crt-utility-l1-1-0", "api-ms-win-crt-multibyte-l1-1-0",
                "api-ms-win-core-string-l1-1-0", "api-ms-win-core-errorhandling-l1-1-0", "api-ms-win-core-timezone-l1-1-0", "api-ms-win-core-file-l1-1-0",
                "api-ms-win-core-namedpipe-l1-1-0", "api-ms-win-core-handle-l1-1-0", "api-ms-win-core-file-l2-1-0", "api-ms-win-core-heap-l1-1-0",
                "api-ms-win-core-libraryloader-l1-1-0", "api-ms-win-core-synch-l1-1-0", "api-ms-win-core-processthreads-l1-1-0",
                "api-ms-win-core-processenvironment-l1-1-0", "api-ms-win-core-datetime-l1-1-0", "api-ms-win-core-localization-l1-2-0",
                "api-ms-win-core-sysinfo-l1-1-0", "api-ms-win-core-synch-l1-2-0", "api-ms-win-core-console-l1-1-0", "api-ms-win-core-debug-l1-1-0",
                "api-ms-win-core-rtlsupport-l1-1-0", "api-ms-win-core-processthreads-l1-1-1", "api-ms-win-core-file-l1-2-0", "api-ms-win-core-profile-l1-1-0",
                "api-ms-win-core-memory-l1-1-0", "api-ms-win-core-util-l1-1-0", "api-ms-win-core-interlocked-l1-1-0", "ucrtbase",
                "vcruntime140", "msvcp140", "concrt140", "libusb-1.0", "glfw3", "turbojpeg", "freenect2-openni2"},
            preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x64/Microsoft.VC140.CRT/",
                           "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x64/"}) })
public class freenect2 implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("LIBFREENECT2_WITH_CUDA_SUPPORT", "LIBFREENECT2_WITH_OPENCL_SUPPORT").define(false))
               .put(new Info("libfreenect2::Frame::Type").valueTypes("@Cast(\"libfreenect2::Frame::Type\") int"))
               .put(new Info("std::map<libfreenect2::Frame::Type,libfreenect2::Frame*>").pointerTypes("FrameMap").define())
               .put(new Info("LIBFREENECT2_API").skip());
    }
}
