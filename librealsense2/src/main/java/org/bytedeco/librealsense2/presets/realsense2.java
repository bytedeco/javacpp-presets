/*
 * Copyright (C) 2019 Samuel Audet
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
package org.bytedeco.librealsense2.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    value = {
        @Platform(
            value = {"linux-x86", "macosx-x86", "windows-x86"},
            compiler = "cpp11",
            include = {
                "librealsense2/h/rs_types.h",
                "librealsense2/h/rs_context.h",
                "librealsense2/h/rs_device.h",
                "librealsense2/h/rs_frame.h",
                "librealsense2/h/rs_option.h",
                "librealsense2/h/rs_processing.h",
                "librealsense2/h/rs_record_playback.h",
                "librealsense2/h/rs_sensor.h",
                "librealsense2/h/rs_config.h",
                "librealsense2/h/rs_pipeline.h",
                "librealsense2/h/rs_advanced_mode_command.h",
                "librealsense2/rs.h",
                "librealsense2/rs_advanced_mode.h",
                "librealsense2/rsutil.h"
            },
            link = "realsense2@.2.25"
        ),
        @Platform(value = "macosx", preload = "usb-1.0@.0", preloadpath = "/usr/local/lib/"),
        @Platform(value = "windows", preload = {
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
            "vcruntime140", "msvcp140", "concrt140"}),
        @Platform(value = "windows-x86", preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x86/Microsoft.VC140.CRT/",
                                                        "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x86/"}),
        @Platform(value = "windows-x86_64", preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x64/Microsoft.VC140.CRT/",
                                                           "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x64/"})
    },
    target = "org.bytedeco.librealsense2",
    global = "org.bytedeco.librealsense2.global.realsense2"
)
public class realsense2 implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "librealsense2"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> preloadpaths = properties.get("platform.preloadpath");

        String vcredistdir = System.getenv("VCToolsRedistDir");
        if (vcredistdir != null && vcredistdir.length() > 0) {
            switch (platform) {
                case "windows-x86":
                    preloadpaths.add(0, vcredistdir + "\\x86\\Microsoft.VC141.CRT");
                    break;
                case "windows-x86_64":
                    preloadpaths.add(0, vcredistdir + "\\x64\\Microsoft.VC141.CRT");
                    break;
                default:
                    // not Windows
            }
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("rs2_camera_info", "rs2_stream").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("RS2_API_VERSION_STR").cppTypes("const char*").pointerTypes("String").translate(false))
               .put(new Info("rs2_create_playback_device").skip());
    }

}
