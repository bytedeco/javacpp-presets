/*
 * Copyright (C) 2019-2020 Samuel Audet
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
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            value = {"linux-armhf", "linux-arm64", "linux-x86", "macosx-x86", "windows-x86"},
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
            link = "realsense2@.2.50"
        ),
        @Platform(value = "macosx", preload = "usb-1.0@.0", preloadpath = "/usr/local/lib/")
    },
    target = "org.bytedeco.librealsense2",
    global = "org.bytedeco.librealsense2.global.realsense2"
)
public class realsense2 implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "librealsense2"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("rs2_camera_info", "rs2_stream").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("RS2_API_VERSION_STR").cppTypes("const char*").pointerTypes("String").translate(false))
               .put(new Info("RS2_API_FULL_VERSION_STR").cppTypes("const char*").pointerTypes("String").translate(false))
               .put(new Info("rs2_get_frame_data").javaText(
                       "public static native @Const Pointer rs2_get_frame_data(@Const rs2_frame frame, @Cast(\"rs2_error**\") PointerPointer error);\n"
                     + "public static native @Const Pointer rs2_get_frame_data(@Const rs2_frame frame, @ByPtrPtr rs2_error error);\n"
                     + "public static native @Cast(\"const void*\") @Name(\"rs2_get_frame_data\") long rs2_get_frame_data_address(@Const rs2_frame frame, @Cast(\"rs2_error**\") PointerPointer error);\n"
                     + "public static native @Cast(\"const void*\") @Name(\"rs2_get_frame_data\") long rs2_get_frame_data_address(@Const rs2_frame frame, @ByPtrPtr rs2_error error);\n"))
               .put(new Info("rs2_create_playback_device", "rs2_cah_trigger_to_string", "rs2_ambient_light_to_string", "rs2_digital_gain_to_string",
                             "rs2_create_y411_decoder").skip());
    }
}
