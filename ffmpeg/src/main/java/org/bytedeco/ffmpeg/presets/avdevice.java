/*
 * Copyright (C) 2013-2023 Samuel Audet
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
    inherit = avfilter.class,
    target = "org.bytedeco.ffmpeg.avdevice",
    global = "org.bytedeco.ffmpeg.global.avdevice",
    value = {
        @Platform(cinclude = {"<libavdevice/avdevice.h>", "<libavdevice/version_major.h>", "<libavdevice/version.h>"}, link = "avdevice@.60"),
        @Platform(value = "windows", preload = "avdevice-60")
    }
)
public class avdevice implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.putFirst(new Info("AVDeviceInfoList").pointerTypes("AVDeviceInfoList"))
               .putFirst(new Info("AVDeviceCapabilitiesQuery").pointerTypes("AVDeviceCapabilitiesQuery"))
               .put(new Info("LIBAVDEVICE_VERSION").cppTypes())
               .put(new Info("LIBAVDEVICE_VERSION_INT", "LIBAVDEVICE_IDENT").translate(false))
               .put(new Info("FF_API_DEVICE_CAPABILITIES").define().translate().cppTypes("bool"))
               .put(new Info("av_device_capabilities").skip());
    }
}
