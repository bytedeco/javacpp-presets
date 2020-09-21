/*
 * Copyright (C) 2013-2018 Samuel Audet
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
    inherit = swresample.class,
    target = "org.bytedeco.ffmpeg.avcodec",
    global = "org.bytedeco.ffmpeg.global.avcodec",
    value = {
        @Platform(cinclude = {"<libavcodec/codec_id.h>", "<libavcodec/codec_desc.h>", "<libavcodec/codec_par.h>", "<libavcodec/packet.h>",
            "<libavcodec/bsf.h>", "<libavcodec/codec.h>", "<libavcodec/avcodec.h>", "<libavcodec/jni.h>", "<libavcodec/avfft.h>"},
            link = "avcodec@.58"),
        @Platform(cinclude = {"<libavcodec/codec_id.h>", "<libavcodec/codec_desc.h>", "<libavcodec/codec_par.h>", "<libavcodec/packet.h>",
            "<libavcodec/bsf.h>", "<libavcodec/codec.h>", "<libavcodec/avcodec.h>", "<libavcodec/jni.h>", "<libavcodec/avfft.h>"},
            link = "avcodec@.58",
            extension = "-gpl"),
        @Platform(value = "linux-arm", preload = {"asound@.2", "vchiq_arm", "vcos", "vcsm", "bcm_host", "mmal_core", "mmal_util", "mmal_vc_client"}),
        @Platform(value = "linux-arm", preload = {"asound@.2", "vchiq_arm", "vcos", "vcsm", "bcm_host", "mmal_core", "mmal_util", "mmal_vc_client"}, extension = "-gpl"),
        @Platform(value = "windows", preload = "avcodec-58"),
        @Platform(value = "windows", preload = "avcodec-58", extension = "-gpl"),
    }
)
public class avcodec implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("!FF_API_LOWRES", "!FF_API_DEBUG_MV").define(false))
               .put(new Info("CODEC_FLAG_CLOSED_GOP").translate().cppTypes("long"))
               .put(new Info("AVCodecHWConfigInternal").cast().pointerTypes("Pointer"))
               .putFirst(new Info("AVPanScan").pointerTypes("AVPanScan"))
               .putFirst(new Info("AVCodecContext").pointerTypes("AVCodecContext"));
    }
}
