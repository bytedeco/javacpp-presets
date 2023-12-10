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
    inherit = swresample.class,
    target = "org.bytedeco.ffmpeg.avcodec",
    global = "org.bytedeco.ffmpeg.global.avcodec",
    value = {
        @Platform(cinclude = {"<libavcodec/codec_id.h>", "<libavcodec/codec_desc.h>", "<libavcodec/defs.h>", "<libavcodec/codec_par.h>", "<libavcodec/packet.h>",
                              "<libavcodec/bsf.h>", "<libavcodec/codec.h>", "<libavcodec/avcodec.h>", "<libavcodec/jni.h>", "<libavcodec/avfft.h>", "<libavcodec/version_major.h>", "<libavcodec/version.h>"},
                  link = "avcodec@.60"),
        @Platform(value = "linux-arm", preload = {"asound@.2", "vchiq_arm", "vcos", "vcsm", "bcm_host", "mmal_core", "mmal_util", "mmal_vc_client"}),
        @Platform(value = "windows", preload = "avcodec-60")
    }
)
public class avcodec implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("!FF_API_LOWRES", "!FF_API_DEBUG_MV").define(false))
               .put(new Info("CODEC_FLAG_CLOSED_GOP").translate().cppTypes("long"))
               .put(new Info("LIBAVCODEC_VERSION").cppTypes())
               .put(new Info("LIBAVCODEC_VERSION_INT", "LIBAVCODEC_IDENT").translate(false))
               .put(new Info("FF_API_OPENH264_SLICE_MODE", "FF_API_OPENH264_CABAC", "FF_API_UNUSED_CODEC_CAPS", "FF_API_THREAD_SAFE_CALLBACKS",
                             "FF_API_DEBUG_MV", "FF_API_GET_FRAME_CLASS", "FF_API_AUTO_THREADS", "FF_API_INIT_PACKET", "FF_API_AVCTX_TIMEBASE",
                             "FF_API_MPEGVIDEO_OPTS", "FF_API_FLAG_TRUNCATED", "FF_API_SUB_TEXT_FORMAT", "FF_API_IDCT_NONE", "FF_API_SVTAV1_OPTS",
                             "FF_API_AYUV_CODECID", "FF_API_VT_OUTPUT_CALLBACK", "FF_API_AVCODEC_CHROMA_POS", "FF_API_VT_HWACCEL_CONTEXT",
                             "FF_API_AVCTX_FRAME_NUMBER", "FF_CODEC_CRYSTAL_HD", "FF_API_SLICE_OFFSET", "FF_API_SUBFRAMES", "FF_API_TICKS_PER_FRAME",
                             "FF_API_DROPCHANGED", "FF_API_AVFFT", "FF_API_FF_PROFILE_LEVEL").define().translate().cppTypes("bool"))
               .put(new Info("AVCodecInternal", "AVCodecHWConfigInternal").cast().pointerTypes("Pointer"))
               .put(new Info("AVCodec::hw_configs", "av_mdct_init", "av_imdct_calc", "av_imdct_half", "av_mdct_calc", "av_mdct_end").skip())
               .putFirst(new Info("AVPanScan").pointerTypes("AVPanScan"))
               .putFirst(new Info("AVCodecContext").pointerTypes("AVCodecContext"));
    }
}
