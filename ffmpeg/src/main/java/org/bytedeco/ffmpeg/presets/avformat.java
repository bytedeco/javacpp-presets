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
    inherit = avcodec.class,
    target = "org.bytedeco.ffmpeg.avformat",
    global = "org.bytedeco.ffmpeg.global.avformat",
    value = {
        @Platform(cinclude = {"<libavformat/avio.h>", "<libavformat/avformat.h>", "<libavformat/version_major.h>", "<libavformat/version.h>"}, link = "avformat@.60"),
        @Platform(value = "windows", preload = "avformat-60")
    }
)
public class avformat implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("AVDeviceInfoList", "AVDeviceCapabilitiesQuery", "AVBPrint", "URLContext", "FFFrac").cast().pointerTypes("Pointer"))
               .put(new Info("LIBAVFORMAT_VERSION").cppTypes())
               .put(new Info("LIBAVFORMAT_VERSION_INT", "LIBAVFORMAT_IDENT", "AVPROBE_SCORE_RETRY", "AVPROBE_SCORE_STREAM_RETRY").translate(false))
               .put(new Info("FF_API_LAVF_PRIV_OPT", "FF_API_COMPUTE_PKT_FIELDS2", "FF_API_AVIOCONTEXT_WRITTEN",
                             "FF_HLS_TS_OPTIONS", "FF_HTTP_CACHE_REDIRECT_DEFAULT", "FF_API_GET_END_PTS", "FF_API_AVIODIRCONTEXT",
                             "FF_API_AVFORMAT_IO_CLOSE", "FF_API_AVIO_WRITE_NONCONST", "FF_API_LAVF_SHORTEST", "FF_API_ALLOW_FLUSH",
                             "FF_API_AVSTREAM_SIDE_DATA").define().translate().cppTypes("bool"))
               .put(new Info("LIBAVFORMAT_VERSION_MAJOR <= 54", "FF_API_ALLOC_OUTPUT_CONTEXT", "FF_API_FORMAT_PARAMETERS",
                             "FF_API_READ_PACKET", "FF_API_CLOSE_INPUT_FILE", "FF_API_NEW_STREAM", "FF_API_SET_PTS_INFO",
                             "FF_API_AVSTREAM_CLASS").define(false).translate().cppTypes("bool"));
    }
}
