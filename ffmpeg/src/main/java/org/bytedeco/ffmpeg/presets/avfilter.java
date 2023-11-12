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
    inherit = {avformat.class, postproc.class, swresample.class, swscale.class},
    target = "org.bytedeco.ffmpeg.avfilter",
    global = "org.bytedeco.ffmpeg.global.avfilter",
    value = {
        @Platform(cinclude = {"<libavfilter/avfilter.h>", "<libavfilter/buffersink.h>", "<libavfilter/buffersrc.h>", "<libavfilter/version_major.h>", "<libavfilter/version.h>"}, link = "avfilter@.9"),
        @Platform(value = "windows", preload = "avfilter-9")
    }
)
public class avfilter implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("AVFilterPool", "AVFilterCommand", "AVFilterChannelLayouts", "FFFrameQueue").cast().pointerTypes("Pointer"))
               .put(new Info("LIBAVFILTER_VERSION").cppTypes())
               .put(new Info("LIBAVFILTER_VERSION_INT", "LIBAVFILTER_IDENT").translate(false))
               .put(new Info("FF_API_SWS_PARAM_OPTION", "FF_API_BUFFERSINK_ALLOC", "FF_API_PAD_COUNT", "FF_API_LIBPLACEBO_OPTS").define().translate().cppTypes("bool"))
               .put(new Info("AV_HAVE_INCOMPATIBLE_LIBAV_ABI || !FF_API_OLD_GRAPH_PARSE").define(true))
               .put(new Info("!FF_API_FOO_COUNT", "FF_INTERNAL_FIELDS").define(false));
    }
}
