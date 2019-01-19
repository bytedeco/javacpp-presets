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

package org.bytedeco.javacpp.avfilter;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.javacpp.avformat.*;
import org.bytedeco.javacpp.postproc.*;
import org.bytedeco.javacpp.swresample.*;
import org.bytedeco.javacpp.swscale.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit={avformat_presets.class, postproc_presets.class, swresample_presets.class, swscale_presets.class}, target="org.bytedeco.javacpp.avfilter", global="avfilter", value={
    @Platform(cinclude={"<libavfilter/avfilter.h>", "<libavfilter/buffersink.h>", "<libavfilter/buffersrc.h>"}, link="avfilter@.7"),
    @Platform(value="windows", preload="avfilter-7") })
public class avfilter_presets implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("AVFilterPool", "AVFilterCommand", "AVFilterChannelLayouts", "FFFrameQueue").cast().pointerTypes("Pointer"))
               .put(new Info("AV_HAVE_INCOMPATIBLE_LIBAV_ABI || !FF_API_OLD_GRAPH_PARSE").define(true))
               .put(new Info("!FF_API_FOO_COUNT", "FF_INTERNAL_FIELDS").define(false));
    }
}
