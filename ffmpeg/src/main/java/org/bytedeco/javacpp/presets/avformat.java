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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit=avcodec.class, target="org.bytedeco.javacpp.avformat", value={
    @Platform(cinclude={"<libavformat/avio.h>", "<libavformat/avformat.h>"}, link="avformat@.58"),
    @Platform(value="windows", preload="avformat-58") })
public class avformat implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("URLContext", "FFFrac").cast().pointerTypes("Pointer"))
               .put(new Info("AVPROBE_SCORE_RETRY", "AVPROBE_SCORE_STREAM_RETRY").translate(false))
               .put(new Info("LIBAVFORMAT_VERSION_MAJOR <= 54", "FF_API_ALLOC_OUTPUT_CONTEXT", "FF_API_FORMAT_PARAMETERS",
                             "FF_API_READ_PACKET", "FF_API_CLOSE_INPUT_FILE", "FF_API_NEW_STREAM", "FF_API_SET_PTS_INFO").define(false));
    }
}
