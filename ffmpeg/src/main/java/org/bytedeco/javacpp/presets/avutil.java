/*
 * Copyright (C) 2013,2014 Samuel Audet
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
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
@Properties(target="org.bytedeco.javacpp.avutil", value={
    @Platform(define="__STDC_CONSTANT_MACROS", cinclude={"<libavutil/avutil.h>", "<libavutil/error.h>", "<libavutil/mem.h>",
        "<libavutil/mathematics.h>", "<libavutil/rational.h>", "<libavutil/log.h>", "<libavutil/buffer.h>", "<libavutil/pixfmt.h>",
        "<libavutil/frame.h>", "<libavutil/samplefmt.h>", "<libavutil/channel_layout.h>", "<libavutil/cpu.h>", "<libavutil/dict.h>",
        "<libavutil/opt.h>", "<libavutil/audioconvert.h>", "<libavutil/pixdesc.h>", "<libavutil/imgutils.h>",
        "<libavutil/downmix_info.h>", "<libavutil/stereo3d.h>"},
        includepath={"/usr/local/include/ffmpeg/", "/opt/local/include/ffmpeg/", "/usr/include/ffmpeg/"},
        link="avutil@.54", compiler={"default", "nodeprecated"}),
    @Platform(value="windows", includepath={"C:/MinGW/local/include/ffmpeg/", "C:/MinGW/include/ffmpeg/"}, preload="avutil-54") })
public class avutil implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("AV_NOPTS_VALUE").cppTypes("int64_t").translate(false))
               .put(new Info("AV_TIME_BASE_Q", "PixelFormat", "CodecID").cppTypes())
               .put(new Info("av_const").annotations("@Const"))
               .put(new Info("FF_CONST_AVUTIL55").annotations())
               .put(new Info("av_malloc_attrib", "av_alloc_size", "av_always_inline").cppTypes().annotations())
               .put(new Info("attribute_deprecated").annotations("@Deprecated"))
               .put(new Info("AVPanScan", "AVCodecContext").cast().pointerTypes("Pointer"))
               .put(new Info("AV_PIX_FMT_ABI_GIT_MASTER", "AV_HAVE_INCOMPATIBLE_LIBAV_ABI", "!FF_API_XVMC",
                             "FF_API_GET_BITS_PER_SAMPLE_FMT", "FF_API_FIND_OPT").define(false))
               .put(new Info("AV_PIX_FMT_Y400A", "ff_check_pixfmt_descriptors").skip());
    }
}
