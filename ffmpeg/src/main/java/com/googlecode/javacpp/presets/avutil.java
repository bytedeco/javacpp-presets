/*
 * Copyright (C) 2013 Samuel Audet
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

package com.googlecode.javacpp.presets;

import com.googlecode.javacpp.Parser;
import com.googlecode.javacpp.annotation.Platform;
import com.googlecode.javacpp.annotation.Properties;

/**
 *
 * @author Samuel Audet
 */
@Properties(target="com.googlecode.javacpp.avutil", value={
    @Platform(define="__STDC_CONSTANT_MACROS", cinclude={"<libavutil/avutil.h>", "<libavutil/error.h>", "<libavutil/mem.h>",
        "<libavutil/mathematics.h>", "<libavutil/rational.h>", "<libavutil/log.h>", "<libavutil/buffer.h>", "<libavutil/frame.h>",
        "<libavutil/pixfmt.h>", "<libavutil/samplefmt.h>", "<libavutil/channel_layout.h>", "<libavutil/cpu.h>", "<libavutil/dict.h>",
        "<libavutil/opt.h>", "<libavutil/audioconvert.h>", "<libavutil/pixdesc.h>", "<libavutil/imgutils.h>"},
        includepath={"/usr/local/include/ffmpeg/", "/opt/local/include/ffmpeg/", "/usr/include/ffmpeg/"},
        link="avutil@.52", options={"default", "nodeprecated"}),
    @Platform(value="windows", includepath={"C:/MinGW/local/include/ffmpeg/", "C:/MinGW/include/ffmpeg/"}, preload="avutil-52") })
public class avutil implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        infoMap.put(new Parser.Info("AV_NOPTS_VALUE").genericTypes("long"))
               .put(new Parser.Info("AV_TIME_BASE_Q", "PixelFormat", "CodecID").genericTypes())
               .put(new Parser.Info("av_const").annotations("@Const"))
               .put(new Parser.Info("av_malloc_attrib").genericTypes().annotations())
               .put(new Parser.Info("attribute_deprecated").annotations("@Deprecated"))
               .put(new Parser.Info("AVOptionRanges").forwardDeclared(true))
               .put(new Parser.Info("AVPanScan", "AVCodecContext").pointerTypes("Pointer").cast(true))
               .put(new Parser.Info("AV_PIX_FMT_ABI_GIT_MASTER", "AV_HAVE_INCOMPATIBLE_LIBAV_ABI").define(false))
               .put(new Parser.Info("ff_get_cpu_flags_arm", "ff_get_cpu_flags_ppc", "ff_get_cpu_flags_x86").javaNames());
    }
}
