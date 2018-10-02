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

import org.bytedeco.javacpp.annotation.MemberGetter;
import org.bytedeco.javacpp.annotation.Name;
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
    @Platform(define="__STDC_CONSTANT_MACROS", cinclude={"<libavutil/avutil.h>", "<libavutil/error.h>", "<libavutil/mem.h>", "<libavutil/time.h>",
        "<libavutil/mathematics.h>", "<libavutil/rational.h>", "<libavutil/log.h>", "<libavutil/buffer.h>", "<libavutil/pixfmt.h>",
        "<libavutil/frame.h>", "<libavutil/samplefmt.h>", "<libavutil/channel_layout.h>", "<libavutil/cpu.h>", "<libavutil/dict.h>",
        "<libavutil/opt.h>", "<libavutil/pixdesc.h>", "<libavutil/imgutils.h>", "<libavutil/downmix_info.h>", "<libavutil/stereo3d.h>",
        "<libavutil/ffversion.h>", "<libavutil/motion_vector.h>", "<libavutil/fifo.h>", "<libavutil/audio_fifo.h>", "<libavutil/hwcontext.h>",
        "log_callback.h"},
        includepath={"/usr/local/include/ffmpeg/", "/opt/local/include/ffmpeg/", "/usr/include/ffmpeg/"},
        link="avutil@.56", compiler={"default", "nodeprecated"}),
    @Platform(value="windows", includepath={"C:/MinGW/local/include/ffmpeg/", "C:/MinGW/include/ffmpeg/"}, preload="avutil-56") })
public class avutil implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("AV_NOPTS_VALUE").cppTypes("int64_t").translate(false))
               .put(new Info("NAN", "INFINITY").cppTypes("double"))
               .put(new Info("AV_TIME_BASE_Q", "PixelFormat", "CodecID", "AVCOL_SPC_YCGCO", "AVCOL_SPC_YCOCG").cppTypes())
               .put(new Info("av_const").annotations("@Const"))
               .put(new Info("FF_CONST_AVUTIL55").annotations())
               .put(new Info("av_malloc_attrib", "av_alloc_size", "av_always_inline", "av_warn_unused_result").cppTypes().annotations())
               .put(new Info("attribute_deprecated").annotations("@Deprecated"))
               .put(new Info("AVPanScan", "AVCodecContext").cast().pointerTypes("Pointer"))
               .put(new Info("FF_API_VAAPI").define())
               .put(new Info("AV_PIX_FMT_ABI_GIT_MASTER", "AV_HAVE_INCOMPATIBLE_LIBAV_ABI", "!FF_API_XVMC",
                             "FF_API_GET_BITS_PER_SAMPLE_FMT", "FF_API_FIND_OPT").define(false))
               .put(new Info("ff_check_pixfmt_descriptors").skip())
               .put(new Info("AV_CH_FRONT_LEFT",
                             "AV_CH_FRONT_RIGHT",
                             "AV_CH_FRONT_CENTER",
                             "AV_CH_LOW_FREQUENCY",
                             "AV_CH_BACK_LEFT",
                             "AV_CH_BACK_RIGHT",
                             "AV_CH_FRONT_LEFT_OF_CENTER",
                             "AV_CH_FRONT_RIGHT_OF_CENTER",
                             "AV_CH_BACK_CENTER",
                             "AV_CH_SIDE_LEFT",
                             "AV_CH_SIDE_RIGHT",
                             "AV_CH_TOP_CENTER",
                             "AV_CH_TOP_FRONT_LEFT",
                             "AV_CH_TOP_FRONT_CENTER",
                             "AV_CH_TOP_FRONT_RIGHT",
                             "AV_CH_TOP_BACK_LEFT",
                             "AV_CH_TOP_BACK_CENTER",
                             "AV_CH_TOP_BACK_RIGHT",
                             "AV_CH_STEREO_LEFT",
                             "AV_CH_STEREO_RIGHT",
                             "AV_CH_WIDE_LEFT",
                             "AV_CH_WIDE_RIGHT",
                             "AV_CH_SURROUND_DIRECT_LEFT",
                             "AV_CH_SURROUND_DIRECT_RIGHT",
                             "AV_CH_LOW_FREQUENCY_2",
                             "AV_CH_LAYOUT_NATIVE",
                             "AV_CH_LAYOUT_MONO",
                             "AV_CH_LAYOUT_STEREO",
                             "AV_CH_LAYOUT_2POINT1",
                             "AV_CH_LAYOUT_2_1",
                             "AV_CH_LAYOUT_SURROUND",
                             "AV_CH_LAYOUT_3POINT1",
                             "AV_CH_LAYOUT_4POINT0",
                             "AV_CH_LAYOUT_4POINT1",
                             "AV_CH_LAYOUT_2_2",
                             "AV_CH_LAYOUT_QUAD",
                             "AV_CH_LAYOUT_5POINT0",
                             "AV_CH_LAYOUT_5POINT1",
                             "AV_CH_LAYOUT_5POINT0_BACK",
                             "AV_CH_LAYOUT_5POINT1_BACK",
                             "AV_CH_LAYOUT_6POINT0",
                             "AV_CH_LAYOUT_6POINT0_FRONT",
                             "AV_CH_LAYOUT_HEXAGONAL",
                             "AV_CH_LAYOUT_6POINT1",
                             "AV_CH_LAYOUT_6POINT1_BACK",
                             "AV_CH_LAYOUT_6POINT1_FRONT",
                             "AV_CH_LAYOUT_7POINT0",
                             "AV_CH_LAYOUT_7POINT0_FRONT",
                             "AV_CH_LAYOUT_7POINT1",
                             "AV_CH_LAYOUT_7POINT1_WIDE",
                             "AV_CH_LAYOUT_7POINT1_WIDE_BACK",
                             "AV_CH_LAYOUT_OCTAGONAL",
                             "AV_CH_LAYOUT_HEXADECAGONAL",
                             "AV_CH_LAYOUT_STEREO_DOWNMIX").translate().cppTypes("long"))
               .put(new Info("int (*)(void*, void*, int)").pointerTypes("Int_func_Pointer_Pointer_int"));
    }

    public static native @MemberGetter @Name("AVERROR(EACCES)") int AVERROR_EACCES();
    public static native @MemberGetter @Name("AVERROR(EAGAIN)") int AVERROR_EAGAIN();
    public static native @MemberGetter @Name("AVERROR(EBADF)") int AVERROR_EBADF();
    public static native @MemberGetter @Name("AVERROR(EDOM)") int AVERROR_EDOM();
    public static native @MemberGetter @Name("AVERROR(EEXIST)") int AVERROR_EEXIST();
    public static native @MemberGetter @Name("AVERROR(EFAULT)") int AVERROR_EFAULT();
    public static native @MemberGetter @Name("AVERROR(EFBIG)") int AVERROR_EFBIG();
    public static native @MemberGetter @Name("AVERROR(EILSEQ)") int AVERROR_EILSEQ();
    public static native @MemberGetter @Name("AVERROR(EINTR)") int AVERROR_EINTR();
    public static native @MemberGetter @Name("AVERROR(EINVAL)") int AVERROR_EINVAL();
    public static native @MemberGetter @Name("AVERROR(EIO)") int AVERROR_EIO();
    public static native @MemberGetter @Name("AVERROR(ENAMETOOLONG)") int AVERROR_ENAMETOOLONG();
    public static native @MemberGetter @Name("AVERROR(ENODEV)") int AVERROR_ENODEV();
    public static native @MemberGetter @Name("AVERROR(ENOENT)") int AVERROR_ENOENT();
    public static native @MemberGetter @Name("AVERROR(ENOMEM)") int AVERROR_ENOMEM();
    public static native @MemberGetter @Name("AVERROR(ENOSPC)") int AVERROR_ENOSPC();
    public static native @MemberGetter @Name("AVERROR(ENOSYS)") int AVERROR_ENOSYS();
    public static native @MemberGetter @Name("AVERROR(ENXIO)") int AVERROR_ENXIO();
    public static native @MemberGetter @Name("AVERROR(EPERM)") int AVERROR_EPERM();
    public static native @MemberGetter @Name("AVERROR(EPIPE)") int AVERROR_EPIPE();
    public static native @MemberGetter @Name("AVERROR(ERANGE)") int AVERROR_ERANGE();
    public static native @MemberGetter @Name("AVERROR(ESPIPE)") int AVERROR_ESPIPE();
    public static native @MemberGetter @Name("AVERROR(EXDEV)") int AVERROR_EXDEV();
}
