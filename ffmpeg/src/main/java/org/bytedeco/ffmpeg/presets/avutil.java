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

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.MemberGetter;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = javacpp.class,
    target = "org.bytedeco.ffmpeg.avutil",
    global = "org.bytedeco.ffmpeg.global.avutil",
    value = {
        @Platform(define = {"__STDC_CONSTANT_MACROS", "__STDC_FORMAT_MACROS", "STRING_BYTES_CHARSET \"UTF-8\""},
            cinclude = {"<libavutil/avutil.h>", /*"<libavutil/attributes.h>",*/ "<libavutil/error.h>", "<libavutil/mem.h>", "<libavutil/time.h>",
            "<libavutil/mathematics.h>", "<libavutil/rational.h>", "<libavutil/log.h>", "<libavutil/buffer.h>", "<libavutil/pixfmt.h>",
            "<libavutil/frame.h>", "<libavutil/samplefmt.h>", "<libavutil/channel_layout.h>", "<libavutil/cpu.h>", "<libavutil/dict.h>",
            "<libavutil/opt.h>", "<libavutil/pixdesc.h>", "<libavutil/imgutils.h>", "<libavutil/downmix_info.h>", "<libavutil/stereo3d.h>",
            "<libavutil/ffversion.h>", "<libavutil/motion_vector.h>", "<libavutil/fifo.h>", "<libavutil/audio_fifo.h>", "<libavutil/hwcontext.h>",
            /*"<libavutil/hwcontext_cuda.h>", "<libavutil/hwcontext_d3d11va.h>", "<libavutil/hwcontext_dxva2.h>", "<libavutil/hwcontext_drm.h>",
            "<libavutil/hwcontext_mediacodec.h>", "<libavutil/hwcontext_qsv.h>", "<libavutil/hwcontext_vaapi.h>", "<libavutil/hwcontext_vdpau.h>",
            "<libavutil/hwcontext_videotoolbox.h>",*/ "<libavutil/adler32.h>", "<libavutil/aes.h>", "<libavutil/aes_ctr.h>", "<libavutil/base64.h>",
            "<libavutil/blowfish.h>", "<libavutil/cast5.h>", "<libavutil/camellia.h>", "<libavutil/crc.h>", "<libavutil/des.h>", "<libavutil/lfg.h>",
            "<libavutil/hmac.h>", "<libavutil/md5.h>", "<libavutil/rc4.h>", "<libavutil/ripemd.h>", "<libavutil/tea.h>", "<libavutil/twofish.h>",
            "<libavutil/sha.h>", "<libavutil/sha512.h>", "<libavutil/xtea.h>", "<libavutil/avstring.h>", "<libavutil/bprint.h>", "<libavutil/common.h>",
            "<libavutil/display.h>", "<libavutil/eval.h>", "<libavutil/encryption_info.h>", "<libavutil/file.h>", "<libavutil/hash.h>",
            "<libavutil/hdr_dynamic_metadata.h>", "<libavutil/intfloat.h>", "<libavutil/intreadwrite.h>", "<libavutil/mastering_display_metadata.h>",
            "<libavutil/murmur3.h>", "<libavutil/parseutils.h>", "<libavutil/pixelutils.h>", "<libavutil/random_seed.h>", "<libavutil/replaygain.h>",
            "<libavutil/spherical.h>", "<libavutil/threadmessage.h>", "<libavutil/timecode.h>", "<libavutil/timestamp.h>", "<libavutil/tree.h>",
            "<libavutil/tx.h>", "<libavutil/version.h>", "<libavutil/macros.h>", "log_callback.h"},
            includepath = {"/usr/local/include/ffmpeg/", "/opt/local/include/ffmpeg/", "/usr/include/ffmpeg/"},
            link = "avutil@.58", compiler = {"default", "nodeprecated"}),
        @Platform(value = "linux-x86", preload = {"va@.1", "drm@.2", "va-drm@.1"}, preloadpath = {"/usr/lib32/", "/usr/lib/"}),
        @Platform(value = "linux-x86_64", preloadpath = {"/usr/lib64/", "/usr/lib/"}),
        @Platform(value = "windows", includepath = {"C:/MinGW/local/include/ffmpeg/", "C:/MinGW/include/ffmpeg/"}, preload = "avutil-58"),
        @Platform(extension = "-gpl")
    }
)
public class avutil implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "ffmpeg"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("AV_NOPTS_VALUE").cppTypes("int64_t").translate(false))
               .put(new Info("NAN", "INFINITY").cppTypes("double"))
               .put(new Info("AV_TIME_BASE_Q", "PixelFormat", "CodecID", "AVCOL_SPC_YCGCO", "AVCOL_SPC_YCOCG", "FF_CEIL_RSHIFT",
                             "av_ceil_log2", "av_clip", "av_clip64", "av_clip_uint8", "av_clip_int8", "av_clip_uint16", "av_clip_int16",
                             "av_clipl_int32", "av_clip_intp2", "av_clip_uintp2", "av_mod_uintp2", "av_sat_add32", "av_sat_dadd32",
                             "av_sat_sub32", "av_sat_dsub32", "av_clipf", "av_clipd", "av_popcount", "av_popcount64", "av_parity",
                             "av_sat_add64", "av_sat_sub64", "LIBAVUTIL_VERSION").cppTypes().translate())
               .put(new Info("LIBAVUTIL_VERSION_INT", "LIBAVUTIL_IDENT").translate(false))
               .put(new Info("FF_API_D2STR", "FF_API_DECLARE_ALIGNED", "FF_API_COLORSPACE_NAME", "FF_API_AV_MALLOCZ_ARRAY", "FF_API_FIFO_PEEK2",
                             "FF_API_FIFO_OLD_API", "FF_API_XVMC", "FF_API_OLD_CHANNEL_LAYOUT", "FF_API_AV_FOPEN_UTF8", "FF_API_PKT_DURATION",
                             "FF_API_REORDERED_OPAQUE", "FF_API_FRAME_PICTURE_NUMBER", "FF_API_HDR_VIVID_THREE_SPLINE", "FF_API_FRAME_PKT",
                             "FF_API_INTERLACED_FRAME", "FF_API_FRAME_KEY", "FF_API_PALETTE_HAS_CHANGED", "FF_API_VULKAN_CONTIGUOUS_MEMORY").define().translate().cppTypes("bool"))
               .put(new Info("av_const").annotations("@Const"))
               .put(new Info("FF_CONST_AVUTIL55").annotations())
               .put(new Info("av_malloc_attrib", "av_alloc_size", "av_always_inline", "av_warn_unused_result", "av_alias").cppTypes().annotations())
               .put(new Info("attribute_deprecated").annotations("@Deprecated"))
               .put(new Info("DWORD", "UINT").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("AVPanScan", "AVCodecContext", "AVMurMur3", "CUcontext", "CUstream",
                             "ID3D11Device", "ID3D11DeviceContext", "ID3D11Texture2D", "ID3D11VideoContext", "ID3D11VideoDevice",
                             "IDirect3DDeviceManager9", "IDirect3DSurface9", "IDirectXVideoDecoder", "mfxFrameSurface1", "mfxSession",
                             "VAConfigID", "VASurfaceID", "VASurfaceAttrib", "VADisplay", "VdpDevice", "VdpGetProcAddress").cast().pointerTypes("Pointer"))
               .put(new Info("FF_API_VAAPI").define())
               .put(new Info("AV_PIX_FMT_ABI_GIT_MASTER", "AV_HAVE_INCOMPATIBLE_LIBAV_ABI", "!FF_API_XVMC",
                             "FF_API_GET_BITS_PER_SAMPLE_FMT", "FF_API_FIND_OPT").define(false))
               .put(new Info("FF_API_BUFFER_SIZE_T", "FF_API_CRYPTO_SIZE_T").define(true))
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
                             "AV_CH_TOP_SIDE_LEFT",
                             "AV_CH_TOP_SIDE_RIGHT",
                             "AV_CH_BOTTOM_FRONT_CENTER",
                             "AV_CH_BOTTOM_FRONT_LEFT",
                             "AV_CH_BOTTOM_FRONT_RIGHT",
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
                             "AV_CH_LAYOUT_3POINT1POINT2",
                             "AV_CH_LAYOUT_6POINT1",
                             "AV_CH_LAYOUT_6POINT1_BACK",
                             "AV_CH_LAYOUT_6POINT1_FRONT",
                             "AV_CH_LAYOUT_7POINT0",
                             "AV_CH_LAYOUT_7POINT0_FRONT",
                             "AV_CH_LAYOUT_7POINT1",
                             "AV_CH_LAYOUT_7POINT1_WIDE",
                             "AV_CH_LAYOUT_7POINT1_WIDE_BACK",
                             "AV_CH_LAYOUT_5POINT1POINT2_BACK",
                             "AV_CH_LAYOUT_OCTAGONAL",
                             "AV_CH_LAYOUT_CUBE",
                             "AV_CH_LAYOUT_5POINT1POINT4_BACK",
                             "AV_CH_LAYOUT_7POINT1POINT2",
                             "AV_CH_LAYOUT_7POINT1POINT4_BACK",
                             "AV_CH_LAYOUT_HEXADECAGONAL",
                             "AV_CH_LAYOUT_STEREO_DOWNMIX",
                             "AV_CH_LAYOUT_22POINT2",
                             "AV_CH_LAYOUT_7POINT1_TOP_BACK").translate(true).cppTypes("long"))
               .put(new Info("AV_CHANNEL_LAYOUT_MONO",
                             "AV_CHANNEL_LAYOUT_STEREO",
                             "AV_CHANNEL_LAYOUT_2POINT1",
                             "AV_CHANNEL_LAYOUT_2_1",
                             "AV_CHANNEL_LAYOUT_SURROUND",
                             "AV_CHANNEL_LAYOUT_3POINT1",
                             "AV_CHANNEL_LAYOUT_4POINT0",
                             "AV_CHANNEL_LAYOUT_4POINT1",
                             "AV_CHANNEL_LAYOUT_2_2",
                             "AV_CHANNEL_LAYOUT_QUAD",
                             "AV_CHANNEL_LAYOUT_5POINT0",
                             "AV_CHANNEL_LAYOUT_5POINT1",
                             "AV_CHANNEL_LAYOUT_5POINT0_BACK",
                             "AV_CHANNEL_LAYOUT_5POINT1_BACK",
                             "AV_CHANNEL_LAYOUT_6POINT0",
                             "AV_CHANNEL_LAYOUT_6POINT0_FRONT",
                             "AV_CHANNEL_LAYOUT_3POINT1POINT2",
                             "AV_CHANNEL_LAYOUT_HEXAGONAL",
                             "AV_CHANNEL_LAYOUT_6POINT1",
                             "AV_CHANNEL_LAYOUT_6POINT1_BACK",
                             "AV_CHANNEL_LAYOUT_6POINT1_FRONT",
                             "AV_CHANNEL_LAYOUT_7POINT0",
                             "AV_CHANNEL_LAYOUT_7POINT0_FRONT",
                             "AV_CHANNEL_LAYOUT_7POINT1",
                             "AV_CHANNEL_LAYOUT_7POINT1_WIDE",
                             "AV_CHANNEL_LAYOUT_7POINT1_WIDE_BACK",
                             "AV_CHANNEL_LAYOUT_7POINT1_TOP_BACK",
                             "AV_CHANNEL_LAYOUT_5POINT1POINT2_BACK",
                             "AV_CHANNEL_LAYOUT_OCTAGONAL",
                             "AV_CHANNEL_LAYOUT_CUBE",
                             "AV_CHANNEL_LAYOUT_5POINT1POINT4_BACK",
                             "AV_CHANNEL_LAYOUT_7POINT1POINT2",
                             "AV_CHANNEL_LAYOUT_7POINT1POINT4_BACK",
                             "AV_CHANNEL_LAYOUT_HEXADECAGONAL",
                             "AV_CHANNEL_LAYOUT_STEREO_DOWNMIX",
                             "AV_CHANNEL_LAYOUT_22POINT2",
                             "AV_CHANNEL_LAYOUT_AMBISONIC_FIRST_ORDER").translate(false).cppTypes("AVChannelLayout"))
               .put(new Info("MKTAG", "MKBETAG").cppTypes("int", "char", "char", "char", "char"))
               .put(new Info("int (*)(const void*, const void*)").cast().pointerTypes("Cmp_Const_Pointer_Const_Pointer"))
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

    public static native @MemberGetter @Cast("void (*)(void*, int, const char*, va_list)") Pointer av_log_default_callback();
    @NoException public static native void av_log_set_callback(@Cast("void (*)(void*, int, const char*, va_list)") Pointer callback);
}
