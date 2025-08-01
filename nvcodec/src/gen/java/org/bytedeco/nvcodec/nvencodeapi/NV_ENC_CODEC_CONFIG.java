// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.nvcodec.nvencodeapi;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.nvcodec.nvcuvid.*;
import static org.bytedeco.nvcodec.global.nvcuvid.*;

import static org.bytedeco.nvcodec.global.nvencodeapi.*;


/**
 * \struct _NV_ENC_CODEC_CONFIG
 * Codec-specific encoder configuration parameters to be set during initialization.
 */
@Properties(inherit = org.bytedeco.nvcodec.presets.nvencodeapi.class)
public class NV_ENC_CODEC_CONFIG extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NV_ENC_CODEC_CONFIG() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NV_ENC_CODEC_CONFIG(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NV_ENC_CODEC_CONFIG(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NV_ENC_CODEC_CONFIG position(long position) {
        return (NV_ENC_CODEC_CONFIG)super.position(position);
    }
    @Override public NV_ENC_CODEC_CONFIG getPointer(long i) {
        return new NV_ENC_CODEC_CONFIG((Pointer)this).offsetAddress(i);
    }

    /** [in]: Specifies the H.264-specific encoder configuration. */
    public native @ByRef NV_ENC_CONFIG_H264 h264Config(); public native NV_ENC_CODEC_CONFIG h264Config(NV_ENC_CONFIG_H264 setter);
    /** [in]: Specifies the HEVC-specific encoder configuration. */
    public native @ByRef NV_ENC_CONFIG_HEVC hevcConfig(); public native NV_ENC_CODEC_CONFIG hevcConfig(NV_ENC_CONFIG_HEVC setter);
    /** [in]: Specifies the AV1-specific encoder configuration. */
    public native @ByRef NV_ENC_CONFIG_AV1 av1Config(); public native NV_ENC_CODEC_CONFIG av1Config(NV_ENC_CONFIG_AV1 setter);
    /** [in]: Specifies the H.264-specific ME only encoder configuration. */
    public native @ByRef NV_ENC_CONFIG_H264_MEONLY h264MeOnlyConfig(); public native NV_ENC_CODEC_CONFIG h264MeOnlyConfig(NV_ENC_CONFIG_H264_MEONLY setter);
    /** [in]: Specifies the HEVC-specific ME only encoder configuration. */
    public native @ByRef NV_ENC_CONFIG_HEVC_MEONLY hevcMeOnlyConfig(); public native NV_ENC_CODEC_CONFIG hevcMeOnlyConfig(NV_ENC_CONFIG_HEVC_MEONLY setter);
    /** [in]: Reserved and must be set to 0 */
    public native @Cast("uint32_t") int reserved(int i); public native NV_ENC_CODEC_CONFIG reserved(int i, int setter);
    @MemberGetter public native @Cast("uint32_t*") IntPointer reserved();
}
