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
 * \struct _NV_ENC_PRESET_CONFIG
 * Encoder preset config
 */
@Properties(inherit = org.bytedeco.nvcodec.presets.nvencodeapi.class)
public class NV_ENC_PRESET_CONFIG extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NV_ENC_PRESET_CONFIG() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NV_ENC_PRESET_CONFIG(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NV_ENC_PRESET_CONFIG(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NV_ENC_PRESET_CONFIG position(long position) {
        return (NV_ENC_PRESET_CONFIG)super.position(position);
    }
    @Override public NV_ENC_PRESET_CONFIG getPointer(long i) {
        return new NV_ENC_PRESET_CONFIG((Pointer)this).offsetAddress(i);
    }

    /** [in]:  Struct version. Must be set to ::NV_ENC_PRESET_CONFIG_VER. */
    public native @Cast("uint32_t") int version(); public native NV_ENC_PRESET_CONFIG version(int setter);
    /** [in]: Reserved and must be set to 0 */
    public native @Cast("uint32_t") int reserved(); public native NV_ENC_PRESET_CONFIG reserved(int setter);
    /** [out]: preset config returned by the Nvidia Video Encoder interface. */
    public native @ByRef NV_ENC_CONFIG presetCfg(); public native NV_ENC_PRESET_CONFIG presetCfg(NV_ENC_CONFIG setter);
    /** [in]: Reserved and must be set to 0 */
    public native @Cast("uint32_t") int reserved1(int i); public native NV_ENC_PRESET_CONFIG reserved1(int i, int setter);
    @MemberGetter public native @Cast("uint32_t*") IntPointer reserved1();
    /** [in]: Reserved and must be set to NULL */
    public native Pointer reserved2(int i); public native NV_ENC_PRESET_CONFIG reserved2(int i, Pointer setter);
    @MemberGetter public native @Cast("void**") PointerPointer reserved2();
}
