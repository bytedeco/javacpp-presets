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
 * Encoder Output parameters
 */
@Properties(inherit = org.bytedeco.nvcodec.presets.nvencodeapi.class)
public class NV_ENC_ENCODE_OUT_PARAMS extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NV_ENC_ENCODE_OUT_PARAMS() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NV_ENC_ENCODE_OUT_PARAMS(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NV_ENC_ENCODE_OUT_PARAMS(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NV_ENC_ENCODE_OUT_PARAMS position(long position) {
        return (NV_ENC_ENCODE_OUT_PARAMS)super.position(position);
    }
    @Override public NV_ENC_ENCODE_OUT_PARAMS getPointer(long i) {
        return new NV_ENC_ENCODE_OUT_PARAMS((Pointer)this).offsetAddress(i);
    }

    /** [out]: Struct version. */
    public native @Cast("uint32_t") int version(); public native NV_ENC_ENCODE_OUT_PARAMS version(int setter);
    /** [out]: Encoded bitstream size in bytes */
    public native @Cast("uint32_t") int bitstreamSizeInBytes(); public native NV_ENC_ENCODE_OUT_PARAMS bitstreamSizeInBytes(int setter);
    /** [out]: Reserved and must be set to 0 */
    public native @Cast("uint32_t") int reserved(int i); public native NV_ENC_ENCODE_OUT_PARAMS reserved(int i, int setter);
    @MemberGetter public native @Cast("uint32_t*") IntPointer reserved();
}
