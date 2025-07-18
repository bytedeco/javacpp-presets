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
 * Event registration/unregistration parameters.
 */
@Properties(inherit = org.bytedeco.nvcodec.presets.nvencodeapi.class)
public class NV_ENC_EVENT_PARAMS extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NV_ENC_EVENT_PARAMS() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NV_ENC_EVENT_PARAMS(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NV_ENC_EVENT_PARAMS(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NV_ENC_EVENT_PARAMS position(long position) {
        return (NV_ENC_EVENT_PARAMS)super.position(position);
    }
    @Override public NV_ENC_EVENT_PARAMS getPointer(long i) {
        return new NV_ENC_EVENT_PARAMS((Pointer)this).offsetAddress(i);
    }

    /** [in]: Struct version. Must be set to ::NV_ENC_EVENT_PARAMS_VER. */
    public native @Cast("uint32_t") int version(); public native NV_ENC_EVENT_PARAMS version(int setter);
    /** [in]: Reserved and must be set to 0 */
    public native @Cast("uint32_t") int reserved(); public native NV_ENC_EVENT_PARAMS reserved(int setter);
    /** [in]: Handle to event to be registered/unregistered with the NvEncodeAPI interface. */
    public native Pointer completionEvent(); public native NV_ENC_EVENT_PARAMS completionEvent(Pointer setter);
    /** [in]: Reserved and must be set to 0    */
    public native @Cast("uint32_t") int reserved1(int i); public native NV_ENC_EVENT_PARAMS reserved1(int i, int setter);
    @MemberGetter public native @Cast("uint32_t*") IntPointer reserved1();
    /** [in]: Reserved and must be set to NULL */
    public native Pointer reserved2(int i); public native NV_ENC_EVENT_PARAMS reserved2(int i, Pointer setter);
    @MemberGetter public native @Cast("void**") PointerPointer reserved2();
}
