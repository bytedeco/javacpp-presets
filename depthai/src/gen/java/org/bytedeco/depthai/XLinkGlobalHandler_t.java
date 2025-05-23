// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.depthai;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import static org.bytedeco.depthai.global.depthai.*;


@Properties(inherit = org.bytedeco.depthai.presets.depthai.class)
public class XLinkGlobalHandler_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public XLinkGlobalHandler_t() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public XLinkGlobalHandler_t(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public XLinkGlobalHandler_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public XLinkGlobalHandler_t position(long position) {
        return (XLinkGlobalHandler_t)super.position(position);
    }
    @Override public XLinkGlobalHandler_t getPointer(long i) {
        return new XLinkGlobalHandler_t((Pointer)this).offsetAddress(i);
    }

    public native int profEnable(); public native XLinkGlobalHandler_t profEnable(int setter);
    public native @ByRef XLinkProf_t profilingData(); public native XLinkGlobalHandler_t profilingData(XLinkProf_t setter);
    public native Pointer options(); public native XLinkGlobalHandler_t options(Pointer setter);

    //Deprecated fields. Begin.
    public native int loglevel(); public native XLinkGlobalHandler_t loglevel(int setter);
    public native int protocol(); public native XLinkGlobalHandler_t protocol(int setter);
    //Deprecated fields. End.
}
