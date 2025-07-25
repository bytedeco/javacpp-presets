// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.onnx;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.onnx.global.onnx.*;


@Namespace("google::protobuf::internal") @Properties(inherit = org.bytedeco.onnx.presets.onnx.class)
public class BoundedZCIS extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public BoundedZCIS() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BoundedZCIS(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BoundedZCIS(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public BoundedZCIS position(long position) {
        return (BoundedZCIS)super.position(position);
    }
    @Override public BoundedZCIS getPointer(long i) {
        return new BoundedZCIS((Pointer)this).offsetAddress(i);
    }

  public native ZeroCopyInputStream zcis(); public native BoundedZCIS zcis(ZeroCopyInputStream setter);
  public native @Name("limit") int _limit(); public native BoundedZCIS _limit(int setter);
}
