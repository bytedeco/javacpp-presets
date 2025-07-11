// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


// TfLite interpreter could apply a TfLite delegate by default. To completely
// disable this behavior, one could choose to use the following class
// BuiltinOpResolverWithoutDefaultDelegates.
@Namespace("tflite::ops::builtin") @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class BuiltinOpResolverWithoutDefaultDelegates extends BuiltinOpResolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BuiltinOpResolverWithoutDefaultDelegates(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BuiltinOpResolverWithoutDefaultDelegates(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public BuiltinOpResolverWithoutDefaultDelegates position(long position) {
        return (BuiltinOpResolverWithoutDefaultDelegates)super.position(position);
    }
    @Override public BuiltinOpResolverWithoutDefaultDelegates getPointer(long i) {
        return new BuiltinOpResolverWithoutDefaultDelegates((Pointer)this).offsetAddress(i);
    }

  public BuiltinOpResolverWithoutDefaultDelegates() { super((Pointer)null); allocate(); }
  private native void allocate();
}
