// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


// This built-in op resolver enables XNNPACK by default for all types.
// Unsigned quantized inference (QU8) can be disabled by setting
// `enable_xnnpack_unsigned_quantized` to false. \warning Experimental
// interface, subject to change.
@Namespace("tflite::ops::builtin") @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class BuiltinOpResolverWithXNNPACK extends BuiltinOpResolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BuiltinOpResolverWithXNNPACK(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BuiltinOpResolverWithXNNPACK(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public BuiltinOpResolverWithXNNPACK position(long position) {
        return (BuiltinOpResolverWithXNNPACK)super.position(position);
    }
    @Override public BuiltinOpResolverWithXNNPACK getPointer(long i) {
        return new BuiltinOpResolverWithXNNPACK((Pointer)this).offsetAddress(i);
    }

  public BuiltinOpResolverWithXNNPACK(
        @Cast("bool") boolean enable_xnnpack_unsigned_quantized/*=true*/) { super((Pointer)null); allocate(enable_xnnpack_unsigned_quantized); }
  private native void allocate(
        @Cast("bool") boolean enable_xnnpack_unsigned_quantized/*=true*/);
  public BuiltinOpResolverWithXNNPACK() { super((Pointer)null); allocate(); }
  private native void allocate();
}
