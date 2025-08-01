// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.llvm.clang;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.llvm.LLVM.*;
import static org.bytedeco.llvm.global.LLVM.*;

import static org.bytedeco.llvm.global.clang.*;


@Properties(inherit = org.bytedeco.llvm.presets.clang.class)
public class CXTUResourceUsageEntry extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CXTUResourceUsageEntry() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CXTUResourceUsageEntry(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CXTUResourceUsageEntry(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CXTUResourceUsageEntry position(long position) {
        return (CXTUResourceUsageEntry)super.position(position);
    }
    @Override public CXTUResourceUsageEntry getPointer(long i) {
        return new CXTUResourceUsageEntry((Pointer)this).offsetAddress(i);
    }

  /* The memory usage category. */
  public native @Cast("CXTUResourceUsageKind") int kind(); public native CXTUResourceUsageEntry kind(int setter);
  /* Amount of resources used.
      The units will depend on the resource kind. */
  public native @Cast("unsigned long") long amount(); public native CXTUResourceUsageEntry amount(long setter);
}
