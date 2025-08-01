// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.llvm.LLVM;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.llvm.global.LLVM.*;


/**
 * Represents a pair of a JITDylib and LLVMOrcCSymbolsList.
 */
@Properties(inherit = org.bytedeco.llvm.presets.LLVM.class)
public class LLVMOrcCDependenceMapPair extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public LLVMOrcCDependenceMapPair() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public LLVMOrcCDependenceMapPair(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LLVMOrcCDependenceMapPair(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public LLVMOrcCDependenceMapPair position(long position) {
        return (LLVMOrcCDependenceMapPair)super.position(position);
    }
    @Override public LLVMOrcCDependenceMapPair getPointer(long i) {
        return new LLVMOrcCDependenceMapPair((Pointer)this).offsetAddress(i);
    }

  public native LLVMOrcJITDylibRef JD(); public native LLVMOrcCDependenceMapPair JD(LLVMOrcJITDylibRef setter);
  public native @ByRef LLVMOrcCSymbolsList Names(); public native LLVMOrcCDependenceMapPair Names(LLVMOrcCSymbolsList setter);
}
