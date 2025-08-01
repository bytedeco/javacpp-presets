// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.llvm.LLVM;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.llvm.global.LLVM.*;


/**
 * Represents a SymbolAliasMapEntry
 */
@Properties(inherit = org.bytedeco.llvm.presets.LLVM.class)
public class LLVMOrcCSymbolAliasMapEntry extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public LLVMOrcCSymbolAliasMapEntry() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public LLVMOrcCSymbolAliasMapEntry(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LLVMOrcCSymbolAliasMapEntry(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public LLVMOrcCSymbolAliasMapEntry position(long position) {
        return (LLVMOrcCSymbolAliasMapEntry)super.position(position);
    }
    @Override public LLVMOrcCSymbolAliasMapEntry getPointer(long i) {
        return new LLVMOrcCSymbolAliasMapEntry((Pointer)this).offsetAddress(i);
    }

  public native LLVMOrcSymbolStringPoolEntryRef Name(); public native LLVMOrcCSymbolAliasMapEntry Name(LLVMOrcSymbolStringPoolEntryRef setter);
  public native @ByRef LLVMJITSymbolFlags Flags(); public native LLVMOrcCSymbolAliasMapEntry Flags(LLVMJITSymbolFlags setter);
}
