// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.llvm.LLVM;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.llvm.global.LLVM.*;


/**
 * Represents an LLVM basic block builder.
 *
 * This models llvm::IRBuilder.
 */
@Name("LLVMOpaqueBuilder") @Opaque @Properties(inherit = org.bytedeco.llvm.presets.LLVM.class)
public class LLVMBuilderRef extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public LLVMBuilderRef() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LLVMBuilderRef(Pointer p) { super(p); }
}
