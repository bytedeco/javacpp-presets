// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.llvm.clang;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.llvm.LLVM.*;
import static org.bytedeco.llvm.global.LLVM.*;

import static org.bytedeco.llvm.global.clang.*;


/**
 * Contains the results of a search in the compilation database
 *
 * When searching for the compile command for a file, the compilation db can
 * return several commands, as the file may have been compiled with
 * different options in different places of the project. This choice of compile
 * commands is wrapped in this opaque data structure. It must be freed by
 * \c clang_CompileCommands_dispose.
 */
@Namespace @Name("void") @Opaque @Properties(inherit = org.bytedeco.llvm.presets.clang.class)
public class CXCompileCommands extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public CXCompileCommands() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CXCompileCommands(Pointer p) { super(p); }
}
