// Targeted by JavaCPP version 1.5.9-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.Module;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.pytorch.global.torch.*;


// An owning pointer to a Function. Just a pair of a raw Function ptr and it's
// owning CU. We need this because pybind requires a ref-counted way to refer to
// Functions.
@Namespace("torch::jit") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StrongFunctionPtr extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StrongFunctionPtr(Pointer p) { super(p); }

  public StrongFunctionPtr(@SharedPtr CompilationUnit cu, Function function) { super((Pointer)null); allocate(cu, function); }
  private native void allocate(@SharedPtr CompilationUnit cu, Function function);
  public native @SharedPtr CompilationUnit cu_(); public native StrongFunctionPtr cu_(CompilationUnit setter);
  public native Function function_(); public native StrongFunctionPtr function_(Function setter);
}