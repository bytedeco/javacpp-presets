// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.javacpp.chrono.*;
import static org.bytedeco.javacpp.global.chrono.*;

import static org.bytedeco.pytorch.global.torch.*;
 // namespace functional

// ============================================================================

/** Options for the {@code GLU} module.
 * 
 *  Example:
 *  <pre>{@code
 *  GLU model(GLUOptions(1));
 *  }</pre> */
@Namespace("torch::nn") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GLUOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GLUOptions(Pointer p) { super(p); }

  /* implicit */ public GLUOptions(@Cast("int64_t") long dim/*=-1*/) { super((Pointer)null); allocate(dim); }
private native void allocate(@Cast("int64_t") long dim/*=-1*/);
public GLUOptions() { super((Pointer)null); allocate(); }
private native void allocate();
  public native @Cast("int64_t*") @ByRef @NoException(true) LongPointer dim();
}
