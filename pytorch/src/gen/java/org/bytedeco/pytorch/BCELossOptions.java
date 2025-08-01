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

/** Options for the {@code BCELoss} module.
 * 
 *  Example:
 *  <pre>{@code
 *  BCELoss model(BCELossOptions().reduction(torch::kNone).weight(weight));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BCELossOptions extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public BCELossOptions() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BCELossOptions(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BCELossOptions(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public BCELossOptions position(long position) {
        return (BCELossOptions)super.position(position);
    }
    @Override public BCELossOptions getPointer(long i) {
        return new BCELossOptions((Pointer)this).offsetAddress(i);
    }

  public native @ByRef @NoException(true) Tensor weight();
  public native @ByRef @NoException(true) LossReduction reduction();
}
