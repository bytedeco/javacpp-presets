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


/** Options for the {@code AdaptiveLogSoftmaxWithLoss} module.
 * 
 *  Example:
 *  <pre>{@code
 *  AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
 *  {4, 8}).div_value(2.).head_bias(true));
 *  }</pre> */
@Namespace("torch::nn") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AdaptiveLogSoftmaxWithLossOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AdaptiveLogSoftmaxWithLossOptions(Pointer p) { super(p); }

  /* implicit */ public AdaptiveLogSoftmaxWithLossOptions(
      @Cast("int64_t") long in_features,
      @Cast("int64_t") long n_classes,
      @ByVal @Cast("std::vector<int64_t>*") LongVector cutoffs) { super((Pointer)null); allocate(in_features, n_classes, cutoffs); }
private native void allocate(
      @Cast("int64_t") long in_features,
      @Cast("int64_t") long n_classes,
      @ByVal @Cast("std::vector<int64_t>*") LongVector cutoffs);
  public native @Cast("int64_t*") @ByRef @NoException(true) LongPointer in_features();
  public native @Cast("int64_t*") @ByRef @NoException(true) LongPointer n_classes();
  public native @Cast("std::vector<int64_t>*") @ByRef @NoException(true) LongVector cutoffs();
  public native @ByRef @NoException(true) DoublePointer div_value();
  public native @Cast("bool*") @ByRef @NoException(true) BoolPointer head_bias();
}
