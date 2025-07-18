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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FractionalMaxPool2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** Applies fractional maxpool over a 2-D input.
 *  See https://pytorch.org/docs/main/nn.html#torch.nn.FractionalMaxPool2d to
 *  learn about the exact behavior of this module.
 * 
 *  See the documentation for {@code torch::nn::FractionalMaxPool2dOptions} class to
 *  learn what constructor arguments are supported for this module.
 * 
 *  Example:
 *  <pre>{@code
 *  FractionalMaxPool2d model(FractionalMaxPool2dOptions(5).output_size(1));
 *  }</pre> */
@Namespace("torch::nn") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FractionalMaxPool2dImpl extends FractionalMaxPool2dImplCloneable {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FractionalMaxPool2dImpl(Pointer p) { super(p); }

  public FractionalMaxPool2dImpl(@ByVal @Cast("torch::ExpandingArray<2>*") LongPointer kernel_size) { super((Pointer)null); allocate(kernel_size); }
  @SharedPtr @Name("std::make_shared<torch::nn::FractionalMaxPool2dImpl>") private native void allocate(@ByVal @Cast("torch::ExpandingArray<2>*") LongPointer kernel_size);
  public FractionalMaxPool2dImpl(@ByVal FractionalMaxPool2dOptions options_) { super((Pointer)null); allocate(options_); }
  @SharedPtr @Name("std::make_shared<torch::nn::FractionalMaxPool2dImpl>") private native void allocate(@ByVal FractionalMaxPool2dOptions options_);

  public native void reset();

  /** Pretty prints the {@code FractionalMaxPool2d} module into the given {@code stream}. */
  public native void pretty_print(@Cast("std::ostream*") @ByRef Pointer stream);

  public native @ByVal Tensor forward(@Const @ByRef Tensor input);

  /** Returns the outputs and the indices of the max values.
   *  Useful for {@code torch::nn::MaxUnpool2d} later. */
  public native @ByVal T_TensorTensor_T forward_with_indices(@Const @ByRef Tensor input);

  /** The options with which this {@code Module} was constructed. */
  public native @ByRef FractionalMaxPool2dOptions options(); public native FractionalMaxPool2dImpl options(FractionalMaxPool2dOptions setter);

  public native @ByRef Tensor _random_samples(); public native FractionalMaxPool2dImpl _random_samples(Tensor setter);
}
