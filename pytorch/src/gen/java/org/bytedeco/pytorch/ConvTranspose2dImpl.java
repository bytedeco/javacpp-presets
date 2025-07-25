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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** Applies the ConvTranspose2d function.
 *  See https://pytorch.org/docs/main/nn.html#torch.nn.ConvTranspose2d to
 *  learn about the exact behavior of this module.
 * 
 *  See the documentation for {@code torch::nn::ConvTranspose2dOptions} class to learn
 *  what constructor arguments are supported for this module.
 * 
 *  Example:
 *  <pre>{@code
 *  ConvTranspose2d model(ConvTranspose2dOptions(3, 2,
 *  3).stride(1).bias(false));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ConvTranspose2dImpl extends ConvTranspose2dImplBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ConvTranspose2dImpl(Pointer p) { super(p); }

  public ConvTranspose2dImpl(
        @Cast("int64_t") long input_channels,
        @Cast("int64_t") long output_channels,
        @ByVal @Cast("torch::ExpandingArray<2>*") LongPointer kernel_size) { super((Pointer)null); allocate(input_channels, output_channels, kernel_size); }
  @SharedPtr @Name("std::make_shared<torch::nn::ConvTranspose2dImpl>") private native void allocate(
        @Cast("int64_t") long input_channels,
        @Cast("int64_t") long output_channels,
        @ByVal @Cast("torch::ExpandingArray<2>*") LongPointer kernel_size);
  public ConvTranspose2dImpl(@ByVal ConvTranspose2dOptions options_) { super((Pointer)null); allocate(options_); }
  @SharedPtr @Name("std::make_shared<torch::nn::ConvTranspose2dImpl>") private native void allocate(@ByVal ConvTranspose2dOptions options_);
  public native @ByVal Tensor forward(
        @Const @ByRef Tensor input,
        @Const @ByRef(nullValue = "std::optional<at::IntArrayRef>(std::nullopt)") LongArrayRefOptional output_size);
  public native @ByVal Tensor forward(
        @Const @ByRef Tensor input);
  public native @ByVal Tensor forward(
        @Const @ByRef Tensor input,
        @ByRef(nullValue = "std::optional<at::IntArrayRef>(std::nullopt)") @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector long... output_size);
}
