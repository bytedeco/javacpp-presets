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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveLogSoftmaxWithLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** Efficient softmax approximation as described in
 *  {@code Efficient softmax approximation for GPUs}_ by Edouard Grave, Armand Joulin,
 *  Moustapha Cissé, David Grangier, and Hervé Jégou.
 *  See
 *  https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveLogSoftmaxWithLoss
 *  to learn about the exact behavior of this module.
 * 
 *  See the documentation for {@code torch::nn::AdaptiveLogSoftmaxWithLossOptions}
 *  class to learn what constructor arguments are supported for this module.
 * 
 *  Example:
 *  <pre>{@code
 *  AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
 *  {4, 8}).div_value(2.).head_bias(true));
 *  }</pre> */
@Namespace("torch::nn") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AdaptiveLogSoftmaxWithLossImpl extends AdaptiveLogSoftmaxWithLossImplCloneable {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AdaptiveLogSoftmaxWithLossImpl(Pointer p) { super(p); }

  public AdaptiveLogSoftmaxWithLossImpl(
        @Cast("int64_t") long in_features,
        @Cast("int64_t") long n_classes,
        @ByVal @Cast("std::vector<int64_t>*") LongVector cutoffs) { super((Pointer)null); allocate(in_features, n_classes, cutoffs); }
  @SharedPtr @Name("std::make_shared<torch::nn::AdaptiveLogSoftmaxWithLossImpl>") private native void allocate(
        @Cast("int64_t") long in_features,
        @Cast("int64_t") long n_classes,
        @ByVal @Cast("std::vector<int64_t>*") LongVector cutoffs);

  public AdaptiveLogSoftmaxWithLossImpl(
        @ByVal AdaptiveLogSoftmaxWithLossOptions options_) { super((Pointer)null); allocate(options_); }
  @SharedPtr @Name("std::make_shared<torch::nn::AdaptiveLogSoftmaxWithLossImpl>") private native void allocate(
        @ByVal AdaptiveLogSoftmaxWithLossOptions options_);

  public native @ByVal ASMoutput forward(@Const @ByRef Tensor input, @Const @ByRef Tensor target);

  public native void reset();

  public native void reset_parameters();

  /** Pretty prints the {@code AdaptiveLogSoftmaxWithLoss} module into the given
   *  {@code stream}. */
  public native void pretty_print(@Cast("std::ostream*") @ByRef Pointer stream);

  /** Given input tensor, and output of {@code head}, computes the log of the full
   *  distribution */
  public native @ByVal Tensor _get_full_log_prob(@Const @ByRef Tensor input, @Const @ByRef Tensor head_output);

  /** Computes log probabilities for all n_classes */
  public native @ByVal Tensor log_prob(@Const @ByRef Tensor input);

  /** This is equivalent to {@code log_pob(input).argmax(1)} but is more efficient in
   *  some cases */
  public native @ByVal Tensor predict(@Const @ByRef Tensor input);

  /** The options with which this {@code Module} was constructed */
  public native @ByRef AdaptiveLogSoftmaxWithLossOptions options(); public native AdaptiveLogSoftmaxWithLossImpl options(AdaptiveLogSoftmaxWithLossOptions setter);

  /** Cutoffs used to assign targets to their buckets. It should be an ordered
   *  Sequence of integers sorted in the increasing order */
  public native @ByRef @Cast("std::vector<int64_t>*") LongVector cutoffs(); public native AdaptiveLogSoftmaxWithLossImpl cutoffs(LongVector setter);

  public native @Cast("int64_t") long shortlist_size(); public native AdaptiveLogSoftmaxWithLossImpl shortlist_size(long setter);

  /** Number of clusters */
  public native @Cast("int64_t") long n_clusters(); public native AdaptiveLogSoftmaxWithLossImpl n_clusters(long setter);

  /** Output size of head classifier */
  public native @Cast("int64_t") long head_size(); public native AdaptiveLogSoftmaxWithLossImpl head_size(long setter);
}
