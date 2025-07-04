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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EmbeddingBag
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** Computes sums or means of 'bags' of embeddings, without instantiating the
 *  intermediate embeddings.
 *  See https://pytorch.org/docs/main/nn.html#torch.nn.EmbeddingBag to learn
 *  about the exact behavior of this module.
 * 
 *  See the documentation for {@code torch::nn::EmbeddingBagOptions} class to learn
 *  what constructor arguments are supported for this module.
 * 
 *  Example:
 *  <pre>{@code
 *  EmbeddingBag model(EmbeddingBagOptions(10,
 *  2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch::kSum).padding_idx(1));
 *  }</pre> */
@Namespace("torch::nn") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class EmbeddingBagImpl extends EmbeddingBagImplCloneable {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EmbeddingBagImpl(Pointer p) { super(p); }

  public EmbeddingBagImpl(@Cast("int64_t") long num_embeddings, @Cast("int64_t") long embedding_dim) { super((Pointer)null); allocate(num_embeddings, embedding_dim); }
  @SharedPtr @Name("std::make_shared<torch::nn::EmbeddingBagImpl>") private native void allocate(@Cast("int64_t") long num_embeddings, @Cast("int64_t") long embedding_dim);
  public EmbeddingBagImpl(@ByVal EmbeddingBagOptions options_) { super((Pointer)null); allocate(options_); }
  @SharedPtr @Name("std::make_shared<torch::nn::EmbeddingBagImpl>") private native void allocate(@ByVal EmbeddingBagOptions options_);

  public native void reset();

  public native void reset_parameters();

  /** Pretty prints the {@code EmbeddingBag} module into the given {@code stream}. */
  public native void pretty_print(@Cast("std::ostream*") @ByRef Pointer stream);

  /** The {@code Options} used to configure this {@code EmbeddingBag} module. */
  public native @ByRef EmbeddingBagOptions options(); public native EmbeddingBagImpl options(EmbeddingBagOptions setter);
  /** The embedding table. */
  public native @ByRef Tensor weight(); public native EmbeddingBagImpl weight(Tensor setter);

  public native @ByVal Tensor forward(
        @Const @ByRef Tensor input,
        @Const @ByRef(nullValue = "torch::Tensor{}") Tensor offsets,
        @Const @ByRef(nullValue = "torch::Tensor{}") Tensor per_sample_weights);
  public native @ByVal Tensor forward(
        @Const @ByRef Tensor input);
}
