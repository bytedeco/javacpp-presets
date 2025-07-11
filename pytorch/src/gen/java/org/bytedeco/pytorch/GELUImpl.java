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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** Applies gelu over a given input.
 *  See https://pytorch.org/docs/main/nn.html#torch.nn.GELU to learn
 *  about the exact behavior of this module. */
@Namespace("torch::nn") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GELUImpl extends GELUImplCloneable {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GELUImpl(Pointer p) { super(p); }

  public GELUImpl(@ByVal(nullValue = "torch::nn::GELUOptions{}") GELUOptions options_) { super((Pointer)null); allocate(options_); }
  @SharedPtr @Name("std::make_shared<torch::nn::GELUImpl>") private native void allocate(@ByVal(nullValue = "torch::nn::GELUOptions{}") GELUOptions options_);
  public GELUImpl() { super((Pointer)null); allocate(); }
  @SharedPtr @Name("std::make_shared<torch::nn::GELUImpl>") private native void allocate();

  public native @ByVal Tensor forward(@Const @ByRef Tensor input);

  public native void reset();

  /** Pretty prints the {@code GELU} module into the given {@code stream}. */
  public native void pretty_print(@Cast("std::ostream*") @ByRef Pointer stream);

  /** The options with which this {@code Module} was constructed. */
  public native @ByRef GELUOptions options(); public native GELUImpl options(GELUOptions setter);
}
