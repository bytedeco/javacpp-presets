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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad1d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** Applies ReflectionPad over a 1-D input.
 *  See https://pytorch.org/docs/main/nn.html#torch.nn.ReflectionPad1d to
 *  learn about the exact behavior of this module.
 * 
 *  See the documentation for {@code torch::nn::ReflectionPad1dOptions} class to learn
 *  what constructor arguments are supported for this module.
 * 
 *  Example:
 *  <pre>{@code
 *  ReflectionPad1d model(ReflectionPad1dOptions({3, 1}));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ReflectionPad1dImpl extends ReflectionPad1dImplBase {
    static { Loader.load(); }

  
    public ReflectionPad1dImpl(@ByVal @Cast("torch::ExpandingArray<1*2>*") LongPointer padding) { super((Pointer)null); allocate(padding); }
    private native void allocate(@ByVal @Cast("torch::ExpandingArray<1*2>*") LongPointer padding);
    public ReflectionPad1dImpl(@Const @ByRef ReflectionPad1dOptions options_) { super((Pointer)null); allocate(options_); }
    private native void allocate(@Const @ByRef ReflectionPad1dOptions options_);
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ReflectionPad1dImpl(Pointer p) { super(p); }

}
