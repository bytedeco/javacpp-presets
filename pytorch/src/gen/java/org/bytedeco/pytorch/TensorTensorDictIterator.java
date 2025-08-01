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

@Name("c10::impl::DictIterator<torch::Tensor,torch::Tensor,c10::detail::DictImpl::dict_map_type::iterator>") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorTensorDictIterator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorTensorDictIterator(Pointer p) { super(p); }

   // C++17 friendly std::iterator implementation
  public native @ByRef @Name("operator =") TensorTensorDictIterator put(@Const @ByRef TensorTensorDictIterator rhs);

  public native @ByRef @Name("operator ++") TensorTensorDictIterator increment();

  public native @ByVal @Name("operator ++") TensorTensorDictIterator increment(int arg0);

  public native @Const @ByRef @Name("operator *") GenericDictEntryRef multiply();

  public native @Const @Name("operator ->") GenericDictEntryRef access();

  

  private static native @Namespace @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef TensorTensorDictIterator lhs, @Const @ByRef TensorTensorDictIterator rhs);
  public boolean equals(TensorTensorDictIterator rhs) { return equals(this, rhs); }

  private static native @Namespace @Cast("bool") @Name("operator !=") boolean notEquals(@Const @ByRef TensorTensorDictIterator lhs, @Const @ByRef TensorTensorDictIterator rhs);
  public boolean notEquals(TensorTensorDictIterator rhs) { return notEquals(this, rhs); }
}
