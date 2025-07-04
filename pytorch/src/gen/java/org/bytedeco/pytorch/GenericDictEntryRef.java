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


/**
 * A reference to an entry in the Dict.
 * Use the {@code key()} and {@code value()} methods to read the element.
 */
@Name("c10::impl::DictEntryRef<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GenericDictEntryRef extends Pointer {
    static { Loader.load(); }

  public GenericDictEntryRef(@ByVal @Cast("c10::detail::DictImpl::dict_map_type::iterator*") Pointer iterator) { super((Pointer)null); allocate(iterator); }
  private native void allocate(@ByVal @Cast("c10::detail::DictImpl::dict_map_type::iterator*") Pointer iterator);

  public native @ByVal IValue key();

  public native @ByVal IValue value();
}
