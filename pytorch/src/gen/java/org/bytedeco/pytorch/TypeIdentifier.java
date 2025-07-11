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
 * A type id is a unique id for a given C++ type.
 * You need to register your types using CAFFE_KNOWN_TYPE(MyType) to be able to
 * use TypeIdentifier with custom types. This is for example used to store the
 * dtype of tensors.
 */
@Namespace("caffe2") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TypeIdentifier extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TypeIdentifier(Pointer p) { super(p); }

  private static native @Namespace @Cast("std::ostream*") @ByRef @Name("operator <<") Pointer shiftLeft(@Cast("std::ostream*") @ByRef Pointer stream, @ByVal TypeIdentifier typeId);
  public Pointer shiftLeft(Pointer stream) { return shiftLeft(stream, this); }
  private static native @Namespace @Cast("const bool") @Name("operator <") boolean lessThan(@ByVal TypeIdentifier lhs, @ByVal TypeIdentifier rhs);
  public boolean lessThan(TypeIdentifier rhs) { return lessThan(this, rhs); }

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their ids are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */

  public static native @Const @ByVal TypeIdentifier uninitialized();
}
