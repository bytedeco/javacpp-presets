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


// virtual ivalue PyObjectHolder that hold a py::object, we make this virtual
// because the py::object and refcounting logic should happen in libtorch_python
// see concrete implementation in python_ivalue.h
@Name("c10::ivalue::PyObjectHolder") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PyObjectHolder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PyObjectHolder(Pointer p) { super(p); }

  public native @Cast("PyObject*") Pointer getPyObject();
  public native @ByVal InferredType tryToInferType();
  public native @ByVal IValue toIValue(@Const @ByRef Type.TypePtr type, @ByVal(nullValue = "std::optional<int32_t>(std::nullopt)") IntOptional N);
  public native @ByVal IValue toIValue(@Const @ByRef Type.TypePtr type);
  public native @StdString BytePointer toStr();
  public native @ByVal TensorVector extractTensors();
}
