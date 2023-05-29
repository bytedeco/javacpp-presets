// Targeted by JavaCPP version 1.5.9-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.functions.*;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.pytorch.global.torch.*;

@Name("std::vector<std::pair<std::string,torch::Tensor> >") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StringTensorVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringTensorVector(Pointer p) { super(p); }
    public StringTensorVector(BytePointer[] firstValue, Tensor[] secondValue) { this(Math.min(firstValue.length, secondValue.length)); put(firstValue, secondValue); }
    public StringTensorVector(String[] firstValue, Tensor[] secondValue) { this(Math.min(firstValue.length, secondValue.length)); put(firstValue, secondValue); }
    public StringTensorVector()       { allocate();  }
    public StringTensorVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator =") @ByRef StringTensorVector put(@ByRef StringTensorVector x);

    public boolean empty() { return size() == 0; }
    public native long size();
    public void clear() { resize(0); }
    public native void resize(@Cast("size_t") long n);

    @Index(function = "at") public native @StdString BytePointer first(@Cast("size_t") long i); public native StringTensorVector first(@Cast("size_t") long i, BytePointer first);
    @Index(function = "at") public native @ByRef Tensor second(@Cast("size_t") long i);  public native StringTensorVector second(@Cast("size_t") long i, Tensor second);
    @MemberSetter @Index(function = "at") public native StringTensorVector first(@Cast("size_t") long i, @StdString String first);

    public StringTensorVector put(BytePointer[] firstValue, Tensor[] secondValue) {
        for (int i = 0; i < firstValue.length && i < secondValue.length; i++) {
            first(i, firstValue[i]);
            second(i, secondValue[i]);
        }
        return this;
    }

    public StringTensorVector put(String[] firstValue, Tensor[] secondValue) {
        for (int i = 0; i < firstValue.length && i < secondValue.length; i++) {
            first(i, firstValue[i]);
            second(i, secondValue[i]);
        }
        return this;
    }
}
