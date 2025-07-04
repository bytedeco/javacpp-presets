// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.sentencepiece;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.sentencepiece.global.sentencepiece.*;

@Name("std::vector<std::pair<std::vector<int>,float> >") @Properties(inherit = org.bytedeco.sentencepiece.presets.sentencepiece.class)
public class IntVectorFloatPairVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntVectorFloatPairVector(Pointer p) { super(p); }
    public IntVectorFloatPairVector(IntVector[] firstValue, float[] secondValue) { this(Math.min(firstValue.length, secondValue.length)); put(firstValue, secondValue); }
    public IntVectorFloatPairVector()       { allocate();  }
    public IntVectorFloatPairVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator =") @ByRef IntVectorFloatPairVector put(@ByRef IntVectorFloatPairVector x);

    public boolean empty() { return size() == 0; }
    public native long size();
    public void clear() { resize(0); }
    public native void resize(@Cast("size_t") long n);

    @Index(function = "at") public native @ByRef IntVector first(@Cast("size_t") long i); public native IntVectorFloatPairVector first(@Cast("size_t") long i, IntVector first);
    @Index(function = "at") public native float second(@Cast("size_t") long i);  public native IntVectorFloatPairVector second(@Cast("size_t") long i, float second);

    public IntVectorFloatPairVector put(IntVector[] firstValue, float[] secondValue) {
        for (int i = 0; i < firstValue.length && i < secondValue.length; i++) {
            first(i, firstValue[i]);
            second(i, secondValue[i]);
        }
        return this;
    }
}

