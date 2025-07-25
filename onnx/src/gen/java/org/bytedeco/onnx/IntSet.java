// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.onnx;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.onnx.global.onnx.*;

@Name("std::unordered_set<int>") @Properties(inherit = org.bytedeco.onnx.presets.onnx.class)
public class IntSet extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IntSet(Pointer p) { super(p); }
    public IntSet()       { allocate();  }
    private native void allocate();
    public native @Name("operator =") @ByRef IntSet put(@ByRef IntSet x);

    public boolean empty() { return size() == 0; }
    public native long size();

    public int front() { try (Iterator it = begin()) { return it.get(); } }
    public native void insert(int value);
    public native void erase(int value);
    public native @ByVal Iterator begin();
    public native @ByVal Iterator end();
    @NoOffset @Name("iterator") public static class Iterator extends Pointer {
        public Iterator(Pointer p) { super(p); }
        public Iterator() { }

        public native @Name("operator ++") @ByRef Iterator increment();
        public native @Name("operator ==") boolean equals(@ByRef Iterator it);
        public native @Name("operator *") int get();
    }
}

