// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.onnx;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.onnx.global.onnx.*;

@Name("std::unordered_map<std::string,std::pair<int,int> >") @Properties(inherit = org.bytedeco.onnx.presets.onnx.class)
public class StringIntIntPairMap extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringIntIntPairMap(Pointer p) { super(p); }
    public StringIntIntPairMap()       { allocate();  }
    private native void allocate();
    public native @Name("operator =") @ByRef StringIntIntPairMap put(@ByRef StringIntIntPairMap x);

    public boolean empty() { return size() == 0; }
    public native long size();

    @Index(function = "at") public native int first(@StdString BytePointer i); public native StringIntIntPairMap first(@StdString BytePointer i, int first);
    @Index(function = "at") public native int second(@StdString BytePointer i);  public native StringIntIntPairMap second(@StdString BytePointer i, int second);
}

