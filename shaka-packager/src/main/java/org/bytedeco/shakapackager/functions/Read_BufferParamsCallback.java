package org.bytedeco.shakapackager.functions;


import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@NoOffset @Name("std::function<int64_t(const std::string&,void*,uint64_t)>") @Properties(inherit = org.bytedeco.shakapackager.presets.packager.class)
public class Read_BufferParamsCallback extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Read_BufferParamsCallback(Pointer p) { super(p); }
    public Read_BufferParamsCallback(Read_BufferParamsCallback_BytePointer_Pointer_long value) { this(); put(value); }
    public Read_BufferParamsCallback()       { allocate();  }
    private native void allocate();
    public native @Name("operator =") @ByRef Read_BufferParamsCallback put(@ByRef Read_BufferParamsCallback x);

    public native @Name("operator =") @ByRef Read_BufferParamsCallback put(@ByRef Read_BufferParamsCallback_BytePointer_Pointer_long value);
    public native @Name("operator ()") @Cast("int64_t") long call(@StdString BytePointer arg0,Pointer arg1,@Cast("uint64_t") long arg2);
}