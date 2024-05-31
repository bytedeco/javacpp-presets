package org.bytedeco.shakapackager.functions;


import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@NoOffset @Name("std::function<int64_t(const std::string&,const void*,uint64_t)>") @Properties(inherit = org.bytedeco.shakapackager.presets.packager.class)
public class Write_BufferParamsCallback extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Write_BufferParamsCallback(Pointer p) { super(p); }
    public Write_BufferParamsCallback(Write_BufferParamsCallback_BytePointer_Pointer_long value) { this(); put(value); }
    public Write_BufferParamsCallback()       { allocate();  }
    private native void allocate();
    public native  @Name("operator =") @ByRef Write_BufferParamsCallback put(@ByRef Write_BufferParamsCallback x);

    public native @Name("operator =") @ByRef Write_BufferParamsCallback put(@ByRef Write_BufferParamsCallback_BytePointer_Pointer_long value);
    public native @Name("operator ()") @Cast("int64_t") long call(@StdString BytePointer arg0,@Const Pointer arg1,@Cast("uint64_t") long arg2);
}