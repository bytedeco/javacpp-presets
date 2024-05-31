package org.bytedeco.shakapackager.functions;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Properties(inherit = org.bytedeco.shakapackager.presets.packager.class)
public class Write_BufferParamsCallback_BytePointer_Pointer_long extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Write_BufferParamsCallback_BytePointer_Pointer_long(Pointer p) { super(p); }
    protected Write_BufferParamsCallback_BytePointer_Pointer_long() { allocate(); }
    private native void allocate();
    public native  @Cast("int64_t") long call(@Const @StdString BytePointer arg0,@Const Pointer arg1,@Cast("uint64_t") long arg2);
}
