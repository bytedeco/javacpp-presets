package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.Type.TypePtr;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TypeParser extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TypeParser(Pointer p) {
        super(p);
    }

    protected TypeParser() {
        allocate();
    }

    private native void allocate();

    // std::function<c10::TypePtr(const std::string&)>
    public native @ByVal TypePtr call(@Const @StdString BytePointer s);
}
