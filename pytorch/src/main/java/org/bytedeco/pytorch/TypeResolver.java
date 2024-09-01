package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TypeResolver extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TypeResolver(Pointer p) {
        super(p);
    }

    protected TypeResolver() {
        allocate();
    }

    private native void allocate();

    // std::function<c10::StrongTypePtr(const c10::QualifiedName&)>
    public native @ByVal StrongTypePtr call(@Const @ByRef QualifiedName name);
}
