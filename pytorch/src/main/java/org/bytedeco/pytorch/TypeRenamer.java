package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.ClassType;
import org.bytedeco.pytorch.QualifiedName;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TypeRenamer extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TypeRenamer(Pointer p) {
        super(p);
    }

    protected TypeRenamer() {
        allocate();
    }

    private native void allocate();

    // std::function<c10::QualifiedName(const c10::ClassTypePtr&)>
    public native @ByVal QualifiedName call(@SharedPtr @Cast({"", "std::shared_ptr<c10::ClassType>"}) ClassType classType);
}
