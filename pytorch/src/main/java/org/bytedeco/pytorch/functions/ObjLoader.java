package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;

import org.bytedeco.pytorch.ObjPtr;
import org.bytedeco.pytorch.StrongTypePtr;
import org.bytedeco.pytorch.IValue;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ObjLoader extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public ObjLoader(Pointer p) {
        super(p);
    }

    protected ObjLoader() {
        allocate();
    }

    private native void allocate();

    // std::function<c10::intrusive_ptr<c10::ivalue::Object>(const at::StrongTypePtr&, IValue)>
    public native @ByVal ObjPtr call(@Const @ByRef StrongTypePtr stp, @ByVal IValue iv);
}
