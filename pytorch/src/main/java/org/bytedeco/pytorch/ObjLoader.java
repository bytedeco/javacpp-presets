package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;

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
    // Without @Cast, the generated JavaCPP_org_bytedeco_pytorch_functions_ObjLoader::ptr would return an ivalue::Object
    public native @ByVal @Cast({"", "c10::intrusive_ptr<c10::ivalue::Object>"}) @IntrusivePtr Obj call(@Const @ByRef StrongTypePtr stp, @ByVal IValue iv);
}
