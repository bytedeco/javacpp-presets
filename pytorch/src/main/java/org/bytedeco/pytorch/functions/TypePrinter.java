package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.StringOptional;
import org.bytedeco.pytorch.Type;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TypePrinter extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TypePrinter(Pointer p) {
        super(p);
    }

    protected TypePrinter() {
        allocate();
    }

    private native void allocate();

    // std::function<c10::optional<std::string>(const c10::Type&)>
    public native @ByVal StringOptional call(@Const @ByRef Type type);
}
