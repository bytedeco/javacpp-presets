package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.GraphFunction;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GraphFunctionCreator extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public GraphFunctionCreator(Pointer p) {
        super(p);
    }

    protected GraphFunctionCreator() {
        allocate();
    }

    private native void allocate();

    public native void call(@ByRef GraphFunction f);
}
