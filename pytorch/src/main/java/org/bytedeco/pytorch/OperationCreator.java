package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.Const;


@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class OperationCreator extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public OperationCreator(Pointer p) {
        super(p);
    }

    protected OperationCreator() {
        allocate();
    }

    private native void allocate();

    public native @ByVal Operation call(@Const JitNode arg0);

}
