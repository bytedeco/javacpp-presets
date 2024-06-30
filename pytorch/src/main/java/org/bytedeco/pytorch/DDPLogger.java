package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.DDPLoggingData;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DDPLogger extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public DDPLogger(Pointer p) {
        super(p);
    }

    protected DDPLogger() {
        allocate();
    }

    private native void allocate();

    public native void call(@Const @ByRef DDPLoggingData d);
}
