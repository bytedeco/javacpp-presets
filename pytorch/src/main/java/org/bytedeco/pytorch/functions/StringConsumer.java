package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.StdString;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StringConsumer extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public StringConsumer(Pointer p) {
        super(p);
    }

    protected StringConsumer() {
        allocate();
    }

    private native void allocate();

    public native void call(@Cast({"", "const std::string&"}) @StdString String s);
}
