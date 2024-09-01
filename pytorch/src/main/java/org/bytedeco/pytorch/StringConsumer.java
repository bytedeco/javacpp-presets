package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Const;
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

    // std::function<void(const std::string&)>
    public native void call(@Const @StdString BytePointer s);
}
