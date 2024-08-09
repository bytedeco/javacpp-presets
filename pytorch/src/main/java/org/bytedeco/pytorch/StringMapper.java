package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.StdString;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StringMapper extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public StringMapper(Pointer p) {
        super(p);
    }

    protected StringMapper() {
        allocate();
    }

    private native void allocate();

    // std::function<std::string(const std::string&)>
    public native  @StdString @Cast({"", "char*"}) BytePointer call(@Const @StdString BytePointer s);
}
