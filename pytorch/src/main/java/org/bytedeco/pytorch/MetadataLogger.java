package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.StdString;
import org.bytedeco.pytorch.DDPLoggingData;
import org.bytedeco.pytorch.StringStringMap;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MetadataLogger extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public MetadataLogger(Pointer p) {
        super(p);
    }

    protected MetadataLogger() {
        allocate();
    }

    private native void allocate();

    // std::function<void(const std::string&,const std::map<std::string,std::string>&)>
    public native void call(@Const @StdString BytePointer s, @Const @ByRef StringStringMap map);
}
