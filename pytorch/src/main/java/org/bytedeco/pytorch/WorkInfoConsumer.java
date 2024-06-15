package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.SharedPtr;
import org.bytedeco.pytorch.WorkInfo;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class WorkInfoConsumer extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public WorkInfoConsumer(Pointer p) {
        super(p);
    }

    protected WorkInfoConsumer() {
        allocate();
    }

    private native void allocate();

    // std::function<void(std::shared_ptr<c10d::WorkInfo>)>
    public native void call(@SharedPtr @Cast({"", "std::shared_ptr<c10d::WorkInfo>"}) WorkInfo wi);
}