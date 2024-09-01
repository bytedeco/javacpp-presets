package org.bytedeco.pytorch.cuda;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch_cuda.class)
public class OutOfMemoryObserver extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public OutOfMemoryObserver(Pointer p) {
        super(p);
    }

    protected OutOfMemoryObserver() {
        allocate();
    }

    private native void allocate();

    // std::function<void(int64_t device, int64_t allocated, int64_t device_total, int64_t device_free)
    public native void call(long device, long allocated, long device_total, long device_free);
}
