package org.bytedeco.pytorch.cuda;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch_cuda.class)
public class StreamFilter extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public StreamFilter(Pointer p) {
        super(p);
    }

    protected StreamFilter() {
        allocate();
    }

    private native void allocate();

    // std::function<bool(cudaStream_t)>
    public native boolean call(Pointer stream);
}
