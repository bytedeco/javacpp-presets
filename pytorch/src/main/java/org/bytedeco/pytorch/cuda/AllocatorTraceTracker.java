package org.bytedeco.pytorch.cuda;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.cuda.TraceEntry;

@Properties(inherit = org.bytedeco.pytorch.presets.torch_cuda.class)
public class AllocatorTraceTracker extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public AllocatorTraceTracker(Pointer p) {
        super(p);
    }

    protected AllocatorTraceTracker() {
        allocate();
    }

    private native void allocate();

    // std::function<void(const TraceEntry&)>
    public native void call(@Const @ByRef TraceEntry e);
}
