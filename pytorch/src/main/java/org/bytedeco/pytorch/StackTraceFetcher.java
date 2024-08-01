package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.SharedPtr;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StackTraceFetcher extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public StackTraceFetcher(Pointer p) {
        super(p);
    }

    protected StackTraceFetcher() {
        allocate();
    }

    private native void allocate();

    // std::function<std::shared_ptr<const c10::LazyValue<std::string> >()>
    public native @Cast({"", "std::shared_ptr<c10::LazyValue<std::string>>"}) @SharedPtr Backtrace call();
}
