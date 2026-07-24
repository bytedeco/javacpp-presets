package org.bytedeco.pytorch;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;

/**
 * Adapter for {@code std::function<torch::Tensor(torch::Tensor)>}
 * used by {@code torch::data::transforms::TensorLambda}.
 * Distinct from {@link TensorMapper} which takes {@code const Tensor&}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorValueMapper extends FunctionPointer {
    static {
        Loader.load();
    }

    public TensorValueMapper(Pointer p) {
        super(p);
    }

    protected TensorValueMapper() {
        allocate();
    }

    private native void allocate();

    public native @ByVal Tensor call(@ByVal Tensor t);
}
