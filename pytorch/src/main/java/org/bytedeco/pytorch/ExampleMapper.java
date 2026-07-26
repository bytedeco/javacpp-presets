package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.utils.spacy.Example;

/**
 * Adapter for {@code std::function<torch::data::Example<>(torch::data::Example<>)>}
 * used by {@code torch::data::transforms::Lambda}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ExampleMapper extends FunctionPointer {
    static {
        Loader.load();
    }

    public ExampleMapper(Pointer p) {
        super(p);
    }

    protected ExampleMapper() {
        allocate();
    }

    private native void allocate();

    public native @ByVal Example call(@ByVal Example example);
}
