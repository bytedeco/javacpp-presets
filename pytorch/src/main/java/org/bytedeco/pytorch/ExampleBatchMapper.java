package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.utils.spacy.Example;
import org.bytedeco.pytorch.data.ExampleVector;

/**
 * Adapter for
 * {@code std::function<Example(std::vector<Example>)>}
 * used by {@code torch::data::transforms::BatchLambda}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ExampleBatchMapper extends FunctionPointer {
    static {
        Loader.load();
    }

    public ExampleBatchMapper(Pointer p) {
        super(p);
    }

    protected ExampleBatchMapper() {
        allocate();
    }

    private native void allocate();

    public native @ByVal Example call(@ByVal ExampleVector batch);
}
