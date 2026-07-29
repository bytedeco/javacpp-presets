/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/ConcatPooling.scala
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

/**
 * Keep original sequence embedding shape.
 *
 * <p>Shape
 * <ul>
 *   <li>Input: {@code (B, L, D)}</li>
 *   <li>Output: {@code (B, L, D)}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ConcatPooling extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public ConcatPooling() {
        super("ConcatPooling");
    }

    public Tensor forward(Tensor x, Tensor mask) {
        return x;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x;
    }
}
