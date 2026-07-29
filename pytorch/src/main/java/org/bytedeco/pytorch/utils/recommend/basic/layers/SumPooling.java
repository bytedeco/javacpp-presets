/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/SumPooling.scala
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * Sum pooling over sequence embeddings.
 *
 * <p>Shape
 * <ul>
 *   <li>Input x: {@code (B, L, D)}</li>
 *   <li>mask: {@code (B, 1, L)}</li>
 *   <li>Output: {@code (B, D)}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SumPooling extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public SumPooling() {
        super("SumPooling");
    }

    public Tensor forward(Tensor x, Tensor mask) {
        if (mask == null) {
            return torch.sum(x, 1L);
        }
        return torch.bmm(mask, x).squeeze(1L);
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }
}
