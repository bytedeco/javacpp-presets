/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/AveragePooling.scala
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * Mean pooling over sequence embeddings.
 *
 * <p>Shape
 * <ul>
 *   <li>Input x: {@code (B, L, D)}</li>
 *   <li>mask: {@code (B, 1, L)}</li>
 *   <li>Output: {@code (B, D)}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AveragePooling extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public AveragePooling() {
        super("AveragePooling");
    }

    public Tensor forward(Tensor x, Tensor mask) {
        if (mask == null) {
            return torch.mean(x, 1L);
        }
        Tensor sumPoolingMatrix = torch.bmm(mask, x).squeeze(1L);
        Tensor nonPaddingLength = mask.sum(1L);
        return sumPoolingMatrix.div(nonPaddingLength.add(new Scalar(1e-16f)));
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }
}
