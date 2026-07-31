/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/LeakyReLUImpl.scala
 *
 * Note: stock nn.modules.LeakyReLUImpl also exists; this is the torchrec custom
 * wrapper used by some models that import torchrec.basic.layers.LeakyReLUImpl.
 * Class name RecLeakyReLU avoids clash with org.bytedeco.pytorch.nn.modules.LeakyReLUImpl.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/** Leaky ReLU (torchrec custom Module wrapper). */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RecLeakyReLU extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public RecLeakyReLU() {
        super("LeakyReLUImpl");
    }

    @Override
    public Tensor forward(Tensor x) {
        return torch.leaky_relu(x, new Scalar(0.01f));
    }
}
