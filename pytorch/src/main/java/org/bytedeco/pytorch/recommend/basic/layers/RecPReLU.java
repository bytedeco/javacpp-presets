/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/PReLUImpl.scala
 *
 * Note: stock nn.modules.PReLUImpl also exists; this is the torchrec custom
 * wrapper. Class name RecPReLU avoids clash with org.bytedeco.pytorch.nn.modules.PReLUImpl.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;

/** PReLU activation (torchrec custom Module wrapper). */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RecPReLU extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private Tensor weight;

    public RecPReLU() {
        super("PReLUImpl");
    }

    @Override
    public Tensor forward(Tensor x) {
        if (weight == null || !weight.device().equals(x.device())) {
            if (weight != null) {
                weight.close();
            }
            weight = torch.zeros(new long[]{1L},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(x.device(), ScalarType.Float);
        }
        return torch.prelu(x, weight);
    }
}
