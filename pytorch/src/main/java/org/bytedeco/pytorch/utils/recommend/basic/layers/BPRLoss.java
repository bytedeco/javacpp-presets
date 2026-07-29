/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/LossFunc.scala (BPRLoss)
 *
 * Bayesian Personalized Ranking (BPR) Loss.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BPRLoss extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public BPRLoss() {
        super("BPRLoss");
    }

    public Tensor forward(Tensor posScore, Tensor negScore, boolean inBatchNeg) {
        Tensor posFlat = posScore.view(-1);
        Tensor diff;
        if (negScore.dim() == 1L) {
            diff = posFlat.sub(negScore);
        } else {
            diff = posFlat.view(-1, 1).sub(negScore);
        }
        // Use .neg() instead of unary minus — pure API (mirrors Scala)
        return torch.sigmoid(diff).log().mean().neg();
    }

    public Tensor forward(Tensor posScore, Tensor negScore) {
        return forward(posScore, negScore, false);
    }
}
