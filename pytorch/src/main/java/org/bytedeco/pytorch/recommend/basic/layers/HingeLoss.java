/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/LossFunc.scala
 * (HingeLoss, NCELoss, InBatchNCELoss, BPRLoss)
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;

/**
 * Hinge loss for pairwise learning.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class HingeLoss extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final float margin;
    private final Long numItems;

    public HingeLoss() {
        this(2.0f, null);
    }

    public HingeLoss(float margin) {
        this(margin, null);
    }

    public HingeLoss(float margin, Long numItems) {
        super("HingeLoss");
        this.margin = margin;
        this.numItems = numItems;
    }

    public Tensor forward(Tensor posScore, Tensor negScore, boolean inBatchNeg) {
        Tensor posFlat = posScore.view(-1);
        Tensor maxNeg = torch.max(negScore, -1).get0();
        Tensor marginTensor = torch.tensor(new float[]{margin});
        Tensor loss = torch.maximum(maxNeg.sub(posFlat).add(marginTensor), torch.zeros_like(maxNeg));

        if (numItems != null) {
            Tensor impostors = negScore.sub(posFlat.view(-1, 1)).add(new Scalar(margin)).gt(new Scalar(0.0f));
            Tensor rank = torch.mean(impostors.toType(ScalarType.Float), -1).mul(new Scalar(numItems.floatValue()));
            return torch.mean(loss.mul(torch.log(rank.sub(new Scalar(1.0f)))));
        }
        return torch.mean(loss);
    }

    public Tensor forward(Tensor posScore, Tensor negScore) {
        return forward(posScore, negScore, false);
    }
}
