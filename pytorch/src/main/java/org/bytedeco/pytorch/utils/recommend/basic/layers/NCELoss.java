/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/LossFunc.scala (NCELoss)
 *
 * Noise Contrastive Estimation (NCE) loss for recommender systems.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NCELoss extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final float temperature;
    private final long ignoreIndex;
    private final String reduction;

    public NCELoss() {
        this(1.0f, 0L, "mean");
    }

    public NCELoss(float temperature, long ignoreIndex, String reduction) {
        super("NCELoss");
        this.temperature = temperature;
        this.ignoreIndex = ignoreIndex;
        this.reduction = reduction;
    }

    @Override
    public Tensor forward(Tensor logits, Tensor targets) {
        Tensor scaledLogits = logits.div(new Scalar(temperature));
        Tensor logProbs = torch.log_softmax(scaledLogits, -1);
        int batchSize = (int) targets.size(0);

        float lossSum = 0.0f;
        int lossCount = 0;
        for (int i = 0; i < batchSize; i++) {
            long targetIdx = targets.select(0, i).item().toLong();
            float lp = logProbs.select(0, i).select(0, targetIdx).item().toFloat();
            if (targetIdx != ignoreIndex) {
                lossSum -= lp;
                lossCount += 1;
            }
        }

        switch (reduction) {
            case "mean":
                return torch.tensor(new float[]{lossCount == 0 ? 0f : lossSum / lossCount});
            case "sum":
                return torch.tensor(new float[]{lossSum});
            default: {
                float[] arr = new float[batchSize];
                for (int i = 0; i < batchSize; i++) {
                    long t = targets.select(0, i).item().toLong();
                    float lp = logProbs.select(0, i).select(0, t).item().toFloat();
                    arr[i] = (t == ignoreIndex) ? 0f : -lp;
                }
                return torch.tensor(arr);
            }
        }
    }
}
