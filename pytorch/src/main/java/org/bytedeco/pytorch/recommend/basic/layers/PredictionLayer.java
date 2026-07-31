/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/PredictionLayer.scala
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * Prediction layer.
 *
 * <p>Parameters
 * <ul>
 *   <li>taskType — {@code classification} or {@code regression}.
 *       Classification applies sigmoid to logits; regression returns logits.</li>
 * </ul>
 *
 * <p>Shape: Input {@code (B, *)}, Output {@code (B, *)} with sigmoid if classification.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PredictionLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String taskType;

    public PredictionLayer() {
        this("classification");
    }

    public PredictionLayer(String taskType) {
        super("PredictionLayer");
        if (!"classification".equals(taskType) && !"regression".equals(taskType)) {
            throw new IllegalArgumentException("taskType must be classification or regression");
        }
        this.taskType = taskType;
    }

    public String taskType() {
        return taskType;
    }

    @Override
    public Tensor forward(Tensor x) {
        if ("classification".equals(taskType)) {
            return torch.sigmoid(x);
        }
        return x;
    }
}
