/*
 * Ported from torch-rechub-scala: torchrec/trainers/CTRTrainer.scala
 *
 * Trainer for CTR / ranking models. Extends {@link Trainer}.
 * Model-specific forward is delegated to {@link ModelForwards#ctr(Module)}.
 */
package org.bytedeco.pytorch.utils.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.basic.losses.Losses;
import org.bytedeco.pytorch.utils.recommend.data.Batch;

import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CTRTrainer extends Trainer<CTRTrainer> {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Losses.BCEWithLogitsLoss bceLoss;
    private BatchForward forward;

    public CTRTrainer(Module model) {
        super(model);
        this.bceLoss = new Losses.BCEWithLogitsLoss();
        this.forward = ModelForwards.ctr(model);
        // early-stop on AUC (maximize)
        maximizeMetric(true);
    }

    /**
     * Override automatic dispatch with a custom BatchForward
     * (e.g. when the model is not yet registered in {@link ModelForwards}).
     */
    public CTRTrainer withForward(BatchForward forward) {
        this.forward = forward != null ? forward : ModelForwards.ctr(model);
        return this;
    }

    public BatchForward forward() {
        return forward;
    }

    @Override
    protected String primaryMetricName() {
        return "AUC";
    }

    @Override
    protected Tensor computeTrainLoss(Batch batch) {
        if (batch == null || batch.labels == null) {
            return null;
        }
        Tensor logits = forward.forward(batch);
        if (logits == null) {
            return null;
        }
        return bceWithLogits(logits, batch.labels, bceLoss);
    }

    @Override
    protected Tensor predictBatch(Batch batch) {
        if (batch == null) {
            return null;
        }
        return forward.forward(batch);
    }

    @Override
    public Map<String, Float> evaluate(Iterable<Batch> dataLoader) {
        // Use base binary metrics over sigmoid(logits) vs labels
        return super.evaluate(dataLoader);
    }
}
