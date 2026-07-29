/*
 * Ported from torch-rechub-scala: torchrec/trainers/TrainLoop.scala
 *
 * Lightweight generic train / eval loop that works with any Module + BatchForward.
 * Prefer the full {@link Trainer} subclasses for CTR / Match / MTL; use this when
 * you only need step / epoch plumbing with a custom loss lambda.
 */
package org.bytedeco.pytorch.utils.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.data.Batch;
import org.bytedeco.pytorch.utils.recommend.trainers.Trainer.BatchForward;
import org.bytedeco.pytorch.utils.recommend.trainers.Trainer.BatchLoss;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TrainLoop {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Module model;
    private final Optimizer optimizer;
    private final String device;
    private final Float gradientClip;
    private int trainStep;

    public TrainLoop(Module model, Optimizer optimizer) {
        this(model, optimizer, DeviceSupport.backend(), null);
    }

    public TrainLoop(Module model, Optimizer optimizer, String device, Float gradientClip) {
        this.model = Objects.requireNonNull(model, "model");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        this.device = device != null ? device : DeviceSupport.backend();
        this.gradientClip = gradientClip;
        this.trainStep = 0;
    }

    public Module model() {
        return model;
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public int getTrainStep() {
        return trainStep;
    }

    public void reset() {
        trainStep = 0;
    }

    /**
     * Single training step with an arbitrary loss function over a batch.
     *
     * @return map with at least {@code "loss"}
     */
    public Map<String, Float> step(Batch batch, BatchLoss lossFn) {
        optimizer.zero_grad();
        Tensor loss = lossFn.loss(batch);
        float v = (float) TensorHelpers.itemSafe(loss);
        if (Float.isNaN(v) || Float.isInfinite(v)) {
            Map<String, Float> m = new LinkedHashMap<>();
            m.put("loss", v);
            m.put("skipped", 1.0f);
            return m;
        }
        loss.backward();
        if (gradientClip != null && gradientClip > 0f) {
            Trainer.clipGradNorm(model, gradientClip);
        }
        optimizer.step();
        trainStep += 1;
        Map<String, Float> m = new LinkedHashMap<>();
        m.put("loss", v);
        return m;
    }

    /**
     * Train one epoch. {@code lossFn} receives each batch and returns a scalar loss.
     */
    public Map<String, Float> trainEpoch(Iterable<Batch> dataLoader, BatchLoss lossFn) {
        model.train(true);
        double total = 0.0;
        int n = 0;
        for (Batch batch : dataLoader) {
            Map<String, Float> stepMetrics = step(batch, lossFn);
            if (!stepMetrics.containsKey("skipped")) {
                total += stepMetrics.getOrDefault("loss", 0.0f);
                n += 1;
            }
        }
        Map<String, Float> out = new LinkedHashMap<>();
        out.put("loss", n > 0 ? (float) (total / n) : 0.0f);
        out.put("num_batches", (float) n);
        return out;
    }

    /**
     * Evaluate a loader: {@code evalFn} returns per-batch metric map; values are averaged.
     */
    public Map<String, Float> evaluate(Iterable<Batch> dataLoader,
                                       Function<Batch, Map<String, Float>> evalFn) {
        model.eval();
        Map<String, Double> acc = new LinkedHashMap<>();
        int n = 0;
        for (Batch batch : dataLoader) {
            Map<String, Float> batchMetrics = evalFn.apply(batch);
            if (batchMetrics == null || batchMetrics.isEmpty()) {
                continue;
            }
            for (Map.Entry<String, Float> e : batchMetrics.entrySet()) {
                acc.merge(e.getKey(), e.getValue().doubleValue(), Double::sum);
            }
            n += 1;
        }
        model.train(true);
        Map<String, Float> out = new LinkedHashMap<>();
        if (n == 0) {
            return out;
        }
        for (Map.Entry<String, Double> e : acc.entrySet()) {
            out.put(e.getKey(), (float) (e.getValue() / n));
        }
        return out;
    }

    /**
     * Convenience: build a TrainLoop around a Module with default Adam.
     */
    public static TrainLoop adam(Module model, float lr) {
        return new TrainLoop(model, Trainer.defaultAdam(model, lr, 0f));
    }

    public static TrainLoop adam(Module model, float lr, float weightDecay, String device, Float clip) {
        return new TrainLoop(model, Trainer.defaultAdam(model, lr, weightDecay), device, clip);
    }

    /**
     * Loss from a BatchForward that returns logits, combined with labels via a provided reducer.
     */
    public static BatchLoss fromLogits(BatchForward forward,
                                       Function<Tensor /*logits*/, Function<Tensor /*labels*/, Tensor>> lossBuilder) {
        return batch -> {
            if (batch.labels == null) {
                throw new IllegalArgumentException("batch has no labels");
            }
            Tensor logits = forward.forward(batch);
            return lossBuilder.apply(logits).apply(batch.labels);
        };
    }
}
