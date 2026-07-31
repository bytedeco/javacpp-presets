/*
 * Ported from torch-rechub-scala: torchrec/trainers/MTLTrainer.scala
 *
 * Trainer for multi-task models (MMOE, SharedBottom, PLE, AITM, ESMM, OMoE,
 * SingleTaskModel, MetaHeac). Extends {@link Trainer}.
 *
 * Models return either:
 * <ul>
 *   <li>{@code Tensor} concatenated along dim=1 (most MTL models)</li>
 *   <li>{@code Map<String, Tensor>} per task name (MetaHeac)</li>
 * </ul>
 */
package org.bytedeco.pytorch.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.basic.losses.Losses;
import org.bytedeco.pytorch.recommend.basic.metrics.AUC;
import org.bytedeco.pytorch.recommend.basic.metrics.Accuracy;
import org.bytedeco.pytorch.recommend.basic.metrics.LogLoss;
import org.bytedeco.pytorch.recommend.data.Batch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MTLTrainer extends Trainer<MTLTrainer> {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<String> taskNames;
    private final Map<String, Float> taskWeights;
    private final Losses.BCEWithLogitsLoss bceLoss;

    public MTLTrainer(Module model, List<String> taskNames) {
        this(model, taskNames, null);
    }

    public MTLTrainer(Module model, List<String> taskNames, Map<String, Float> taskWeights) {
        super(model);
        Objects.requireNonNull(taskNames, "taskNames");
        if (taskNames.isEmpty()) {
            throw new IllegalArgumentException("taskNames must not be empty");
        }
        this.taskNames = new ArrayList<>(taskNames);
        Map<String, Float> weights = new LinkedHashMap<>();
        if (taskWeights != null) {
            weights.putAll(taskWeights);
        }
        for (String name : this.taskNames) {
            weights.putIfAbsent(name, 1.0f);
        }
        this.taskWeights = weights;
        this.bceLoss = new Losses.BCEWithLogitsLoss();
        maximizeMetric(true);
    }

    public List<String> taskNames() {
        return Collections.unmodifiableList(taskNames);
    }

    public Map<String, Float> taskWeights() {
        return Collections.unmodifiableMap(taskWeights);
    }

    public MTLTrainer taskWeight(String task, float weight) {
        taskWeights.put(task, weight);
        return this;
    }

    @Override
    protected String primaryMetricName() {
        // early-stop on first task's AUC
        return taskNames.get(0) + "_AUC";
    }

    @Override
    protected Tensor computeTrainLoss(Batch batch) {
        if (batch == null) {
            return null;
        }
        Map<String, Tensor> sparse = batch.sparseFeatures;
        Map<String, Tensor> taskLabels = batch.taskLabels;
        if (sparse == null || sparse.isEmpty() || taskLabels == null || taskLabels.isEmpty()) {
            return null;
        }

        Object raw;
        try {
            raw = ModelForwards.mtlRaw(model, batch);
        } catch (IllegalArgumentException ex) {
            System.err.println("[WARN] " + ex.getMessage());
            return null;
        }

        Tensor totalWeightedLoss = null;
        float totalWeight = 0.0f;
        int nTask = taskNames.size();

        for (int i = 0; i < nTask; i++) {
            String taskName = taskNames.get(i);
            Tensor pred = ModelForwards.mtlTaskLogit(raw, taskName, i);
            Tensor label = taskLabels.get(taskName);
            if (pred == null || label == null) {
                continue;
            }

            long actualBatch = label.size(0);
            Tensor pred2D = pred.dim() == 1L ? pred.reshape(actualBatch, 1L) : pred;
            Tensor label2D = label.view(actualBatch, 1L).toType(ScalarType.Float);
            Tensor taskLoss = bceLoss.apply(pred2D, label2D);

            float weight = taskWeights.getOrDefault(taskName, 1.0f);
            Tensor weighted = taskLoss.mul(new Scalar((double) weight));
            totalWeightedLoss = totalWeightedLoss == null
                    ? weighted
                    : totalWeightedLoss.add(weighted);
            totalWeight += weight;
        }

        if (totalWeightedLoss == null || totalWeight <= 0f) {
            return null;
        }
        return totalWeightedLoss.div(new Scalar((double) totalWeight));
    }

    @Override
    protected Tensor predictBatch(Batch batch) {
        // Return concatenated / first-task logits for generic callers
        if (batch == null) {
            return null;
        }
        try {
            Object raw = ModelForwards.mtlRaw(model, batch);
            if (raw instanceof Tensor) {
                return (Tensor) raw;
            }
            return ModelForwards.mtlTaskLogit(raw, taskNames.get(0), 0);
        } catch (Throwable t) {
            return null;
        }
    }

    @Override
    public Map<String, Float> evaluate(Iterable<Batch> dataLoader) {
        model.eval();
        int nTask = taskNames.size();

        Map<String, AUC> aucs = new LinkedHashMap<>();
        Map<String, LogLoss> loglosses = new LinkedHashMap<>();
        Map<String, Accuracy> accs = new LinkedHashMap<>();
        for (String name : taskNames) {
            aucs.put(name, new AUC());
            loglosses.put(name, new LogLoss());
            accs.put(name, new Accuracy());
        }

        for (Batch batch : dataLoader) {
            if (batch == null) continue;
            Map<String, Tensor> sparse = batch.sparseFeatures;
            Map<String, Tensor> taskLabels = batch.taskLabels;
            if (sparse == null || sparse.isEmpty() || taskLabels == null || taskLabels.isEmpty()) {
                continue;
            }

            Object raw;
            try {
                raw = ModelForwards.mtlRaw(model, batch);
            } catch (Throwable t) {
                continue;
            }

            for (int i = 0; i < nTask; i++) {
                String taskName = taskNames.get(i);
                Tensor pred = ModelForwards.mtlTaskLogit(raw, taskName, i);
                Tensor label = taskLabels.get(taskName);
                if (pred == null || label == null) {
                    continue;
                }
                try {
                    Tensor predProb = pred.sigmoid();
                    float[] predArr = TensorHelpers.toFloatArray(
                            predProb.squeeze().to(ScalarType.Float).contiguous().cpu());
                    float[] labelArr = TensorHelpers.toFloatArray(
                            label.squeeze().to(ScalarType.Float).contiguous().cpu());
                    if (predArr.length == 0 || labelArr.length == 0) {
                        continue;
                    }
                    int n = Math.min(predArr.length, labelArr.length);
                    if (n < predArr.length || n < labelArr.length) {
                        float[] p2 = new float[n];
                        float[] l2 = new float[n];
                        System.arraycopy(predArr, 0, p2, 0, n);
                        System.arraycopy(labelArr, 0, l2, 0, n);
                        predArr = p2;
                        labelArr = l2;
                    }
                    aucs.get(taskName).update(predArr, labelArr);
                    loglosses.get(taskName).update(predArr, labelArr);
                    accs.get(taskName).update(predArr, labelArr);
                } catch (Throwable ignored) {
                    // skip task/batch
                }
            }
        }

        model.train(true);

        Map<String, Float> out = new LinkedHashMap<>();
        for (String name : taskNames) {
            out.put(name + "_AUC", aucs.get(name).compute());
            out.put(name + "_LogLoss", loglosses.get(name).compute());
            out.put(name + "_Accuracy", accs.get(name).compute());
        }
        return out;
    }
}
