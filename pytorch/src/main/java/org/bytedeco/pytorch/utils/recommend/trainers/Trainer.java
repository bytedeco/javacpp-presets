/*
 * Unified trainer base for the recommend stack.
 *
 * Ported / redesigned from torch-rechub-scala: torchrec/trainers/*
 *
 * All concrete trainers (CTR / Match / MTL / custom) extend this class so callers
 * share one fit / evaluate / predict surface. Model-specific forward is pluggable
 * via {@link BatchForward} — Module subclasses already expose typed forward
 * overloads; adapters call those instead of giant match-case trees.
 */
package org.bytedeco.pytorch.utils.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.data.Batch;
import org.bytedeco.pytorch.utils.recommend.data.DataLoader;
import org.bytedeco.pytorch.utils.recommend.data.RecommendDataset;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Abstract base for recommend trainers (CRTP: {@code class Foo extends Trainer<Foo>}).
 *
 * <p>Typical usage:
 * <pre>{@code
 * CTRTrainer trainer = new CTRTrainer(model)
 *     .learningRate(1e-3f)
 *     .numEpochs(10)
 *     .device(DeviceSupport.backend());
 * trainer.fit(trainBatches, valBatches);
 * Map&lt;String, Float&gt; metrics = trainer.evaluate(valBatches);
 * }</pre>
 *
 * <p>Subclass hooks:
 * <ul>
 *   <li>{@link #computeTrainLoss(Batch)} — required, returns scalar loss for one batch</li>
 *   <li>{@link #predictBatch(Batch)} — optional logits / scores used by default evaluate</li>
 *   <li>{@link #evaluate(Iterable)} — override for task-specific metrics</li>
 *   <li>{@link #primaryMetricName()} — early-stop metric key (default {@code "AUC"})</li>
 * </ul>
 */
/**
 * @param <T> concrete trainer type (CRTP) so fluent setters keep the subclass type
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public abstract class Trainer<T extends Trainer<T>> {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    protected final Module model;
    protected Optimizer optimizer;
    protected String device;
    protected float learningRate;
    protected float weightDecay;
    protected int numEpochs;
    protected int earlyStopPatience;
    protected boolean verbose;
    protected Float gradientClip; // nullable — null disables clipping
    protected int trainStep;
    protected float bestMetric;
    protected int patienceCounter;
    protected boolean maximizeMetric; // true: higher is better (AUC); false: lower is better (loss)

    protected Trainer(Module model) {
        this.model = Objects.requireNonNull(model, "model");
        this.device = DeviceSupport.backend();
        this.learningRate = 1e-3f;
        this.weightDecay = 1e-6f;
        this.numEpochs = 10;
        this.earlyStopPatience = 500;
        this.verbose = true;
        this.gradientClip = null;
        this.trainStep = 0;
        this.bestMetric = Float.NEGATIVE_INFINITY;
        this.patienceCounter = 0;
        this.maximizeMetric = true;
        this.optimizer = null; // lazily created on first fit, or via withOptimizer
    }

    // ---- fluent config (CRTP) ------------------------------------------------

    @SuppressWarnings("unchecked")
    protected final T self() {
        return (T) this;
    }

    public T learningRate(float lr) {
        this.learningRate = lr;
        return self();
    }

    public T weightDecay(float wd) {
        this.weightDecay = wd;
        return self();
    }

    public T device(String device) {
        this.device = device != null ? device : DeviceSupport.backend();
        return self();
    }

    public T numEpochs(int epochs) {
        this.numEpochs = Math.max(1, epochs);
        return self();
    }

    public T earlyStopPatience(int patience) {
        this.earlyStopPatience = Math.max(1, patience);
        return self();
    }

    public T verbose(boolean verbose) {
        this.verbose = verbose;
        return self();
    }

    public T gradientClip(Float clip) {
        this.gradientClip = clip;
        return self();
    }

    public T maximizeMetric(boolean maximize) {
        this.maximizeMetric = maximize;
        if (maximize) {
            this.bestMetric = Float.NEGATIVE_INFINITY;
        } else {
            this.bestMetric = Float.POSITIVE_INFINITY;
        }
        return self();
    }

    /** Inject a custom optimizer (skips default Adam construction). */
    public T withOptimizer(Optimizer optimizer) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        return self();
    }

    // ---- accessors -----------------------------------------------------------

    public Module model() {
        return model;
    }

    public Optimizer optimizer() {
        ensureOptimizer();
        return optimizer;
    }

    public String device() {
        return device;
    }

    public int trainStep() {
        return trainStep;
    }

    public float bestMetric() {
        return bestMetric;
    }

    public void reset() {
        trainStep = 0;
        patienceCounter = 0;
        bestMetric = maximizeMetric ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
    }

    // ---- abstract / overridable hooks ----------------------------------------

    /**
     * Compute scalar training loss for one batch. Called under {@code model.train(true)}.
     * Subclasses must not call {@code optimizer.step()} themselves — the base loop does.
     */
    protected abstract Tensor computeTrainLoss(Batch batch);

    /**
     * Optional: raw prediction / logits for one batch (used by default evaluate / predict).
     * Return null to skip the batch. Default throws — override or provide a full evaluate.
     */
    protected Tensor predictBatch(Batch batch) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " must override predictBatch() or evaluate()");
    }

    /** Metric key used for early stopping (must appear in evaluate() result). */
    protected String primaryMetricName() {
        return "AUC";
    }

    // ---- public training API -------------------------------------------------

    public void fit(Iterable<Batch> trainLoader) {
        fit(trainLoader, null);
    }

    public void fit(RecommendDataset train, int batchSize) {
        fit(DataLoader.batches(train, batchSize, true, false, device), null);
    }

    public void fit(RecommendDataset train, RecommendDataset val, int batchSize) {
        fit(DataLoader.batches(train, batchSize, true, false, device),
                val != null ? DataLoader.batches(val, batchSize, false, false, device) : null);
    }

    /**
     * Full training loop with optional validation + early stopping.
     */
    public void fit(Iterable<Batch> trainLoader, Iterable<Batch> valLoader) {
        ensureOptimizer();
        model.train(true);
        patienceCounter = 0;

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            Map<String, Float> trainMetrics = trainEpoch(trainLoader);
            float avgLoss = getOrElse(trainMetrics, "loss", 0.0f);

            if (verbose) {
                System.out.printf("Epoch %d: train_loss=%.4f", epoch, avgLoss);
            }

            if (valLoader != null) {
                try {
                    model.eval();
                } catch (Throwable ignored) {
                    // some modules may not implement eval cleanly
                }
                Map<String, Float> valMetrics = evaluate(valLoader);
                String metricKey = primaryMetricName();
                float metric = valMetrics.containsKey(metricKey)
                        ? valMetrics.get(metricKey)
                        : firstMetric(valMetrics);

                if (verbose) {
                    System.out.printf(", val_%s=%.4f", metricKey, metric);
                    for (Map.Entry<String, Float> e : valMetrics.entrySet()) {
                        if (!e.getKey().equals(metricKey)) {
                            System.out.printf(" %s=%.4f", e.getKey(), e.getValue());
                        }
                    }
                }

                boolean improved = maximizeMetric
                        ? metric > bestMetric
                        : metric < bestMetric;
                if (improved) {
                    bestMetric = metric;
                    patienceCounter = 0;
                } else {
                    patienceCounter += 1;
                }
                model.train(true);
            }

            if (verbose) {
                System.out.println();
            }

            if (valLoader != null && patienceCounter >= earlyStopPatience) {
                if (verbose) {
                    System.out.println("Early stopping at epoch " + epoch);
                }
                return;
            }
        }
    }

    /**
     * One epoch of training. Returns aggregated metrics (at least {@code "loss"}).
     */
    public Map<String, Float> trainEpoch(Iterable<Batch> trainLoader) {
        ensureOptimizer();
        model.train(true);
        double totalLoss = 0.0;
        int numBatches = 0;

        for (Batch batch : trainLoader) {
            Float lossVal = trainStep(batch);
            if (lossVal != null) {
                totalLoss += lossVal;
                numBatches += 1;
            }
        }

        Map<String, Float> out = new LinkedHashMap<>();
        out.put("loss", numBatches > 0 ? (float) (totalLoss / numBatches) : 0.0f);
        out.put("num_batches", (float) numBatches);
        return out;
    }

    /**
     * Single optimization step. Returns finite loss value, or null if the step was skipped.
     */
    public Float trainStep(Batch batch) {
        ensureOptimizer();
        if (batch == null) {
            return null;
        }
        try {
            optimizer.zero_grad();
            Tensor loss = computeTrainLoss(batch);
            if (loss == null) {
                return null;
            }
            return safeBackwardAndStep(loss);
        } catch (Throwable t) {
            System.err.println("[WARN] trainStep exception: " + t.getMessage() + ", skipping");
            return null;
        }
    }

    /**
     * Default evaluation: collect sigmoid(predictBatch) vs labels and compute
     * AUC / LogLoss / Accuracy. Subclasses with richer metrics should override.
     */
    public Map<String, Float> evaluate(Iterable<Batch> dataLoader) {
        model.eval();
        List<Float> preds = new ArrayList<>();
        List<Float> labels = new ArrayList<>();

        for (Batch batch : dataLoader) {
            if (batch == null || batch.labels == null) {
                continue;
            }
            try {
                Tensor pred = predictBatch(batch);
                if (pred == null) {
                    continue;
                }
                Tensor prob = pred.sigmoid();
                appendHost(prob, preds, true);
                appendHost(batch.labels, labels, false);
            } catch (Throwable ignored) {
                // skip bad batch
            }
        }

        model.train(true);
        return binaryClassificationMetrics(preds, labels);
    }

    public Map<String, Float> evaluate(RecommendDataset dataset, int batchSize) {
        return evaluate(DataLoader.batches(dataset, batchSize, false, false, device));
    }

    /**
     * Collect predictions (probabilities after sigmoid) over a loader.
     */
    public float[] predict(Iterable<Batch> dataLoader) {
        model.eval();
        List<Float> preds = new ArrayList<>();
        for (Batch batch : dataLoader) {
            if (batch == null) {
                continue;
            }
            try {
                Tensor pred = predictBatch(batch);
                if (pred == null) {
                    continue;
                }
                appendHost(pred.sigmoid(), preds, true);
            } catch (Throwable ignored) {
                // skip
            }
        }
        model.train(true);
        float[] out = new float[preds.size()];
        for (int i = 0; i < preds.size(); i++) {
            out[i] = preds.get(i);
        }
        return out;
    }

    public float[] predict(RecommendDataset dataset, int batchSize) {
        return predict(DataLoader.batches(dataset, batchSize, false, false, device));
    }

    // ---- shared helpers (protected for subclasses) ---------------------------

    protected void ensureOptimizer() {
        if (optimizer == null) {
            AdamOptions opts = new AdamOptions(learningRate);
            if (weightDecay > 0f) {
                opts.weight_decay(weightDecay);
            }
            optimizer = new Adam(model.parameters(), opts);
        }
    }

    /**
     * Backward + clip + step only when loss is finite. Returns loss value or null.
     */
    protected Float safeBackwardAndStep(Tensor loss) {
        try {
            float v = (float) TensorHelpers.itemSafe(loss);
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                System.err.println("[WARN] non-finite loss=" + v + ", skipping step");
                return null;
            }
            loss.backward();
            if (gradientClip != null && gradientClip > 0f) {
                clipGradNorm(model, gradientClip);
            }
            optimizer.step();
            trainStep += 1;
            return v;
        } catch (Throwable t) {
            System.err.println("[WARN] backward/step failed: " + t.getMessage());
            return null;
        }
    }

    /** Clip total grad L2-norm via libtorch {@code clip_grad_norm_}. */
    protected static void clipGradNorm(Module module, float maxNorm) {
        if (module == null || maxNorm <= 0f) {
            return;
        }
        try {
            torch.clip_grad_norm_(module.parameters(), maxNorm);
        } catch (Throwable t) {
            // Fallback: per-parameter scale (Scala TrainLoop style)
            try {
                TensorVector params = module.parameters();
                long n = params.size();
                for (long i = 0; i < n; i++) {
                    Tensor p = params.get(i);
                    if (p == null) continue;
                    Tensor g = p.grad();
                    if (g == null) continue;
                    Tensor norm = g.norm();
                    float nVal = (float) TensorHelpers.itemSafe(norm);
                    if (nVal > maxNorm && nVal > 0f) {
                        g.div_(new Scalar(nVal / maxNorm));
                    }
                }
            } catch (Throwable ignored) {
                // leave grads unclipped
            }
        }
    }

    /**
     * Normalize logits / labels for BCEWithLogits: squeeze trailing 1-dims, ensure
     * matching batch shapes, float targets.
     */
    protected static Tensor bceWithLogits(Tensor pred, Tensor labels,
                                         org.bytedeco.pytorch.utils.recommend.basic.losses.Losses.BCEWithLogitsLoss lossFn) {
        Tensor p = pred;
        try {
            if (p.dim() >= 2L && p.size(1) > 1L) {
                p = p.view(p.size(0), -1L).mean(1L);
            } else if (p.dim() == 2L && p.size(1) == 1L) {
                p = p.squeeze(1L);
            }
        } catch (Throwable ignored) {
            // keep original
        }

        Tensor t = labels;
        try {
            if (t.dim() == 2L && t.size(1) == 1L) {
                t = t.squeeze(1L);
            }
        } catch (Throwable ignored) {
            // keep
        }

        long batchSize = p.size(0);
        Tensor pForLoss = p.dim() == 1L ? p.view(batchSize, 1L) : p;
        Tensor tForLoss = t.dim() == 1L
                ? t.view(batchSize, 1L).toType(ScalarType.Float)
                : t.toType(ScalarType.Float);
        return lossFn.apply(pForLoss, tForLoss);
    }

    /**
     * Copy tensor values to a host float list.
     * @param clamp01 when true, sanitize NaN/Inf and clamp to [0,1] (for probabilities)
     */
    protected static void appendHost(Tensor tensor, List<Float> out, boolean clamp01) {
        if (tensor == null) {
            return;
        }
        Tensor host = tensor.squeeze().to(ScalarType.Float).contiguous().cpu();
        float[] arr = TensorHelpers.toFloatArray(host);
        for (float v : arr) {
            if (clamp01) {
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    out.add(0.5f);
                } else if (v < 0f) {
                    out.add(0f);
                } else if (v > 1f) {
                    out.add(1f);
                } else {
                    out.add(v);
                }
            } else {
                out.add(v);
            }
        }
    }

    protected static void appendHost(Tensor tensor, List<Float> out) {
        appendHost(tensor, out, true);
    }

    protected static Map<String, Float> binaryClassificationMetrics(List<Float> predList, List<Float> labelList) {
        if (predList.isEmpty() || labelList.isEmpty()) {
            Map<String, Float> empty = new LinkedHashMap<>();
            empty.put("AUC", 0.0f);
            empty.put("LogLoss", 0.0f);
            empty.put("Accuracy", 0.0f);
            empty.put("Hit@10", 0.0f);
            empty.put("NDCG@10", 0.0f);
            empty.put("MRR", 0.0f);
            return empty;
        }
        int n = Math.min(predList.size(), labelList.size());
        float[] preds = new float[n];
        float[] labels = new float[n];
        for (int i = 0; i < n; i++) {
            preds[i] = predList.get(i);
            labels[i] = labelList.get(i);
        }

        org.bytedeco.pytorch.utils.recommend.basic.metrics.AUC auc =
                new org.bytedeco.pytorch.utils.recommend.basic.metrics.AUC();
        org.bytedeco.pytorch.utils.recommend.basic.metrics.LogLoss logloss =
                new org.bytedeco.pytorch.utils.recommend.basic.metrics.LogLoss();
        org.bytedeco.pytorch.utils.recommend.basic.metrics.Accuracy acc =
                new org.bytedeco.pytorch.utils.recommend.basic.metrics.Accuracy();
        org.bytedeco.pytorch.utils.recommend.basic.metrics.HitRate hit =
                new org.bytedeco.pytorch.utils.recommend.basic.metrics.HitRate(10);
        org.bytedeco.pytorch.utils.recommend.basic.metrics.NDCG ndcg =
                new org.bytedeco.pytorch.utils.recommend.basic.metrics.NDCG(10);
        org.bytedeco.pytorch.utils.recommend.basic.metrics.MRR mrr =
                new org.bytedeco.pytorch.utils.recommend.basic.metrics.MRR();

        auc.update(preds, labels);
        logloss.update(preds, labels);
        acc.update(preds, labels);
        hit.update(preds, labels);
        ndcg.update(preds, labels);
        mrr.update(preds, labels);

        Map<String, Float> out = new LinkedHashMap<>();
        out.put("AUC", auc.compute());
        out.put("LogLoss", logloss.compute());
        out.put("Accuracy", acc.compute());
        out.put("Hit@10", hit.compute());
        out.put("NDCG@10", ndcg.compute());
        out.put("MRR", mrr.compute());
        return out;
    }

    protected static float firstMetric(Map<String, Float> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return 0.0f;
        }
        return metrics.values().iterator().next();
    }

    /** Tiny helper: Map.getOrDefault-style with primitive float default. */
    protected static float getOrElse(Map<String, Float> m, String k, float d) {
        Float v = m.get(k);
        return v != null ? v : d;
    }

    /**
     * Functional adapter: {@code Batch -> Tensor} prediction / logits.
     * Prefer typed Module.forward overloads inside implementations.
     */
    @FunctionalInterface
    public interface BatchForward {
        /**
         * @return logits / scores, or null to skip the batch
         */
        Tensor forward(Batch batch);
    }

    /**
     * Functional adapter: {@code Batch -> scalar loss}.
     */
    @FunctionalInterface
    public interface BatchLoss {
        Tensor loss(Batch batch);
    }

    /** Build default Adam for a model. */
    public static Optimizer defaultAdam(Module model, float lr, float weightDecay) {
        AdamOptions opts = new AdamOptions(lr);
        if (weightDecay > 0f) {
            opts.weight_decay(weightDecay);
        }
        return new Adam(model.parameters(), opts);
    }

    /** Move batch to trainer device if needed. */
    protected Batch onDevice(Batch batch) {
        if (batch == null) {
            return null;
        }
        if (device == null || "cpu".equals(device)) {
            return batch;
        }
        try {
            return batch.to(device);
        } catch (Throwable t) {
            return batch;
        }
    }

    protected Device torchDevice() {
        return new Device(device != null ? device : "cpu");
    }

    protected static Tensor arangeLong(long n, String device) {
        Tensor t = torch.arange(new Scalar(0), new Scalar((double) n), new Scalar(1),
                new org.bytedeco.pytorch.TensorOptions()
                        .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Long)));
        if (device != null && !"cpu".equals(device)) {
            t = t.to(new Device(device), ScalarType.Long);
        }
        return t;
    }

    protected static Tensor positionsFor(Tensor tokens, String device) {
        long batch = tokens.size(0);
        long seq = tokens.size(1);
        Tensor pos = TensorHelpers.arange(0, (int) seq).toType(ScalarType.Long)
                .unsqueeze(0)
                .repeat(new long[]{batch, 1L});
        if (device != null && !"cpu".equals(device)) {
            pos = pos.to(new Device(device), ScalarType.Long);
        }
        return pos;
    }
}
