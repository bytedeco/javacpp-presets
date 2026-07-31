/*
 * Full-lifecycle feature + model pipeline:
 *   raw DataFrame
 *     → DataFrame feature engineering (impute / scale / encode)
 *     → FeatureIngest (warehouse register + offline put)
 *     → FeatureMaterializeJob (online store)
 *     → online serve smoke
 *     → FeatureTrainingExport (PIT join training table)
 *     → optional DeepFM / CTR train steps
 *     → quality / freshness report
 *
 * One façade that wires dataframe.feature.* with utils.feature.* end-to-end.
 */
package org.bytedeco.pytorch.feature.pipeline;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.feature.bridge.DataFrameBridge;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.basic.losses.Losses;
import org.bytedeco.pytorch.feature.FeaturePlatform;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.lifecycle.FeatureDriftMonitor;
import org.bytedeco.pytorch.feature.lifecycle.FeatureQualityReport;
import org.bytedeco.pytorch.feature.lifecycle.FeatureValidator;
import org.bytedeco.pytorch.feature.lifecycle.FreshnessMonitor;
import org.bytedeco.pytorch.feature.materialize.MaterializationResult;
import org.bytedeco.pytorch.feature.offline.TrainingDataset;
import org.bytedeco.pytorch.feature.serving.FeatureRequest;
import org.bytedeco.pytorch.feature.serving.FeatureResponse;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.data.Batch;
import org.bytedeco.pytorch.recommend.models.ranking.DeepFM;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * End-to-end lifecycle orchestrator for feature warehouse + model train.
 *
 * <pre>{@code
 * LifecyclePipeline.Result r = LifecyclePipeline.on(fp)
 *     .project("demo")
 *     .view("user_feats")
 *     .entities("user_id")
 *     .featureService("rank_svc")
 *     .raw(rawDf)
 *     .featureEngineering(fe -> fe.impute("mean", "age").standardScale("age", "score").build())
 *     .labelColumn("label")
 *     .trainDeepFM(true)
 *     .trainSteps(20)
 *     .run();
 * }</pre>
 */
public final class LifecyclePipeline {

    public static final class StageTiming {
        public final String stage;
        public final double elapsedMs;
        public final boolean ok;
        public final String detail;

        public StageTiming(String stage, double elapsedMs, boolean ok, String detail) {
            this.stage = stage;
            this.elapsedMs = elapsedMs;
            this.ok = ok;
            this.detail = detail != null ? detail : "";
        }

        @Override
        public String toString() {
            return String.format("[%s] %-18s %8.2f ms  %s",
                    ok ? "PASS" : "FAIL", stage, elapsedMs, detail);
        }
    }

    public static final class Result {
        public final List<StageTiming> stages;
        public final FeatureIngest.Result ingest;
        public final MaterializationResult materialize;
        public final FeatureResponse onlineSmoke;
        public final FeatureTrainingExport.Result trainingExport;
        public final List<Feature> recommendFeatures;
        public final Double finalTrainLoss;
        public final FeatureQualityReport quality;
        public final long totalNanos;
        public final boolean success;
        public final String message;

        Result(List<StageTiming> stages,
               FeatureIngest.Result ingest,
               MaterializationResult materialize,
               FeatureResponse onlineSmoke,
               FeatureTrainingExport.Result trainingExport,
               List<Feature> recommendFeatures,
               Double finalTrainLoss,
               FeatureQualityReport quality,
               long totalNanos,
               boolean success,
               String message) {
            this.stages = List.copyOf(stages);
            this.ingest = ingest;
            this.materialize = materialize;
            this.onlineSmoke = onlineSmoke;
            this.trainingExport = trainingExport;
            this.recommendFeatures = recommendFeatures != null ? List.copyOf(recommendFeatures) : List.of();
            this.finalTrainLoss = finalTrainLoss;
            this.quality = quality;
            this.totalNanos = totalNanos;
            this.success = success;
            this.message = message != null ? message : "";
        }

        public double totalMs() {
            return totalNanos / 1_000_000.0;
        }

        public TrainingDataset dataset() {
            return trainingExport != null ? trainingExport.dataset : null;
        }

        public DataFrame trainingDataFrame() {
            return trainingExport != null ? trainingExport.dataFrame : null;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("LifecyclePipeline.Result{success=").append(success)
                    .append(", totalMs=").append(String.format("%.1f", totalMs()))
                    .append(", stages=").append(stages.size())
                    .append("}\n");
            for (StageTiming s : stages) {
                sb.append("  ").append(s).append('\n');
            }
            if (finalTrainLoss != null) {
                sb.append("  finalTrainLoss=").append(finalTrainLoss).append('\n');
            }
            if (message != null && !message.isEmpty()) {
                sb.append("  message=").append(message).append('\n');
            }
            return sb.toString();
        }
    }

    private final FeaturePlatform platform;
    private String project = "default";
    private String viewName = "lifecycle_view";
    private String featureServiceName = "lifecycle_svc";
    private final List<String> entityCols = new ArrayList<>();
    private String timestampColumn = "event_timestamp";
    private String labelColumn = "label";
    private Duration ttl = Duration.ofDays(7);
    private DataFrame raw;
    private DataFrame entityDf; // optional explicit entity df for PIT; default derived from ingested
    private UnaryOperator<DataFrame> feTransform;
    private boolean trainDeepFM = false;
    private int trainSteps = 10;
    private int batchSize = 32;
    private int embedDim = 8;
    private float learningRate = 1e-2f;
    private boolean materializeOnline = true;
    private boolean runOnlineSmoke = true;
    private boolean runTrainingExport = true;
    private boolean runQuality = true;
    private FreshnessMonitor freshnessMonitor = new FreshnessMonitor();

    private LifecyclePipeline(FeaturePlatform platform) {
        this.platform = Objects.requireNonNull(platform, "platform");
    }

    public static LifecyclePipeline on(FeaturePlatform platform) {
        return new LifecyclePipeline(platform);
    }

    public LifecyclePipeline project(String project) {
        this.project = project != null ? project : "default";
        return this;
    }

    public LifecyclePipeline view(String viewName) {
        this.viewName = viewName != null ? viewName : "lifecycle_view";
        return this;
    }

    public LifecyclePipeline featureService(String name) {
        this.featureServiceName = name != null ? name : "lifecycle_svc";
        return this;
    }

    public LifecyclePipeline entities(String... cols) {
        if (cols != null) {
            for (String c : cols) if (c != null) entityCols.add(c);
        }
        return this;
    }

    public LifecyclePipeline timestampColumn(String col) {
        this.timestampColumn = col != null ? col : "event_timestamp";
        return this;
    }

    public LifecyclePipeline labelColumn(String col) {
        this.labelColumn = col != null ? col : "label";
        return this;
    }

    public LifecyclePipeline ttl(Duration ttl) {
        this.ttl = ttl != null ? ttl : Duration.ofDays(7);
        return this;
    }

    public LifecyclePipeline raw(DataFrame df) {
        this.raw = df;
        return this;
    }

    public LifecyclePipeline entityDataFrame(DataFrame df) {
        this.entityDf = df;
        return this;
    }

    /** Raw DataFrame transform (already-engineered frame or custom logic). */
    public LifecyclePipeline transform(UnaryOperator<DataFrame> transform) {
        this.feTransform = transform;
        return this;
    }

    /**
     * DataFrame feature-engineering steps via {@code df.feature()} façade.
     * Prefer this over {@link #transform} for impute/scale/encode chains.
     */
    public LifecyclePipeline featureEngineering(FeatureIngest.FeatureEngOp op) {
        Objects.requireNonNull(op, "op");
        this.feTransform = df -> {
            try {
                return op.apply(df.feature());
            } catch (Exception e) {
                throw new IllegalStateException("FE failed: " + e.getMessage(), e);
            }
        };
        return this;
    }

    public LifecyclePipeline trainDeepFM(boolean enable) {
        this.trainDeepFM = enable;
        return this;
    }

    public LifecyclePipeline trainSteps(int steps) {
        this.trainSteps = Math.max(1, steps);
        return this;
    }

    public LifecyclePipeline batchSize(int batchSize) {
        this.batchSize = Math.max(1, batchSize);
        return this;
    }

    public LifecyclePipeline embedDim(int embedDim) {
        this.embedDim = Math.max(2, embedDim);
        return this;
    }

    public LifecyclePipeline learningRate(float lr) {
        this.learningRate = lr;
        return this;
    }

    public LifecyclePipeline materializeOnline(boolean v) {
        this.materializeOnline = v;
        return this;
    }

    public LifecyclePipeline runOnlineSmoke(boolean v) {
        this.runOnlineSmoke = v;
        return this;
    }

    public LifecyclePipeline runTrainingExport(boolean v) {
        this.runTrainingExport = v;
        return this;
    }

    public LifecyclePipeline runQuality(boolean v) {
        this.runQuality = v;
        return this;
    }

    public Result run() {
        long tAll = System.nanoTime();
        List<StageTiming> stages = new ArrayList<>();
        FeatureIngest.Result ingestResult = null;
        MaterializationResult matResult = null;
        FeatureResponse onlineResp = null;
        FeatureTrainingExport.Result exportResult = null;
        List<Feature> recFeats = new ArrayList<>();
        Double finalLoss = null;
        FeatureQualityReport quality = null;

        try {
            if (raw == null) {
                throw new IllegalStateException("raw DataFrame required");
            }
            if (entityCols.isEmpty()) {
                entityCols.add("user_id");
            }

            // ── 1. Feature engineering ──────────────────────────────────────
            long t0 = System.nanoTime();
            DataFrame engineered = raw;
            if (feTransform != null) {
                engineered = feTransform.apply(raw);
            }
            stages.add(new StageTiming("feature_eng",
                    (System.nanoTime() - t0) / 1e6,
                    engineered != null && engineered.rowCount() > 0,
                    "rows=" + (engineered != null ? engineered.rowCount() : 0)
                            + " cols=" + (engineered != null ? engineered.columnCount() : 0)));

            // ── 2. Ingest to feature warehouse ──────────────────────────────
            t0 = System.nanoTime();
            // label stays on entity/event table — never register as a FeatureView column
            FeatureIngest ingest = FeatureIngest.into(platform)
                    .project(project)
                    .view(viewName)
                    .entities(entityCols)
                    .timestampColumn(timestampColumn)
                    .ttl(ttl)
                    .autoRegister(true)
                    .validate(true)
                    .replace(true)
                    .exclude(labelColumn)
                    .from(engineered);
            ingestResult = ingest.run();
            stages.add(new StageTiming("ingest",
                    (System.nanoTime() - t0) / 1e6,
                    ingestResult.ok(),
                    ingestResult.toString()));
            if (!ingestResult.ok()) {
                return fail(stages, ingestResult, null, null, null, recFeats, null, null, tAll,
                        "ingest validation failed");
            }

            // Register feature service bundling this view
            t0 = System.nanoTime();
            if (platform.registry().getFeatureService(project, featureServiceName).isEmpty()) {
                platform.featureService(FeatureService.builder(featureServiceName)
                        .project(project)
                        .view(viewName)
                        .description("lifecycle auto service")
                        .build());
            }
            stages.add(new StageTiming("register_svc",
                    (System.nanoTime() - t0) / 1e6, true, featureServiceName));

            // ── 3. Materialize online ───────────────────────────────────────
            if (materializeOnline) {
                t0 = System.nanoTime();
                freshnessMonitor.setSlo(project, viewName, ttl.isZero() ? Duration.ofDays(1) : ttl);
                matResult = FeatureMaterializeJob.on(platform)
                        .fromIngest(ingestResult)
                        .observeFreshness(freshnessMonitor)
                        .run();
                stages.add(new StageTiming("materialize",
                        (System.nanoTime() - t0) / 1e6,
                        matResult.success(),
                        matResult.toString()));
                if (!matResult.success()) {
                    return fail(stages, ingestResult, matResult, null, null, recFeats, null, null, tAll,
                            "materialize failed");
                }
            }

            // ── 4. Online serve smoke ───────────────────────────────────────
            if (runOnlineSmoke && materializeOnline) {
                t0 = System.nanoTime();
                Map<String, Object> ent = new LinkedHashMap<>();
                // pick first row entity keys
                if (engineered.rowCount() > 0) {
                    for (String ec : entityCols) {
                        if (engineered.hasColumn(ec)) {
                            ent.put(ec, engineered.get(0, ec));
                        }
                    }
                }
                onlineResp = platform.getOnlineFeatures(FeatureRequest.builder()
                        .project(project)
                        .featureService(featureServiceName)
                        .entities(ent)
                        .build());
                stages.add(new StageTiming("online_serve",
                        (System.nanoTime() - t0) / 1e6,
                        onlineResp.success(),
                        onlineResp.toString()));
            }

            // ── 5. Training export (PIT) ────────────────────────────────────
            if (runTrainingExport) {
                t0 = System.nanoTime();
                DataFrame entities = entityDf;
                if (entities == null) {
                    // Build entity df from engineered: entities + ts + label
                    entities = buildEntityFrame(engineered);
                }
                exportResult = FeatureTrainingExport.from(platform)
                        .project(project)
                        .featureService(featureServiceName)
                        .entityDataFrame(entities)
                        .labelColumn(labelColumn)
                        .eventTimestampColumn(timestampColumn)
                        .prefixWithViewName(true)
                        .run();
                stages.add(new StageTiming("train_export",
                        (System.nanoTime() - t0) / 1e6,
                        exportResult.size() > 0,
                        exportResult.toString()));
            }

            // ── 6. Optional DeepFM train ────────────────────────────────────
            if (trainDeepFM && exportResult != null && exportResult.size() > 0) {
                t0 = System.nanoTime();
                recFeats = buildSparseFeatures(exportResult);
                if (recFeats.isEmpty()) {
                    stages.add(new StageTiming("train_deepfm",
                            (System.nanoTime() - t0) / 1e6, false, "no sparse features"));
                } else {
                    finalLoss = trainDeepFmSteps(exportResult, recFeats);
                    boolean ok = finalLoss != null && Double.isFinite(finalLoss);
                    stages.add(new StageTiming("train_deepfm",
                            (System.nanoTime() - t0) / 1e6, ok,
                            "steps=" + trainSteps + " loss=" + finalLoss
                                    + " feats=" + recFeats.size()));
                }
            }

            // ── 7. Quality report ───────────────────────────────────────────
            if (runQuality && ingestResult != null) {
                t0 = System.nanoTime();
                FeatureView view = ingestResult.view;
                List<Map<String, Object>> offlineRows =
                        platform.offline().readAll(project, viewName);
                FeatureValidator.Report val = new FeatureValidator().validate(view, offlineRows);
                FreshnessMonitor.Status fresh = freshnessMonitor.check(project, viewName);
                // drift: first half vs second half of a numeric column if any
                List<FeatureDriftMonitor.PsiResult> drifts = new ArrayList<>();
                if (exportResult != null && !exportResult.featureColumns.isEmpty()) {
                    String col = exportResult.featureColumns.get(0);
                    List<Map<String, Object>> all = exportResult.dataset.rows();
                    int mid = Math.max(1, all.size() / 2);
                    FeatureDriftMonitor mon = new FeatureDriftMonitor(8, 0.25);
                    drifts = mon.psiColumns(all.subList(0, mid),
                            all.subList(mid, all.size()), List.of(col));
                }
                quality = FeatureQualityReport.builder(viewName)
                        .project(project)
                        .validation(val)
                        .freshness(fresh)
                        .drift(drifts)
                        .meta("pipeline", "lifecycle")
                        .build();
                stages.add(new StageTiming("quality",
                        (System.nanoTime() - t0) / 1e6,
                        quality.healthy() || val.ok,
                        quality.toString()));
            }

            boolean success = stages.stream().allMatch(s -> s.ok);
            return new Result(stages, ingestResult, matResult, onlineResp, exportResult,
                    recFeats, finalLoss, quality, System.nanoTime() - tAll, success, "ok");
        } catch (Exception e) {
            stages.add(new StageTiming("error", 0, false, e.toString()));
            return fail(stages, ingestResult, matResult, onlineResp, exportResult,
                    recFeats, finalLoss, quality, tAll, e.getMessage());
        }
    }

    private DataFrame buildEntityFrame(DataFrame engineered) {
        List<String> keep = new ArrayList<>(entityCols);
        if (engineered.hasColumn(timestampColumn) && !keep.contains(timestampColumn)) {
            keep.add(timestampColumn);
        }
        if (engineered.hasColumn(labelColumn) && !keep.contains(labelColumn)) {
            keep.add(labelColumn);
        }
        return DataFrameBridge
                .selectColumns(engineered, keep);
    }

    private List<Feature> buildSparseFeatures(FeatureTrainingExport.Result export) {
        List<Feature> out = new ArrayList<>();
        // Prefer entity id columns + integer-like feature columns as sparse
        for (String c : export.featureColumns) {
            String simple = c.contains("__") ? c.substring(c.lastIndexOf("__") + 2) : c;
            // skip pure floats by sampling
            boolean looksFloat = false;
            for (int i = 0; i < Math.min(8, export.dataset.size()); i++) {
                Object v = export.dataset.row(i).get(c);
                if (v instanceof Double || v instanceof Float) {
                    double d = ((Number) v).doubleValue();
                    if (d != Math.rint(d)) {
                        looksFloat = true;
                        break;
                    }
                }
            }
            if (looksFloat) {
                out.add(Features.dense(simple, 1));
            } else {
                out.add(Features.sparse(simple, 10_000L, embedDim));
            }
        }
        // Also add entity keys as sparse if present in rows
        for (String ek : export.entityKeys) {
            boolean exists = false;
            for (Feature f : out) {
                if (f.name().equals(ek)) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                out.add(Features.sparse(ek, 10_000L, embedDim));
            }
        }
        return out;
    }

    private Double trainDeepFmSteps(FeatureTrainingExport.Result export, List<Feature> feats) {
        List<Feature> sparse = new ArrayList<>();
        for (Feature f : feats) {
            if (f instanceof SparseFeature) sparse.add(f);
        }
        if (sparse.isEmpty()) {
            for (String ek : export.entityKeys) {
                sparse.add(Features.sparse(ek, 10_000L, embedDim));
            }
        }
        if (sparse.isEmpty()) return null;

        String device = "cpu";
        try {
            device = DeviceSupport.backend();
        } catch (Throwable ignored) {
            device = "cpu";
        }
        DeepFM model = new DeepFM(sparse, sparse, embedDim, new long[]{64L, 32L}, 0.0f, device);
        List<Batch> batches = export.toBatches(sparse, batchSize);
        if (batches.isEmpty()) return null;

        AdamOptions opt = new AdamOptions(learningRate);
        Optimizer optim = new Adam(model.parameters(), opt);
        var bce = new Losses.BCEWithLogitsLoss();
        double lastLoss = Double.NaN;
        model.train(true);
        for (int step = 0; step < trainSteps; step++) {
            Batch batch = batches.get(step % batches.size());
            optim.zero_grad();
            try {
                Tensor logits = model.forward(batch.sparseFeatures,
                        batch.denseFeatures != null ? batch.denseFeatures : Map.of());
                Tensor y = batch.labels;
                if (y == null) {
                    y = TensorHelpers.zeros(logits.size(0));
                }
                Tensor loss = bce.apply(logits, y);
                loss.backward();
                optim.step();
                lastLoss = loss.item().toFloat();
            } catch (Throwable t) {
                try {
                    Tensor logits = model.forward(batch.sparseFeatures);
                    Tensor y = batch.labels != null ? batch.labels
                            : TensorHelpers.zeros(logits.size(0));
                    Tensor loss = bce.apply(logits, y);
                    loss.backward();
                    optim.step();
                    lastLoss = loss.item().toFloat();
                } catch (Throwable t2) {
                    return Double.NaN;
                }
            }
        }
        return lastLoss;
    }

    private static Result fail(List<StageTiming> stages,
                               FeatureIngest.Result ingest,
                               MaterializationResult mat,
                               FeatureResponse online,
                               FeatureTrainingExport.Result exp,
                               List<Feature> feats,
                               Double loss,
                               FeatureQualityReport quality,
                               long tAll,
                               String msg) {
        return new Result(stages, ingest, mat, online, exp, feats, loss, quality,
                System.nanoTime() - tAll, false, msg != null ? msg : "failed");
    }
}
