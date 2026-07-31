/*
 * Feature training export — entity DataFrame + FeatureService → PIT join →
 * TrainingDataset + DataFrame (+ optional recommend Batch list for model fit).
 *
 * Closes the loop: offline warehouse → point-in-time correct training table
 * for DeepFM / DIN / industry models (Feast get_historical_features).
 */
package org.bytedeco.pytorch.feature.pipeline;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.feature.FeaturePlatform;
import org.bytedeco.pytorch.feature.bridge.DataFrameBridge;
import org.bytedeco.pytorch.feature.bridge.RecommendFeatureBridge;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.offline.PointInTimeJoin;
import org.bytedeco.pytorch.feature.offline.TrainingDataset;
import org.bytedeco.pytorch.feature.serving.FeatureProvider;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.basic.features.DenseFeature;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.data.Batch;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Export point-in-time training features as DataFrame / TrainingDataset / Batches.
 *
 * <pre>{@code
 * FeatureTrainingExport.Result exp = FeatureTrainingExport.from(fp)
 *     .project("recsys")
 *     .featureService("shortvideo_rank")
 *     .entityDataFrame(entityDf)   // user_id, item_id, event_timestamp, label
 *     .labelColumn("label")
 *     .run();
 * DataFrame trainDf = exp.dataFrame();
 * List&lt;Batch&gt; batches = exp.toBatches(features, 256);
 * }</pre>
 */
public final class FeatureTrainingExport {

    public static final class Result {
        public final TrainingDataset dataset;
        public final DataFrame dataFrame;
        public final List<String> entityKeys;
        public final List<String> featureColumns;
        public final String labelColumn;
        public final String featureService;
        public final long elapsedNanos;
        public final PointInTimeJoin.JoinStats joinStats;

        Result(TrainingDataset dataset, DataFrame dataFrame, List<String> entityKeys,
               List<String> featureColumns, String labelColumn, String featureService,
               long elapsedNanos) {
            this.dataset = dataset;
            this.dataFrame = dataFrame;
            this.entityKeys = entityKeys != null ? List.copyOf(entityKeys) : List.of();
            this.featureColumns = featureColumns != null ? List.copyOf(featureColumns) : List.of();
            this.labelColumn = labelColumn != null ? labelColumn : "";
            this.featureService = featureService != null ? featureService : "";
            this.elapsedNanos = elapsedNanos;
            this.joinStats = dataset != null ? dataset.joinStats() : null;
        }

        public int size() {
            return dataset != null ? dataset.size() : 0;
        }

        public double elapsedMs() {
            return elapsedNanos / 1_000_000.0;
        }

        public double[] labels() {
            return dataset != null ? dataset.labels() : new double[0];
        }

        public double[][] denseMatrix() {
            return dataset != null ? dataset.denseMatrix() : new double[0][0];
        }

        /**
         * Convert joined rows into recommend {@link Batch} mini-batches for CTRTrainer etc.
         *
         * @param featureDefs recommend Feature list (Sparse/Dense/Sequence) aligned by name
         * @param batchSize   mini-batch size
         */
        public List<Batch> toBatches(List<? extends Feature> featureDefs, int batchSize) {
            Objects.requireNonNull(featureDefs, "featureDefs");
            int bs = Math.max(1, batchSize);
            List<Map<String, Object>> rows = dataset.rows();
            List<Batch> out = new ArrayList<>();
            for (int start = 0; start < rows.size(); start += bs) {
                int end = Math.min(start + bs, rows.size());
                out.add(rowsToBatch(rows.subList(start, end), featureDefs, labelColumn));
            }
            return out;
        }

        @Override
        public String toString() {
            return "TrainingExport{n=" + size()
                    + ", features=" + featureColumns.size()
                    + ", label=" + labelColumn
                    + ", ms=" + String.format("%.2f", elapsedMs())
                    + (joinStats != null ? ", " + joinStats : "")
                    + "}";
        }
    }

    private final FeaturePlatform platform;
    private String project = "default";
    private String featureService;
    private final List<FeatureView> views = new ArrayList<>();
    private DataFrame entityDf;
    private List<Map<String, Object>> entityRows;
    private String labelColumn = "label";
    private String eventTsColumn = PointInTimeJoin.DEFAULT_EVENT_TS;
    private boolean prefixWithViewName = true;

    private FeatureTrainingExport(FeaturePlatform platform) {
        this.platform = Objects.requireNonNull(platform, "platform");
    }

    public static FeatureTrainingExport from(FeaturePlatform platform) {
        return new FeatureTrainingExport(platform);
    }

    public FeatureTrainingExport project(String project) {
        this.project = project != null ? project : "default";
        return this;
    }

    public FeatureTrainingExport featureService(String featureService) {
        this.featureService = featureService;
        return this;
    }

    public FeatureTrainingExport views(FeatureView... vs) {
        if (vs != null) {
            for (FeatureView v : vs) if (v != null) views.add(v);
        }
        return this;
    }

    public FeatureTrainingExport entityDataFrame(DataFrame df) {
        this.entityDf = df;
        return this;
    }

    public FeatureTrainingExport entityRows(List<Map<String, Object>> rows) {
        this.entityRows = rows;
        return this;
    }

    public FeatureTrainingExport labelColumn(String labelColumn) {
        this.labelColumn = labelColumn != null ? labelColumn : "label";
        return this;
    }

    public FeatureTrainingExport eventTimestampColumn(String col) {
        this.eventTsColumn = col != null ? col : PointInTimeJoin.DEFAULT_EVENT_TS;
        return this;
    }

    public FeatureTrainingExport prefixWithViewName(boolean prefix) {
        this.prefixWithViewName = prefix;
        return this;
    }

    public Result run() {
        long t0 = System.nanoTime();
        List<Map<String, Object>> entities;
        if (entityRows != null) {
            entities = entityRows;
        } else if (entityDf != null) {
            entities = DataFrameBridge.toRows(entityDf);
        } else {
            throw new IllegalStateException("entityDataFrame or entityRows required");
        }
        if (entities.isEmpty()) {
            TrainingDataset empty = TrainingDataset.builder()
                    .labelColumn(labelColumn)
                    .eventTimestampColumn(eventTsColumn)
                    .build();
            return new Result(empty, DataFrame.create(), List.of(), List.of(), labelColumn,
                    featureService, System.nanoTime() - t0);
        }

        // Ensure event ts column name matches join options
        if (!eventTsColumn.equals(PointInTimeJoin.DEFAULT_EVENT_TS)) {
            for (Map<String, Object> e : entities) {
                if (!e.containsKey(PointInTimeJoin.DEFAULT_EVENT_TS) && e.containsKey(eventTsColumn)) {
                    e.put(PointInTimeJoin.DEFAULT_EVENT_TS, e.get(eventTsColumn));
                }
            }
        }

        FeatureProvider provider = platform.provider();
        PointInTimeJoin.Options opts = new PointInTimeJoin.Options()
                .eventTimestampColumn(PointInTimeJoin.DEFAULT_EVENT_TS)
                .prefixWithViewName(prefixWithViewName);

        // Rebuild provider's batch path uses default options; call HistoricalRetrieval via batch service
        TrainingDataset dataset;
        if (featureService != null && !featureService.isEmpty()) {
            dataset = provider.getHistoricalFeatures(entities, project, featureService, labelColumn);
        } else if (!views.isEmpty()) {
            dataset = provider.getHistoricalFeatures(entities, views, labelColumn);
        } else {
            throw new IllegalStateException("featureService or views required");
        }

        DataFrame df = DataFrameBridge.fromTrainingDataset(dataset);
        return new Result(dataset, df, dataset.entityKeys(), dataset.featureColumns(),
                labelColumn, featureService != null ? featureService : "",
                System.nanoTime() - t0);
    }

    /**
     * Infer recommend Feature list from registered FeatureViews (for Batch building / DeepFM).
     */
    public static List<Feature> inferRecommendFeatures(FeaturePlatform platform,
                                                       String project,
                                                       String featureService) {
        FeatureService svc = platform.registry().requireFeatureService(project, featureService);
        List<FeatureView> views = platform.registry().resolveViews(svc);
        List<Feature> out = new ArrayList<>();
        for (FeatureView v : views) {
            out.addAll(RecommendFeatureBridge.toRecommendFeatures(v));
        }
        return out;
    }

    static Batch rowsToBatch(List<Map<String, Object>> rows,
                             List<? extends Feature> featureDefs,
                             String labelColumn) {
        int n = rows.size();
        Map<String, Tensor> sparse = new LinkedHashMap<>();
        Map<String, Tensor> dense = new LinkedHashMap<>();
        Map<String, Tensor> sequence = new LinkedHashMap<>();

        for (Feature f : featureDefs) {
            String name = f.name();
            // also accept view__name prefixed columns
            if (f instanceof SparseFeature) {
                long[] data = new long[n];
                for (int i = 0; i < n; i++) {
                    data[i] = toLongId(lookup(rows.get(i), name));
                }
                sparse.put(name, TensorHelpers.longTensor(data));
            } else if (f instanceof SequenceFeature) {
                int maxLen = 1;
                long[][] seqs = new long[n][];
                for (int i = 0; i < n; i++) {
                    long[] s = toLongSeq(lookup(rows.get(i), name));
                    seqs[i] = s;
                    maxLen = Math.max(maxLen, s.length);
                }
                long[] flat = new long[n * maxLen];
                for (int i = 0; i < n; i++) {
                    long[] s = seqs[i];
                    System.arraycopy(s, 0, flat, i * maxLen, s.length);
                }
                sequence.put(name, TensorHelpers.tensor(flat, n, maxLen));
            } else if (f instanceof DenseFeature) {
                DenseFeature df = (DenseFeature) f;
                int dim = Math.max(1, df.embedDim());
                if (dim == 1) {
                    float[] data = new float[n];
                    for (int i = 0; i < n; i++) {
                        data[i] = (float) toDouble(lookup(rows.get(i), name));
                    }
                    dense.put(name, TensorHelpers.floatTensor(data));
                } else {
                    float[] data = new float[n * dim];
                    for (int i = 0; i < n; i++) {
                        float[] v = toFloatVec(lookup(rows.get(i), name), dim);
                        System.arraycopy(v, 0, data, i * dim, dim);
                    }
                    dense.put(name, TensorHelpers.tensor(data, n, dim));
                }
            }
        }

        Tensor labels = null;
        if (labelColumn != null && !labelColumn.isEmpty()) {
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                y[i] = (float) toDouble(rows.get(i).get(labelColumn));
            }
            labels = TensorHelpers.floatTensor(y);
        }
        return new Batch(sparse, dense, sequence, labels);
    }

    private static Object lookup(Map<String, Object> row, String name) {
        if (row.containsKey(name)) return row.get(name);
        // try suffix match view__name
        for (Map.Entry<String, Object> e : row.entrySet()) {
            if (e.getKey().endsWith("__" + name) || e.getKey().endsWith("." + name)) {
                return e.getValue();
            }
        }
        return null;
    }

    private static long toLongId(Object v) {
        if (v == null) return 0L;
        if (v instanceof Number) return ((Number) v).longValue();
        if (v instanceof Boolean) return ((Boolean) v) ? 1L : 0L;
        try {
            return Long.parseLong(String.valueOf(v));
        } catch (Exception e) {
            return Math.floorMod(v.hashCode(), 1_000_000);
        }
    }

    private static double toDouble(Object v) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        if (v instanceof Boolean) return ((Boolean) v) ? 1.0 : 0.0;
        if (v == null) return 0.0;
        try {
            return Double.parseDouble(String.valueOf(v));
        } catch (Exception e) {
            return 0.0;
        }
    }

    private static long[] toLongSeq(Object v) {
        if (v == null) return new long[0];
        if (v instanceof long[]) return (long[]) v;
        if (v instanceof int[]) {
            int[] a = (int[]) v;
            long[] o = new long[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (v instanceof List) {
            List<?> list = (List<?>) v;
            long[] o = new long[list.size()];
            for (int i = 0; i < list.size(); i++) o[i] = toLongId(list.get(i));
            return o;
        }
        return new long[]{toLongId(v)};
    }

    private static float[] toFloatVec(Object v, int dim) {
        float[] out = new float[dim];
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            System.arraycopy(a, 0, out, 0, Math.min(dim, a.length));
            return out;
        }
        if (v instanceof double[]) {
            double[] a = (double[]) v;
            for (int i = 0; i < Math.min(dim, a.length); i++) out[i] = (float) a[i];
            return out;
        }
        if (v instanceof Number) {
            out[0] = ((Number) v).floatValue();
        }
        return out;
    }
}
