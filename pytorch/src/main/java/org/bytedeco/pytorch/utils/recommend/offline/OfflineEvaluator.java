/*
 * Offline evaluation for recommendation models — the gate before online AB.
 *
 * Industry practice (Meta, Google, ByteDance, Alibaba, Tencent, Netflix):
 *   1. Time-based holdout (NOT random shuffle) to avoid leakage
 *   2. User-level split for ranking metrics (NDCG/HR/MRR per user)
 *   3. Calibration (ECE) for CTR models before online serving
 *   4. AA / replay counterfactual as integrity checks
 *   5. Multi-metric scorecard + ship bar vs production baseline
 *
 * This package reuses {@link org.bytedeco.pytorch.utils.recommend.basic.Metrics}
 * for AUC / NDCG / etc. and adds engineering workflow around them.
 */
package org.bytedeco.pytorch.utils.recommend.offline;

import org.bytedeco.pytorch.utils.recommend.basic.Metrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Offline evaluator producing a multi-metric scorecard for candidate models.
 */
public final class OfflineEvaluator {

    /** Higher-is-better vs lower-is-better. */
    public enum MetricDirection {
        HIGHER_IS_BETTER,
        LOWER_IS_BETTER
    }

    private final List<String> rankingKs;
    private final boolean computeCalibration;
    private final int calibrationBins;

    private OfflineEvaluator(Builder b) {
        this.rankingKs = Collections.unmodifiableList(new ArrayList<>(b.rankingKs));
        this.computeCalibration = b.computeCalibration;
        this.calibrationBins = b.calibrationBins;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Pointwise / CTR-style evaluation: yTrue and yPred aligned arrays.
     */
    public Scorecard evaluatePointwise(String modelId, float[] yTrue, float[] yPred) {
        return evaluatePointwise(modelId, yTrue, yPred, null);
    }

    /**
     * Pointwise with optional per-user ids for GAUC.
     */
    public Scorecard evaluatePointwise(String modelId, float[] yTrue, float[] yPred, int[] userIds) {
        Objects.requireNonNull(yTrue, "yTrue");
        Objects.requireNonNull(yPred, "yPred");
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("yTrue/yPred length mismatch");
        }
        Map<String, Double> metrics = new LinkedHashMap<>();
        metrics.put("AUC", (double) Metrics.aucScore(yTrue, yPred));
        metrics.put("LogLoss", logLoss(yTrue, yPred));
        metrics.put("MSE", mse(yTrue, yPred));
        metrics.put("MAE", mae(yTrue, yPred));
        if (userIds != null && userIds.length == yTrue.length) {
            metrics.put("GAUC", (double) Metrics.gaucScore(yTrue, yPred, userIds));
        }
        if (computeCalibration) {
            CalibrationChecker.Result cal = CalibrationChecker.expectedCalibrationError(
                    yTrue, yPred, calibrationBins);
            metrics.put("ECE", cal.ece);
            metrics.put("CalibrationSlope", cal.slope);
            metrics.put("CalibrationIntercept", cal.intercept);
        }
        return new Scorecard(modelId, "pointwise", yTrue.length, metrics, directionsPointwise());
    }

    /**
     * Top-K ranking evaluation from per-user scored lists.
     *
     * @param modelId model identifier
     * @param users   map userId -> list of (itemId, score, label) sorted or unsorted
     */
    public Scorecard evaluateRanking(String modelId, Map<String, List<ScoredItem>> users) {
        Objects.requireNonNull(users, "users");
        Map<String, Double> acc = new LinkedHashMap<>();
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String kLabel : metricKeysForKs()) {
            acc.put(kLabel, 0.0);
            counts.put(kLabel, 0);
        }
        int userCount = 0;
        for (Map.Entry<String, List<ScoredItem>> e : users.entrySet()) {
            List<ScoredItem> items = new ArrayList<>(e.getValue());
            items.sort(Comparator.comparingDouble((ScoredItem s) -> s.score).reversed());
            // relevance array in rank order
            float[] labels = new float[items.size()];
            for (int i = 0; i < items.size(); i++) {
                labels[i] = items.get(i).label;
            }
            boolean hasPos = false;
            for (float l : labels) {
                if (l > 0) {
                    hasPos = true;
                    break;
                }
            }
            if (!hasPos) {
                continue; // skip users with no positive (standard recsys eval)
            }
            userCount++;
            for (int k : rankingKsAsInt()) {
                acc.merge("NDCG@" + k, ndcgAtK(labels, k), Double::sum);
                acc.merge("HR@" + k, hitAtK(labels, k), Double::sum);
                acc.merge("Recall@" + k, recallAtK(labels, k), Double::sum);
                acc.merge("Precision@" + k, precisionAtK(labels, k), Double::sum);
                acc.merge("MRR@" + k, mrrAtK(labels, k), Double::sum);
                counts.merge("NDCG@" + k, 1, Integer::sum);
                counts.merge("HR@" + k, 1, Integer::sum);
                counts.merge("Recall@" + k, 1, Integer::sum);
                counts.merge("Precision@" + k, 1, Integer::sum);
                counts.merge("MRR@" + k, 1, Integer::sum);
            }
        }
        Map<String, Double> metrics = new LinkedHashMap<>();
        for (String key : acc.keySet()) {
            int c = counts.getOrDefault(key, 0);
            metrics.put(key, c == 0 ? 0.0 : acc.get(key) / c);
        }
        metrics.put("EvalUsers", (double) userCount);
        return new Scorecard(modelId, "ranking", userCount, metrics, directionsRanking());
    }

    /**
     * Compare candidate vs baseline scorecard; produce delta report with ship bar.
     *
     * @param minRelativeLift minimum relative lift on primary metric to recommend ship
     *                        (e.g. 0.001 = 0.1%)
     */
    public DeltaReport compare(
            Scorecard baseline,
            Scorecard candidate,
            String primaryMetric,
            double minRelativeLift) {
        Objects.requireNonNull(baseline, "baseline");
        Objects.requireNonNull(candidate, "candidate");
        Objects.requireNonNull(primaryMetric, "primaryMetric");

        Map<String, Double> deltas = new LinkedHashMap<>();
        Map<String, Double> relDeltas = new LinkedHashMap<>();
        for (String key : unionKeys(baseline.metrics, candidate.metrics)) {
            double b = baseline.metrics.getOrDefault(key, Double.NaN);
            double c = candidate.metrics.getOrDefault(key, Double.NaN);
            if (Double.isNaN(b) || Double.isNaN(c)) {
                continue;
            }
            double d = c - b;
            deltas.put(key, d);
            relDeltas.put(key, b == 0.0 ? Double.NaN : d / Math.abs(b));
        }
        Double primaryRel = relDeltas.get(primaryMetric);
        MetricDirection dir = candidate.directions.getOrDefault(
                primaryMetric, MetricDirection.HIGHER_IS_BETTER);
        boolean ship;
        if (primaryRel == null || primaryRel.isNaN()) {
            ship = false;
        } else if (dir == MetricDirection.HIGHER_IS_BETTER) {
            ship = primaryRel >= minRelativeLift;
        } else {
            ship = primaryRel <= -minRelativeLift; // improvement is negative delta
        }
        return new DeltaReport(baseline.modelId, candidate.modelId, primaryMetric,
                minRelativeLift, deltas, relDeltas, ship, renderDelta(
                baseline, candidate, primaryMetric, deltas, relDeltas, ship));
    }

    // ---- metric primitives --------------------------------------------------

    public static double logLoss(float[] yTrue, float[] yPred) {
        final double eps = 1e-15;
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double p = Math.min(Math.max(yPred[i], eps), 1.0 - eps);
            double y = yTrue[i];
            sum += -(y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p));
        }
        return yTrue.length == 0 ? 0.0 : sum / yTrue.length;
    }

    public static double mse(float[] yTrue, float[] yPred) {
        double s = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double d = yTrue[i] - yPred[i];
            s += d * d;
        }
        return yTrue.length == 0 ? 0.0 : s / yTrue.length;
    }

    public static double mae(float[] yTrue, float[] yPred) {
        double s = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            s += Math.abs(yTrue[i] - yPred[i]);
        }
        return yTrue.length == 0 ? 0.0 : s / yTrue.length;
    }

    public static double ndcgAtK(float[] labelsInRankOrder, int k) {
        int n = Math.min(k, labelsInRankOrder.length);
        double dcg = 0.0;
        for (int i = 0; i < n; i++) {
            dcg += (Math.pow(2.0, labelsInRankOrder[i]) - 1.0) / (Math.log(i + 2.0) / Math.log(2.0));
        }
        float[] ideal = Arrays.copyOf(labelsInRankOrder, labelsInRankOrder.length);
        Arrays.sort(ideal);
        // sort ascending then reverse for ideal
        double idcg = 0.0;
        for (int i = 0; i < n; i++) {
            float rel = ideal[ideal.length - 1 - i];
            idcg += (Math.pow(2.0, rel) - 1.0) / (Math.log(i + 2.0) / Math.log(2.0));
        }
        return idcg == 0.0 ? 0.0 : dcg / idcg;
    }

    public static double hitAtK(float[] labelsInRankOrder, int k) {
        int n = Math.min(k, labelsInRankOrder.length);
        for (int i = 0; i < n; i++) {
            if (labelsInRankOrder[i] > 0) return 1.0;
        }
        return 0.0;
    }

    public static double recallAtK(float[] labelsInRankOrder, int k) {
        double totalPos = 0.0;
        for (float l : labelsInRankOrder) {
            if (l > 0) totalPos += 1.0;
        }
        if (totalPos == 0.0) return 0.0;
        int n = Math.min(k, labelsInRankOrder.length);
        double hit = 0.0;
        for (int i = 0; i < n; i++) {
            if (labelsInRankOrder[i] > 0) hit += 1.0;
        }
        return hit / totalPos;
    }

    public static double precisionAtK(float[] labelsInRankOrder, int k) {
        int n = Math.min(k, labelsInRankOrder.length);
        if (n == 0) return 0.0;
        double hit = 0.0;
        for (int i = 0; i < n; i++) {
            if (labelsInRankOrder[i] > 0) hit += 1.0;
        }
        return hit / n;
    }

    public static double mrrAtK(float[] labelsInRankOrder, int k) {
        int n = Math.min(k, labelsInRankOrder.length);
        for (int i = 0; i < n; i++) {
            if (labelsInRankOrder[i] > 0) {
                return 1.0 / (i + 1.0);
            }
        }
        return 0.0;
    }

    // ---- helpers ------------------------------------------------------------

    private List<String> metricKeysForKs() {
        List<String> keys = new ArrayList<>();
        for (int k : rankingKsAsInt()) {
            keys.add("NDCG@" + k);
            keys.add("HR@" + k);
            keys.add("Recall@" + k);
            keys.add("Precision@" + k);
            keys.add("MRR@" + k);
        }
        return keys;
    }

    private int[] rankingKsAsInt() {
        int[] ks = new int[rankingKs.size()];
        for (int i = 0; i < rankingKs.size(); i++) {
            ks[i] = Integer.parseInt(rankingKs.get(i));
        }
        return ks;
    }

    private static Map<String, MetricDirection> directionsPointwise() {
        Map<String, MetricDirection> d = new LinkedHashMap<>();
        d.put("AUC", MetricDirection.HIGHER_IS_BETTER);
        d.put("GAUC", MetricDirection.HIGHER_IS_BETTER);
        d.put("LogLoss", MetricDirection.LOWER_IS_BETTER);
        d.put("MSE", MetricDirection.LOWER_IS_BETTER);
        d.put("MAE", MetricDirection.LOWER_IS_BETTER);
        d.put("ECE", MetricDirection.LOWER_IS_BETTER);
        d.put("CalibrationSlope", MetricDirection.HIGHER_IS_BETTER); // closer to 1 tracked separately
        d.put("CalibrationIntercept", MetricDirection.LOWER_IS_BETTER);
        return d;
    }

    private Map<String, MetricDirection> directionsRanking() {
        Map<String, MetricDirection> d = new LinkedHashMap<>();
        for (String key : metricKeysForKs()) {
            d.put(key, MetricDirection.HIGHER_IS_BETTER);
        }
        return d;
    }

    private static List<String> unionKeys(Map<String, Double> a, Map<String, Double> b) {
        LinkedHashMap<String, Boolean> u = new LinkedHashMap<>();
        for (String k : a.keySet()) u.put(k, Boolean.TRUE);
        for (String k : b.keySet()) u.put(k, Boolean.TRUE);
        return new ArrayList<>(u.keySet());
    }

    private static String renderDelta(
            Scorecard baseline,
            Scorecard candidate,
            String primary,
            Map<String, Double> deltas,
            Map<String, Double> rel,
            boolean ship) {
        StringBuilder sb = new StringBuilder();
        sb.append("===== Offline Delta Report =====\n");
        sb.append("baseline: ").append(baseline.modelId).append('\n');
        sb.append("candidate: ").append(candidate.modelId).append('\n');
        sb.append("primary: ").append(primary).append('\n');
        for (String key : deltas.keySet()) {
            Double r = rel.get(key);
            sb.append(String.format(Locale.ROOT, "  %s: base=%.6f cand=%.6f delta=%+.6f rel=%s\n",
                    key,
                    baseline.metrics.getOrDefault(key, Double.NaN),
                    candidate.metrics.getOrDefault(key, Double.NaN),
                    deltas.get(key),
                    r == null || r.isNaN() ? "n/a" : String.format(Locale.ROOT, "%+.3f%%", r * 100.0)));
        }
        sb.append("SHIP_OFFLINE: ").append(ship).append('\n');
        return sb.toString();
    }

    // ---- types --------------------------------------------------------------

    /** One scored item in a user list. */
    public static final class ScoredItem {
        public final String itemId;
        public final double score;
        public final float label;

        public ScoredItem(String itemId, double score, float label) {
            this.itemId = itemId;
            this.score = score;
            this.label = label;
        }
    }

    /** Multi-metric evaluation result for one model. */
    public static final class Scorecard {
        public final String modelId;
        public final String evalType;
        public final long sampleSize;
        public final Map<String, Double> metrics;
        public final Map<String, MetricDirection> directions;

        public Scorecard(
                String modelId,
                String evalType,
                long sampleSize,
                Map<String, Double> metrics,
                Map<String, MetricDirection> directions) {
            this.modelId = modelId;
            this.evalType = evalType;
            this.sampleSize = sampleSize;
            this.metrics = Collections.unmodifiableMap(new LinkedHashMap<>(metrics));
            this.directions = Collections.unmodifiableMap(new LinkedHashMap<>(directions));
        }

        public double get(String key) {
            return metrics.getOrDefault(key, Double.NaN);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Scorecard{model=").append(modelId)
                    .append(", type=").append(evalType)
                    .append(", n=").append(sampleSize).append(", metrics={\n");
            for (Map.Entry<String, Double> e : metrics.entrySet()) {
                sb.append(String.format(Locale.ROOT, "  %s: %.6f\n", e.getKey(), e.getValue()));
            }
            sb.append("}}");
            return sb.toString();
        }
    }

    /** Candidate vs baseline comparison. */
    public static final class DeltaReport {
        public final String baselineModelId;
        public final String candidateModelId;
        public final String primaryMetric;
        public final double minRelativeLift;
        public final Map<String, Double> absoluteDeltas;
        public final Map<String, Double> relativeDeltas;
        public final boolean shipRecommended;
        public final String text;

        public DeltaReport(
                String baselineModelId,
                String candidateModelId,
                String primaryMetric,
                double minRelativeLift,
                Map<String, Double> absoluteDeltas,
                Map<String, Double> relativeDeltas,
                boolean shipRecommended,
                String text) {
            this.baselineModelId = baselineModelId;
            this.candidateModelId = candidateModelId;
            this.primaryMetric = primaryMetric;
            this.minRelativeLift = minRelativeLift;
            this.absoluteDeltas = Collections.unmodifiableMap(new LinkedHashMap<>(absoluteDeltas));
            this.relativeDeltas = Collections.unmodifiableMap(new LinkedHashMap<>(relativeDeltas));
            this.shipRecommended = shipRecommended;
            this.text = text;
        }

        @Override
        public String toString() {
            return text;
        }
    }

    public static final class Builder {
        private final List<String> rankingKs = new ArrayList<>(Arrays.asList("5", "10", "20"));
        private boolean computeCalibration = true;
        private int calibrationBins = 10;

        public Builder rankingKs(int... ks) {
            rankingKs.clear();
            for (int k : ks) {
                rankingKs.add(String.valueOf(k));
            }
            return this;
        }

        public Builder computeCalibration(boolean computeCalibration) {
            this.computeCalibration = computeCalibration;
            return this;
        }

        public Builder calibrationBins(int calibrationBins) {
            this.calibrationBins = calibrationBins;
            return this;
        }

        public OfflineEvaluator build() {
            return new OfflineEvaluator(this);
        }
    }
}
