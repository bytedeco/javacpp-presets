/*
 * Feature and prediction drift detection for online recommendation models.
 *
 * Industry practice (Google TFX / Evidently / Alibaba / Meta monitoring):
 *   - Population Stability Index (PSI) on feature distributions
 *   - KL divergence / JS divergence
 *   - Prediction score distribution shift
 *   - Null / missing rate spikes
 *   - Label delay & calibration drift (online vs offline)
 *
 * Alerts feed into modelops rollback / retraining triggers.
 */
package org.bytedeco.pytorch.recommend.modelops;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/** Drift detection utilities and online monitors. */
public final class DriftDetector {

    private DriftDetector() {}

    public static final class PsiResult {
        public final String featureName;
        public final double psi;
        public final int bins;
        public final boolean alert;
        public final double alertThreshold;

        public PsiResult(String featureName, double psi, int bins, boolean alert, double alertThreshold) {
            this.featureName = featureName;
            this.psi = psi;
            this.bins = bins;
            this.alert = alert;
            this.alertThreshold = alertThreshold;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "PSI[%s]=%.6f bins=%d alert=%s (thr=%.3f)",
                    featureName, psi, bins, alert, alertThreshold);
        }
    }

    public static final class DriftReport {
        public final List<PsiResult> featurePsi;
        public final PsiResult scorePsi;
        public final double missingRateBaseline;
        public final double missingRateCurrent;
        public final boolean missingRateAlert;
        public final boolean anyAlert;
        public final String text;

        public DriftReport(
                List<PsiResult> featurePsi,
                PsiResult scorePsi,
                double missingRateBaseline,
                double missingRateCurrent,
                boolean missingRateAlert) {
            this.featurePsi = Collections.unmodifiableList(new ArrayList<>(featurePsi));
            this.scorePsi = scorePsi;
            this.missingRateBaseline = missingRateBaseline;
            this.missingRateCurrent = missingRateCurrent;
            this.missingRateAlert = missingRateAlert;
            boolean any = missingRateAlert || (scorePsi != null && scorePsi.alert);
            for (PsiResult r : featurePsi) {
                if (r.alert) any = true;
            }
            this.anyAlert = any;
            this.text = render();
        }

        private String render() {
            StringBuilder sb = new StringBuilder();
            sb.append("===== Drift Report =====\n");
            for (PsiResult r : featurePsi) {
                sb.append("  ").append(r).append('\n');
            }
            if (scorePsi != null) {
                sb.append("  score ").append(scorePsi).append('\n');
            }
            sb.append(String.format(Locale.ROOT,
                    "  missingRate baseline=%.4f current=%.4f alert=%s\n",
                    missingRateBaseline, missingRateCurrent, missingRateAlert));
            sb.append("ANY_ALERT: ").append(anyAlert).append('\n');
            return sb.toString();
        }

        @Override
        public String toString() {
            return text;
        }
    }

    /**
     * Population Stability Index between baseline and current distributions.
     *
     * <pre>
     *   PSI = sum( (cur_i - base_i) * ln(cur_i / base_i) )
     * </pre>
     *
     * Industry thresholds (common rule of thumb):
     *   &lt; 0.1 : stable
     *   0.1–0.25 : mild shift — monitor
     *   &gt; 0.25 : significant drift — investigate / retrain
     *
     * @param baseline baseline samples
     * @param current  current samples
     * @param nBins    number of equal-width bins over combined range
     * @param featureName name for reporting
     * @param alertThreshold typically 0.25
     */
    public static PsiResult psi(
            double[] baseline, double[] current, int nBins,
            String featureName, double alertThreshold) {
        Objects.requireNonNull(baseline, "baseline");
        Objects.requireNonNull(current, "current");
        if (baseline.length == 0 || current.length == 0) {
            throw new IllegalArgumentException("empty samples");
        }
        if (nBins < 2) throw new IllegalArgumentException("nBins >= 2");

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double v : baseline) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        for (double v : current) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        if (max <= min) {
            max = min + 1e-6;
        }
        double width = (max - min) / nBins;
        double[] baseHist = new double[nBins];
        double[] curHist = new double[nBins];
        for (double v : baseline) {
            int b = binIndex(v, min, width, nBins);
            baseHist[b] += 1.0;
        }
        for (double v : current) {
            int b = binIndex(v, min, width, nBins);
            curHist[b] += 1.0;
        }
        // Normalize with Laplace smoothing to avoid log(0)
        double eps = 1e-4;
        for (int i = 0; i < nBins; i++) {
            baseHist[i] = baseHist[i] / baseline.length + eps;
            curHist[i] = curHist[i] / current.length + eps;
        }
        // Renormalize
        double sumB = 0, sumC = 0;
        for (int i = 0; i < nBins; i++) {
            sumB += baseHist[i];
            sumC += curHist[i];
        }
        double psi = 0.0;
        for (int i = 0; i < nBins; i++) {
            double b = baseHist[i] / sumB;
            double c = curHist[i] / sumC;
            psi += (c - b) * Math.log(c / b);
        }
        return new PsiResult(featureName, psi, nBins, psi >= alertThreshold, alertThreshold);
    }

    private static int binIndex(double v, double min, double width, int nBins) {
        int b = (int) Math.floor((v - min) / width);
        if (b < 0) return 0;
        if (b >= nBins) return nBins - 1;
        return b;
    }

    /**
     * KL divergence KL(P || Q) on two discrete distributions (same length, non-negative).
     */
    public static double klDivergence(double[] p, double[] q) {
        if (p.length != q.length) throw new IllegalArgumentException("length mismatch");
        double sumP = 0, sumQ = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] < 0 || q[i] < 0) throw new IllegalArgumentException("negative mass");
            sumP += p[i];
            sumQ += q[i];
        }
        if (sumP <= 0 || sumQ <= 0) throw new IllegalArgumentException("zero mass");
        double kl = 0.0;
        final double eps = 1e-12;
        for (int i = 0; i < p.length; i++) {
            double pi = p[i] / sumP;
            double qi = Math.max(q[i] / sumQ, eps);
            if (pi > 0) {
                kl += pi * Math.log(pi / qi);
            }
        }
        return kl;
    }

    /**
     * Jensen-Shannon divergence (symmetric, bounded).
     */
    public static double jsDivergence(double[] p, double[] q) {
        if (p.length != q.length) throw new IllegalArgumentException("length mismatch");
        double[] m = new double[p.length];
        double sumP = 0, sumQ = 0;
        for (int i = 0; i < p.length; i++) {
            sumP += p[i];
            sumQ += q[i];
        }
        for (int i = 0; i < p.length; i++) {
            m[i] = 0.5 * (p[i] / sumP + q[i] / sumQ);
        }
        double[] pn = new double[p.length];
        double[] qn = new double[q.length];
        for (int i = 0; i < p.length; i++) {
            pn[i] = p[i] / sumP;
            qn[i] = q[i] / sumQ;
        }
        return 0.5 * klDivergence(pn, m) + 0.5 * klDivergence(qn, m);
    }

    /**
     * Build a multi-feature drift report.
     *
     * @param baselineFeatures map feature -> baseline samples
     * @param currentFeatures  map feature -> current samples
     * @param baselineScores   optional baseline model scores
     * @param currentScores    optional current model scores
     * @param baselineMissingRate missing feature rate in baseline
     * @param currentMissingRate  missing feature rate now
     * @param psiThreshold     alert threshold (e.g. 0.25)
     * @param missingRateDeltaAlert alert if current - baseline > this
     */
    public static DriftReport report(
            Map<String, double[]> baselineFeatures,
            Map<String, double[]> currentFeatures,
            double[] baselineScores,
            double[] currentScores,
            double baselineMissingRate,
            double currentMissingRate,
            double psiThreshold,
            double missingRateDeltaAlert) {
        List<PsiResult> featureResults = new ArrayList<>();
        if (baselineFeatures != null && currentFeatures != null) {
            for (String key : baselineFeatures.keySet()) {
                double[] cur = currentFeatures.get(key);
                if (cur == null) continue;
                featureResults.add(psi(baselineFeatures.get(key), cur, 10, key, psiThreshold));
            }
        }
        PsiResult scorePsi = null;
        if (baselineScores != null && currentScores != null
                && baselineScores.length > 0 && currentScores.length > 0) {
            scorePsi = psi(baselineScores, currentScores, 10, "prediction_score", psiThreshold);
        }
        boolean missingAlert = (currentMissingRate - baselineMissingRate) > missingRateDeltaAlert;
        return new DriftReport(featureResults, scorePsi,
                baselineMissingRate, currentMissingRate, missingAlert);
    }

    /**
     * Online streaming histogram accumulator for a single feature — used to
     * compare a fixed baseline histogram against a live window.
     */
    public static final class StreamingHistogram {
        private final double min;
        private final double max;
        private final int nBins;
        private final long[] counts;
        private long total;
        private long missing;

        public StreamingHistogram(double min, double max, int nBins) {
            if (max <= min || nBins < 2) throw new IllegalArgumentException("invalid histogram");
            this.min = min;
            this.max = max;
            this.nBins = nBins;
            this.counts = new long[nBins];
        }

        public synchronized void observe(double value) {
            if (Double.isNaN(value)) {
                missing++;
                return;
            }
            double width = (max - min) / nBins;
            int b = binIndex(value, min, width, nBins);
            counts[b]++;
            total++;
        }

        public synchronized void observeMissing() {
            missing++;
        }

        public synchronized double[] distribution() {
            double[] d = new double[nBins];
            long t = Math.max(1, total);
            for (int i = 0; i < nBins; i++) {
                d[i] = counts[i] / (double) t;
            }
            return d;
        }

        public synchronized long total() {
            return total;
        }

        public synchronized double missingRate() {
            long all = total + missing;
            return all == 0 ? 0.0 : missing / (double) all;
        }

        public synchronized void reset() {
            Arrays.fill(counts, 0L);
            total = 0;
            missing = 0;
        }

        /**
         * PSI against a baseline distribution (same binning).
         */
        public synchronized PsiResult psiAgainst(double[] baselineDist, String name, double thr) {
            if (baselineDist.length != nBins) {
                throw new IllegalArgumentException("baseline dist length mismatch");
            }
            double[] cur = distribution();
            final double eps = 1e-4;
            double psi = 0.0;
            for (int i = 0; i < nBins; i++) {
                double b = Math.max(baselineDist[i], eps);
                double c = Math.max(cur[i], eps);
                psi += (c - b) * Math.log(c / b);
            }
            return new PsiResult(name, psi, nBins, psi >= thr, thr);
        }
    }

    /**
     * Multi-feature online drift monitor with baseline freeze.
     */
    public static final class OnlineMonitor {
        private final Map<String, StreamingHistogram> live = new ConcurrentHashMap<>();
        private final Map<String, double[]> baselines = new ConcurrentHashMap<>();
        private final double psiThreshold;
        private final int nBins;

        public OnlineMonitor(double psiThreshold, int nBins) {
            this.psiThreshold = psiThreshold;
            this.nBins = nBins;
        }

        public void defineFeature(String name, double min, double max) {
            live.put(name, new StreamingHistogram(min, max, nBins));
        }

        public void freezeBaseline(String name) {
            StreamingHistogram h = live.get(name);
            if (h == null) throw new IllegalArgumentException("unknown feature " + name);
            baselines.put(name, h.distribution());
            h.reset();
        }

        public void observe(String name, double value) {
            StreamingHistogram h = live.get(name);
            if (h != null) h.observe(value);
        }

        public List<PsiResult> evaluate() {
            List<PsiResult> results = new ArrayList<>();
            for (Map.Entry<String, double[]> e : baselines.entrySet()) {
                StreamingHistogram h = live.get(e.getKey());
                if (h == null || h.total() < 100) continue;
                results.add(h.psiAgainst(e.getValue(), e.getKey(), psiThreshold));
            }
            return results;
        }

        public boolean anyAlert() {
            for (PsiResult r : evaluate()) {
                if (r.alert) return true;
            }
            return false;
        }
    }
}
