/*
 * Feature drift via Population Stability Index (PSI).
 * Aligns with recommend.modelops.DriftDetector / Google TFX / Evidently / Alibaba monitoring.
 */
package org.bytedeco.pytorch.utils.feature.lifecycle;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/** PSI / distribution-shift monitor for numeric feature columns. */
public final class FeatureDriftMonitor {

    public static final class PsiResult {
        public final String feature;
        public final double psi;
        public final boolean alert;
        public final double threshold;

        public PsiResult(String feature, double psi, boolean alert, double threshold) {
            this.feature = feature;
            this.psi = psi;
            this.alert = alert;
            this.threshold = threshold;
        }

        @Override
        public String toString() {
            return "PSI{feature=" + feature + ", psi=" + String.format(Locale.ROOT, "%.4f", psi)
                    + ", alert=" + alert + "}";
        }
    }

    private final int bins;
    private final double alertThreshold;

    public FeatureDriftMonitor() {
        this(10, 0.2);
    }

    public FeatureDriftMonitor(int bins, double alertThreshold) {
        this.bins = Math.max(2, bins);
        this.alertThreshold = alertThreshold;
    }

    public PsiResult psi(String feature, double[] baseline, double[] current) {
        Objects.requireNonNull(baseline, "baseline");
        Objects.requireNonNull(current, "current");
        if (baseline.length == 0 || current.length == 0) {
            return new PsiResult(feature, Double.NaN, true, alertThreshold);
        }
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (double v : baseline) {
            if (Double.isFinite(v)) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        for (double v : current) {
            if (Double.isFinite(v)) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        if (!(max > min)) {
            return new PsiResult(feature, 0.0, false, alertThreshold);
        }
        double[] baseHist = hist(baseline, min, max, bins);
        double[] curHist = hist(current, min, max, bins);
        double psi = 0.0;
        for (int i = 0; i < bins; i++) {
            double p = Math.max(baseHist[i], 1e-6);
            double q = Math.max(curHist[i], 1e-6);
            psi += (q - p) * Math.log(q / p);
        }
        return new PsiResult(feature, psi, psi > alertThreshold, alertThreshold);
    }

    public List<PsiResult> psiColumns(List<Map<String, Object>> baseline,
                                      List<Map<String, Object>> current,
                                      List<String> columns) {
        List<PsiResult> out = new ArrayList<>();
        if (columns == null) return out;
        for (String c : columns) {
            out.add(psi(c, toDoubles(baseline, c), toDoubles(current, c)));
        }
        return out;
    }

    private static double[] toDoubles(List<Map<String, Object>> rows, String col) {
        if (rows == null) return new double[0];
        double[] a = new double[rows.size()];
        int n = 0;
        for (Map<String, Object> r : rows) {
            Object v = r.get(col);
            if (v instanceof Number) {
                a[n++] = ((Number) v).doubleValue();
            }
        }
        return n == a.length ? a : Arrays.copyOf(a, n);
    }

    private static double[] hist(double[] values, double min, double max, int bins) {
        double[] h = new double[bins];
        double width = (max - min) / bins;
        int total = 0;
        for (double v : values) {
            if (!Double.isFinite(v)) continue;
            int idx = (int) ((v - min) / width);
            if (idx < 0) idx = 0;
            if (idx >= bins) idx = bins - 1;
            h[idx] += 1.0;
            total++;
        }
        if (total == 0) return h;
        for (int i = 0; i < bins; i++) h[i] /= total;
        return h;
    }
}
