/*
 * Prediction calibration diagnostics for CTR / CVR models.
 *
 * A well-calibrated model satisfies P(y=1 | score=p) ≈ p.
 * Online ranking often re-calibrates (Platt / isotonic / binning) after
 * offline training — Meta, Google Ads, Alibaba all run calibration checks
 * as a ship gate because miscalibration breaks bid / rank mixing.
 *
 * Metrics:
 *   - ECE (Expected Calibration Error) with equal-width or equal-mass bins
 *   - MCE (Maximum Calibration Error)
 *   - Reliability diagram points (bin confidence, accuracy, count)
 *   - Linear calibration slope / intercept (y ~ a * p + b); ideal a=1, b=0
 */
package org.bytedeco.pytorch.utils.recommend.offline;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/** Calibration checker utilities. */
public final class CalibrationChecker {

    private CalibrationChecker() {}

    public static final class BinStat {
        public final int binIndex;
        public final double confLow;
        public final double confHigh;
        public final double avgConfidence;
        public final double avgAccuracy;
        public final long count;

        public BinStat(
                int binIndex,
                double confLow,
                double confHigh,
                double avgConfidence,
                double avgAccuracy,
                long count) {
            this.binIndex = binIndex;
            this.confLow = confLow;
            this.confHigh = confHigh;
            this.avgConfidence = avgConfidence;
            this.avgAccuracy = avgAccuracy;
            this.count = count;
        }

        public double gap() {
            return Math.abs(avgConfidence - avgAccuracy);
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "Bin[%d] (%.2f,%.2f] n=%d conf=%.4f acc=%.4f gap=%.4f",
                    binIndex, confLow, confHigh, count, avgConfidence, avgAccuracy, gap());
        }
    }

    public static final class Result {
        public final double ece;
        public final double mce;
        public final double slope;
        public final double intercept;
        public final List<BinStat> bins;
        public final long n;

        public Result(double ece, double mce, double slope, double intercept, List<BinStat> bins, long n) {
            this.ece = ece;
            this.mce = mce;
            this.slope = slope;
            this.intercept = intercept;
            this.bins = Collections.unmodifiableList(new ArrayList<>(bins));
            this.n = n;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "Calibration{n=%d ECE=%.6f MCE=%.6f slope=%.4f intercept=%.4f bins=%d}",
                    n, ece, mce, slope, intercept, bins.size());
        }
    }

    /**
     * Equal-width bin ECE on [0, 1].
     */
    public static Result expectedCalibrationError(float[] yTrue, float[] yPred, int nBins) {
        Objects.requireNonNull(yTrue, "yTrue");
        Objects.requireNonNull(yPred, "yPred");
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("length mismatch");
        }
        if (nBins < 2) {
            throw new IllegalArgumentException("nBins must be >= 2");
        }
        int n = yTrue.length;
        long[] counts = new long[nBins];
        double[] confSum = new double[nBins];
        double[] accSum = new double[nBins];

        for (int i = 0; i < n; i++) {
            double p = clamp01(yPred[i]);
            int b = (int) Math.min(nBins - 1, Math.floor(p * nBins));
            counts[b]++;
            confSum[b] += p;
            accSum[b] += yTrue[i];
        }

        List<BinStat> bins = new ArrayList<>();
        double ece = 0.0;
        double mce = 0.0;
        for (int b = 0; b < nBins; b++) {
            double low = (double) b / nBins;
            double high = (double) (b + 1) / nBins;
            if (counts[b] == 0) {
                bins.add(new BinStat(b, low, high, 0.0, 0.0, 0));
                continue;
            }
            double avgConf = confSum[b] / counts[b];
            double avgAcc = accSum[b] / counts[b];
            BinStat stat = new BinStat(b, low, high, avgConf, avgAcc, counts[b]);
            bins.add(stat);
            double gap = stat.gap();
            ece += (counts[b] / (double) n) * gap;
            if (gap > mce) mce = gap;
        }

        // OLS: y = slope * p + intercept
        double meanP = 0.0;
        double meanY = 0.0;
        for (int i = 0; i < n; i++) {
            meanP += clamp01(yPred[i]);
            meanY += yTrue[i];
        }
        meanP /= Math.max(1, n);
        meanY /= Math.max(1, n);
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < n; i++) {
            double p = clamp01(yPred[i]);
            num += (p - meanP) * (yTrue[i] - meanY);
            den += (p - meanP) * (p - meanP);
        }
        double slope = den == 0.0 ? 0.0 : num / den;
        double intercept = meanY - slope * meanP;

        return new Result(ece, mce, slope, intercept, bins, n);
    }

    /**
     * Apply simple binning calibration: map raw score to empirical positive
     * rate of its bin (isotonic-lite). Returns calibrated scores.
     */
    public static float[] binCalibrate(float[] yTrueTrain, float[] yPredTrain, float[] yPredApply, int nBins) {
        Result train = expectedCalibrationError(yTrueTrain, yPredTrain, nBins);
        float[] out = new float[yPredApply.length];
        for (int i = 0; i < yPredApply.length; i++) {
            double p = clamp01(yPredApply[i]);
            int b = (int) Math.min(nBins - 1, Math.floor(p * nBins));
            BinStat stat = train.bins.get(b);
            // Fall back to raw if empty bin.
            out[i] = stat.count == 0 ? (float) p : (float) stat.avgAccuracy;
        }
        return out;
    }

    /**
     * Platt-scaling style logistic calibration fit via Newton steps on
     * train predictions. Returns [A, B] for p' = sigma(A * logit(p) + B).
     * Simplified single-feature logistic regression.
     */
    public static double[] fitPlatt(float[] yTrue, float[] yPred) {
        Objects.requireNonNull(yTrue);
        Objects.requireNonNull(yPred);
        int n = yTrue.length;
        // Initialize A, B
        double a = 1.0;
        double b = 0.0;
        final double eps = 1e-15;
        for (int iter = 0; iter < 50; iter++) {
            double gA = 0.0;
            double gB = 0.0;
            double hAA = 0.0;
            double hBB = 0.0;
            double hAB = 0.0;
            for (int i = 0; i < n; i++) {
                double p = clamp01(yPred[i]);
                p = Math.min(Math.max(p, eps), 1.0 - eps);
                double x = Math.log(p / (1.0 - p)); // logit
                double z = a * x + b;
                double pred = sigmoid(z);
                double y = yTrue[i];
                double err = pred - y;
                gA += err * x;
                gB += err;
                double w = pred * (1.0 - pred);
                hAA += w * x * x;
                hBB += w;
                hAB += w * x;
            }
            // Solve 2x2 Newton system
            double det = hAA * hBB - hAB * hAB;
            if (Math.abs(det) < 1e-18) {
                break;
            }
            double dA = (hBB * gA - hAB * gB) / det;
            double dB = (hAA * gB - hAB * gA) / det;
            a -= dA;
            b -= dB;
            if (Math.abs(dA) + Math.abs(dB) < 1e-8) {
                break;
            }
        }
        return new double[] {a, b};
    }

    public static float[] applyPlatt(float[] yPred, double a, double b) {
        final double eps = 1e-15;
        float[] out = new float[yPred.length];
        for (int i = 0; i < yPred.length; i++) {
            double p = clamp01(yPred[i]);
            p = Math.min(Math.max(p, eps), 1.0 - eps);
            double x = Math.log(p / (1.0 - p));
            out[i] = (float) sigmoid(a * x + b);
        }
        return out;
    }

    private static double sigmoid(double z) {
        if (z >= 0) {
            double ez = Math.exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            double ez = Math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    private static double clamp01(double x) {
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
    }
}
