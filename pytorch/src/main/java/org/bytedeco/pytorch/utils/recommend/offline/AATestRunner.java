/*
 * A/A (null) test utilities for offline integrity and online pre-flight.
 *
 * An A/A test assigns units to two identical treatments. Any "significant"
 * difference indicates:
 *   - buggy diversion / logging join
 *   - SRM
 *   - metric definition bugs
 *   - peeking / multiple-comparison inflation
 *
 * Industry (Kohavi, Deng, Fabijan; Meta XP; Microsoft ExP; ByteDance):
 *   Always run A/A before A/B. Target false-positive rate ≈ alpha.
 */
package org.bytedeco.pytorch.utils.recommend.offline;

import org.bytedeco.pytorch.utils.recommend.abtest.StatisticalTest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Random;

/** A/A test runner for metric integrity. */
public final class AATestRunner {

    private AATestRunner() {}

    public static final class TrialResult {
        public final int trialIndex;
        public final StatisticalTest.MeanTestResult meanTest;
        public final boolean falsePositive;

        public TrialResult(int trialIndex, StatisticalTest.MeanTestResult meanTest, boolean falsePositive) {
            this.trialIndex = trialIndex;
            this.meanTest = meanTest;
            this.falsePositive = falsePositive;
        }
    }

    public static final class Summary {
        public final int trials;
        public final int falsePositives;
        public final double empiricalFpr;
        public final double targetAlpha;
        public final boolean healthy;
        public final double fprTolerance;
        public final List<TrialResult> trialsDetail;
        public final String text;

        public Summary(
                int trials,
                int falsePositives,
                double empiricalFpr,
                double targetAlpha,
                boolean healthy,
                double fprTolerance,
                List<TrialResult> trialsDetail,
                String text) {
            this.trials = trials;
            this.falsePositives = falsePositives;
            this.empiricalFpr = empiricalFpr;
            this.targetAlpha = targetAlpha;
            this.healthy = healthy;
            this.fprTolerance = fprTolerance;
            this.trialsDetail = Collections.unmodifiableList(new ArrayList<>(trialsDetail));
            this.text = text;
        }

        @Override
        public String toString() {
            return text;
        }
    }

    /**
     * Repeated random split of a single metric sample into two arms; count
     * how often Welch t-test is significant. Under H0, rate should ≈ alpha.
     *
     * @param values      metric observations (e.g. per-user CTR)
     * @param trials      number of random A/A splits
     * @param alpha       significance level
     * @param fprTolerance allowed absolute deviation from alpha (e.g. 0.02)
     * @param seed        RNG seed
     */
    public static Summary runMeanAA(
            double[] values, int trials, double alpha, double fprTolerance, long seed) {
        Objects.requireNonNull(values, "values");
        if (values.length < 4) {
            throw new IllegalArgumentException("need at least 4 observations");
        }
        if (trials < 1) {
            throw new IllegalArgumentException("trials must be >= 1");
        }
        Random rng = new Random(seed);
        List<TrialResult> detail = new ArrayList<>(trials);
        int fp = 0;
        for (int t = 0; t < trials; t++) {
            // Shuffle indices into two halves
            int n = values.length;
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            // Fisher-Yates
            for (int i = n - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
            int half = n / 2;
            double[] a = new double[half];
            double[] b = new double[n - half];
            for (int i = 0; i < half; i++) a[i] = values[idx[i]];
            for (int i = half; i < n; i++) b[i - half] = values[idx[i]];
            StatisticalTest.MeanTestResult r = StatisticalTest.welchTTest(a, b, alpha);
            boolean falsePos = r.significantAtAlpha;
            if (falsePos) fp++;
            detail.add(new TrialResult(t, r, falsePos));
        }
        double empiric = fp / (double) trials;
        boolean healthy = Math.abs(empiric - alpha) <= fprTolerance;
        String text = String.format(Locale.ROOT,
                "AATest trials=%d FP=%d empiricFPR=%.4f targetAlpha=%.4f tol=%.4f healthy=%s",
                trials, fp, empiric, alpha, fprTolerance, healthy);
        return new Summary(trials, fp, empiric, alpha, healthy, fprTolerance, detail, text);
    }

    /**
     * Simulation-based power check: inject a known lift into treatment half
     * and estimate detection rate (power) at given alpha.
     *
     * @param baseline values under null
     * @param absoluteLift added to treatment arm
     * @param trials simulation count
     * @param alpha significance
     * @param seed RNG
     * @return estimated power in [0,1]
     */
    public static double estimatePower(
            double[] baseline, double absoluteLift, int trials, double alpha, long seed) {
        Objects.requireNonNull(baseline);
        if (baseline.length < 4) {
            throw new IllegalArgumentException("need at least 4 observations");
        }
        Random rng = new Random(seed);
        int detect = 0;
        int n = baseline.length;
        for (int t = 0; t < trials; t++) {
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            for (int i = n - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
            int half = n / 2;
            double[] a = new double[half];
            double[] b = new double[n - half];
            for (int i = 0; i < half; i++) a[i] = baseline[idx[i]];
            for (int i = half; i < n; i++) b[i - half] = baseline[idx[i]] + absoluteLift;
            StatisticalTest.MeanTestResult r = StatisticalTest.welchTTest(a, b, alpha);
            if (r.significantAtAlpha && r.absoluteDelta > 0) {
                detect++;
            }
        }
        return detect / (double) trials;
    }
}
