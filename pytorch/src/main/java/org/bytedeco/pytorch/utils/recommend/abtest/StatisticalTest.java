/*
 * Statistical tests used by large-scale online experiments.
 *
 * References / industry practice:
 *   - Student t-test / Welch t-test for means (CTR, dwell, revenue)
 *   - Two-proportion z-test for rates
 *   - Chi-square SRM (Sample Ratio Mismatch) — Kohavi et al., "Trustworthy
 *     Online Controlled Experiments" / Microsoft ExP
 *   - CUPED (Controlled-experiment Using Pre-Experiment Data) — Deng et al.,
 *     Microsoft, KDD 2013 — variance reduction via pre-period covariate
 *   - Sequential / always-valid p-values are NOT fully implemented here;
 *     we expose fixed-horizon tests + simple peeking warning flags.
 *
 * This class is pure Java numerics — no external stats library dependency.
 */
package org.bytedeco.pytorch.utils.recommend.abtest;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * Offline / online experiment statistical utilities.
 */
public final class StatisticalTest {

    private StatisticalTest() {}

    // ---- result types -------------------------------------------------------

    /** Two-sample mean comparison result. */
    public static final class MeanTestResult {
        public final double controlMean;
        public final double treatmentMean;
        public final double absoluteDelta;
        public final double relativeDelta;
        public final double tStatistic;
        public final double degreesOfFreedom;
        public final double pValue;
        public final double ciLow;
        public final double ciHigh;
        public final double controlN;
        public final double treatmentN;
        public final boolean significantAtAlpha;
        public final double alpha;
        public final String method;

        public MeanTestResult(
                double controlMean,
                double treatmentMean,
                double absoluteDelta,
                double relativeDelta,
                double tStatistic,
                double degreesOfFreedom,
                double pValue,
                double ciLow,
                double ciHigh,
                double controlN,
                double treatmentN,
                boolean significantAtAlpha,
                double alpha,
                String method) {
            this.controlMean = controlMean;
            this.treatmentMean = treatmentMean;
            this.absoluteDelta = absoluteDelta;
            this.relativeDelta = relativeDelta;
            this.tStatistic = tStatistic;
            this.degreesOfFreedom = degreesOfFreedom;
            this.pValue = pValue;
            this.ciLow = ciLow;
            this.ciHigh = ciHigh;
            this.controlN = controlN;
            this.treatmentN = treatmentN;
            this.significantAtAlpha = significantAtAlpha;
            this.alpha = alpha;
            this.method = method;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "MeanTest[%s] ctrl=%.6f treat=%.6f delta=%.6f (%.3f%%) t=%.4f p=%.6f ci=[%.6f,%.6f] sig=%s",
                    method, controlMean, treatmentMean, absoluteDelta, relativeDelta * 100.0,
                    tStatistic, pValue, ciLow, ciHigh, significantAtAlpha);
        }
    }

    /** SRM chi-square result. */
    public static final class SrmResult {
        public final long[] observed;
        public final double[] expected;
        public final double chiSquare;
        public final double degreesOfFreedom;
        public final double pValue;
        public final boolean srmDetected;
        public final double alpha;

        public SrmResult(
                long[] observed,
                double[] expected,
                double chiSquare,
                double degreesOfFreedom,
                double pValue,
                boolean srmDetected,
                double alpha) {
            this.observed = observed;
            this.expected = expected;
            this.chiSquare = chiSquare;
            this.degreesOfFreedom = degreesOfFreedom;
            this.pValue = pValue;
            this.srmDetected = srmDetected;
            this.alpha = alpha;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "SRM chi2=%.4f df=%.0f p=%.6g detected=%s alpha=%.4f",
                    chiSquare, degreesOfFreedom, pValue, srmDetected, alpha);
        }
    }

    /** Proportion (rate) test result. */
    public static final class ProportionTestResult {
        public final double controlRate;
        public final double treatmentRate;
        public final double absoluteDelta;
        public final double relativeDelta;
        public final double zStatistic;
        public final double pValue;
        public final double ciLow;
        public final double ciHigh;
        public final boolean significantAtAlpha;
        public final double alpha;

        public ProportionTestResult(
                double controlRate,
                double treatmentRate,
                double absoluteDelta,
                double relativeDelta,
                double zStatistic,
                double pValue,
                double ciLow,
                double ciHigh,
                boolean significantAtAlpha,
                double alpha) {
            this.controlRate = controlRate;
            this.treatmentRate = treatmentRate;
            this.absoluteDelta = absoluteDelta;
            this.relativeDelta = relativeDelta;
            this.zStatistic = zStatistic;
            this.pValue = pValue;
            this.ciLow = ciLow;
            this.ciHigh = ciHigh;
            this.significantAtAlpha = significantAtAlpha;
            this.alpha = alpha;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "PropTest ctrl=%.6f treat=%.6f delta=%.6f z=%.4f p=%.6f sig=%s",
                    controlRate, treatmentRate, absoluteDelta, zStatistic, pValue, significantAtAlpha);
        }
    }

    // ---- Welch t-test -------------------------------------------------------

    /**
     * Welch's t-test (unequal variance) for two independent samples.
     * Preferred over Student t when sample sizes / variances differ — common
     * in online experiments with uneven traffic.
     *
     * @param control    control observations
     * @param treatment  treatment observations
     * @param alpha      significance level (e.g. 0.05)
     */
    public static MeanTestResult welchTTest(double[] control, double[] treatment, double alpha) {
        Objects.requireNonNull(control, "control");
        Objects.requireNonNull(treatment, "treatment");
        if (control.length < 2 || treatment.length < 2) {
            throw new IllegalArgumentException("need at least 2 observations per group");
        }
        Summary c = summarize(control);
        Summary t = summarize(treatment);
        double se = Math.sqrt(c.variance / c.n + t.variance / t.n);
        if (se == 0.0) {
            double delta = t.mean - c.mean;
            return new MeanTestResult(c.mean, t.mean, delta, rel(c.mean, delta),
                    0.0, c.n + t.n - 2, 1.0, delta, delta, c.n, t.n, false, alpha, "welch");
        }
        double tStat = (t.mean - c.mean) / se;
        // Welch–Satterthwaite degrees of freedom
        double num = c.variance / c.n + t.variance / t.n;
        num = num * num;
        double den = (c.variance / c.n) * (c.variance / c.n) / (c.n - 1)
                + (t.variance / t.n) * (t.variance / t.n) / (t.n - 1);
        double df = den == 0.0 ? 1.0 : num / den;
        double p = 2.0 * studentTCdf(-Math.abs(tStat), df);
        double tCrit = studentTCritical(1.0 - alpha / 2.0, df);
        double margin = tCrit * se;
        double delta = t.mean - c.mean;
        return new MeanTestResult(
                c.mean, t.mean, delta, rel(c.mean, delta),
                tStat, df, p, delta - margin, delta + margin,
                c.n, t.n, p < alpha, alpha, "welch");
    }

    /**
     * Aggregate form of Welch t-test from sufficient statistics
     * (n, mean, variance) — used by online metric collectors.
     */
    public static MeanTestResult welchTTestFromStats(
            long nC, double meanC, double varC,
            long nT, double meanT, double varT,
            double alpha) {
        if (nC < 2 || nT < 2) {
            throw new IllegalArgumentException("need n >= 2 per group");
        }
        double se = Math.sqrt(varC / nC + varT / nT);
        if (se == 0.0) {
            double delta = meanT - meanC;
            return new MeanTestResult(meanC, meanT, delta, rel(meanC, delta),
                    0.0, nC + nT - 2, 1.0, delta, delta, nC, nT, false, alpha, "welch");
        }
        double tStat = (meanT - meanC) / se;
        double num = varC / nC + varT / nT;
        num = num * num;
        double den = (varC / nC) * (varC / nC) / (nC - 1)
                + (varT / nT) * (varT / nT) / (nT - 1);
        double df = den == 0.0 ? 1.0 : num / den;
        double p = 2.0 * studentTCdf(-Math.abs(tStat), df);
        double tCrit = studentTCritical(1.0 - alpha / 2.0, df);
        double margin = tCrit * se;
        double delta = meanT - meanC;
        return new MeanTestResult(
                meanC, meanT, delta, rel(meanC, delta),
                tStat, df, p, delta - margin, delta + margin,
                nC, nT, p < alpha, alpha, "welch");
    }

    // ---- two-proportion z-test ----------------------------------------------

    /**
     * Two-proportion z-test (pooled) for conversion / CTR style metrics.
     *
     * @param successC control successes
     * @param nC       control trials
     * @param successT treatment successes
     * @param nT       treatment trials
     * @param alpha    significance level
     */
    public static ProportionTestResult twoProportionZTest(
            long successC, long nC, long successT, long nT, double alpha) {
        if (nC <= 0 || nT <= 0) {
            throw new IllegalArgumentException("n must be > 0");
        }
        double pC = successC / (double) nC;
        double pT = successT / (double) nT;
        double pPool = (successC + successT) / (double) (nC + nT);
        double se = Math.sqrt(pPool * (1.0 - pPool) * (1.0 / nC + 1.0 / nT));
        double z = se == 0.0 ? 0.0 : (pT - pC) / se;
        double pValue = 2.0 * (1.0 - standardNormalCdf(Math.abs(z)));
        // Unpooled CI for difference
        double seDiff = Math.sqrt(pC * (1.0 - pC) / nC + pT * (1.0 - pT) / nT);
        double zCrit = standardNormalCritical(1.0 - alpha / 2.0);
        double delta = pT - pC;
        return new ProportionTestResult(
                pC, pT, delta, rel(pC, delta), z, pValue,
                delta - zCrit * seDiff, delta + zCrit * seDiff,
                pValue < alpha, alpha);
    }

    // ---- SRM (Sample Ratio Mismatch) ----------------------------------------

    /**
     * Chi-square goodness-of-fit SRM check.
     *
     * <p>Industry rule of thumb (Kohavi / Microsoft / Meta):
     * flag SRM if p-value &lt; 0.001 (stricter than 0.05 because SRM is a
     * system-integrity test, not a product metric).
     *
     * @param observed counts per variant (same order as expectedRatio)
     * @param expectedRatio expected traffic shares (need not sum to 1; will normalize)
     * @param alpha typically 0.001
     */
    public static SrmResult srmTest(long[] observed, double[] expectedRatio, double alpha) {
        Objects.requireNonNull(observed, "observed");
        Objects.requireNonNull(expectedRatio, "expectedRatio");
        if (observed.length != expectedRatio.length || observed.length < 2) {
            throw new IllegalArgumentException("observed and expectedRatio length mismatch / < 2");
        }
        long total = 0L;
        for (long o : observed) {
            if (o < 0) throw new IllegalArgumentException("negative count");
            total += o;
        }
        if (total == 0L) {
            throw new IllegalArgumentException("total observed count is 0");
        }
        double ratioSum = 0.0;
        for (double r : expectedRatio) {
            if (r < 0) throw new IllegalArgumentException("negative ratio");
            ratioSum += r;
        }
        if (ratioSum <= 0.0) {
            throw new IllegalArgumentException("expectedRatio sum must be > 0");
        }
        double[] expected = new double[observed.length];
        double chi2 = 0.0;
        for (int i = 0; i < observed.length; i++) {
            expected[i] = total * (expectedRatio[i] / ratioSum);
            if (expected[i] > 0.0) {
                double diff = observed[i] - expected[i];
                chi2 += diff * diff / expected[i];
            }
        }
        double df = observed.length - 1.0;
        double pValue = chiSquareSf(chi2, df);
        return new SrmResult(Arrays.copyOf(observed, observed.length), expected,
                chi2, df, pValue, pValue < alpha, alpha);
    }

    /**
     * Convenience SRM for equal-weight variants.
     */
    public static SrmResult srmEqualWeight(long[] observed, double alpha) {
        double[] ratio = new double[observed.length];
        Arrays.fill(ratio, 1.0);
        return srmTest(observed, ratio, alpha);
    }

    // ---- CUPED --------------------------------------------------------------

    /**
     * CUPED variance-reduced metric (Deng, Xu, Kohavi, Walker, KDD 2013).
     *
     * <pre>
     *   Y_cuped = Y - θ * (X - E[X])
     *   θ = Cov(Y, X) / Var(X)
     * </pre>
     *
     * where X is the pre-experiment covariate (same metric in pre-period).
     *
     * @param y outcome during experiment
     * @param x pre-experiment covariate (same length, aligned by unit)
     * @return CUPED-adjusted outcomes (new array)
     */
    public static double[] cupedAdjust(double[] y, double[] x) {
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(x, "x");
        if (y.length != x.length || y.length == 0) {
            throw new IllegalArgumentException("y and x must have same non-zero length");
        }
        Summary ys = summarize(y);
        Summary xs = summarize(x);
        double cov = 0.0;
        for (int i = 0; i < y.length; i++) {
            cov += (y[i] - ys.mean) * (x[i] - xs.mean);
        }
        cov /= (y.length - 1.0);
        double theta = xs.variance == 0.0 ? 0.0 : cov / xs.variance;
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = y[i] - theta * (x[i] - xs.mean);
        }
        return out;
    }

    /**
     * Full CUPED pipeline: adjust both arms with a shared theta estimated on
     * pooled data, then run Welch t-test on adjusted outcomes.
     *
     * <p>Shared theta (pooled) is the standard unbiased approach used by
     * Microsoft ExP / many production platforms.
     */
    public static MeanTestResult cupedWelchTTest(
            double[] controlY, double[] controlX,
            double[] treatmentY, double[] treatmentX,
            double alpha) {
        if (controlY.length != controlX.length || treatmentY.length != treatmentX.length) {
            throw new IllegalArgumentException("Y/X length mismatch");
        }
        int nC = controlY.length;
        int nT = treatmentY.length;
        int n = nC + nT;
        double[] yAll = new double[n];
        double[] xAll = new double[n];
        System.arraycopy(controlY, 0, yAll, 0, nC);
        System.arraycopy(treatmentY, 0, yAll, nC, nT);
        System.arraycopy(controlX, 0, xAll, 0, nC);
        System.arraycopy(treatmentX, 0, xAll, nC, nT);

        Summary ys = summarize(yAll);
        Summary xs = summarize(xAll);
        double cov = 0.0;
        for (int i = 0; i < n; i++) {
            cov += (yAll[i] - ys.mean) * (xAll[i] - xs.mean);
        }
        cov /= (n - 1.0);
        double theta = xs.variance == 0.0 ? 0.0 : cov / xs.variance;

        double[] cAdj = new double[nC];
        double[] tAdj = new double[nT];
        for (int i = 0; i < nC; i++) {
            cAdj[i] = controlY[i] - theta * (controlX[i] - xs.mean);
        }
        for (int i = 0; i < nT; i++) {
            tAdj[i] = treatmentY[i] - theta * (treatmentX[i] - xs.mean);
        }
        MeanTestResult base = welchTTest(cAdj, tAdj, alpha);
        return new MeanTestResult(
                base.controlMean, base.treatmentMean, base.absoluteDelta, base.relativeDelta,
                base.tStatistic, base.degreesOfFreedom, base.pValue,
                base.ciLow, base.ciHigh, base.controlN, base.treatmentN,
                base.significantAtAlpha, base.alpha, "cuped+welch");
    }

    // ---- sample size / MDE --------------------------------------------------

    /**
     * Approximate per-arm sample size for two-sample t-test (equal n).
     *
     * <pre>
     *   n = 2 * (z_{1-α/2} + z_{1-β})^2 * σ^2 / δ^2
     * </pre>
     *
     * @param sigma        pooled std of metric
     * @param mde          minimum detectable absolute effect
     * @param alpha        type-I error (e.g. 0.05)
     * @param power        1 - beta (e.g. 0.8)
     * @return required sample size per arm
     */
    public static long sampleSizePerArm(double sigma, double mde, double alpha, double power) {
        if (sigma <= 0.0 || mde <= 0.0) {
            throw new IllegalArgumentException("sigma and mde must be > 0");
        }
        if (alpha <= 0.0 || alpha >= 1.0 || power <= 0.0 || power >= 1.0) {
            throw new IllegalArgumentException("alpha/power out of range");
        }
        double zAlpha = standardNormalCritical(1.0 - alpha / 2.0);
        double zBeta = standardNormalCritical(power);
        double n = 2.0 * (zAlpha + zBeta) * (zAlpha + zBeta) * sigma * sigma / (mde * mde);
        return (long) Math.ceil(n);
    }

    /**
     * Approximate per-arm sample size for two-proportion test.
     *
     * @param pBaseline baseline rate
     * @param mde       absolute lift to detect
     * @param alpha     type-I
     * @param power     1-beta
     */
    public static long sampleSizePerArmProportion(double pBaseline, double mde, double alpha, double power) {
        if (pBaseline <= 0.0 || pBaseline >= 1.0 || mde <= 0.0) {
            throw new IllegalArgumentException("invalid pBaseline/mde");
        }
        double p2 = clamp01(pBaseline + mde);
        double zAlpha = standardNormalCritical(1.0 - alpha / 2.0);
        double zBeta = standardNormalCritical(power);
        double pBar = (pBaseline + p2) / 2.0;
        double num = zAlpha * Math.sqrt(2.0 * pBar * (1.0 - pBar))
                + zBeta * Math.sqrt(pBaseline * (1.0 - pBaseline) + p2 * (1.0 - p2));
        double n = (num * num) / (mde * mde);
        return (long) Math.ceil(n);
    }

    // ---- numeric helpers (normal / t / chi2) --------------------------------

    private static final class Summary {
        final long n;
        final double mean;
        final double variance;

        Summary(long n, double mean, double variance) {
            this.n = n;
            this.mean = mean;
            this.variance = variance;
        }
    }

    private static Summary summarize(double[] xs) {
        long n = xs.length;
        double mean = 0.0;
        for (double x : xs) mean += x;
        mean /= n;
        double var = 0.0;
        for (double x : xs) {
            double d = x - mean;
            var += d * d;
        }
        var /= (n - 1.0);
        return new Summary(n, mean, var);
    }

    private static double rel(double base, double delta) {
        if (base == 0.0) {
            return delta == 0.0 ? 0.0 : Double.POSITIVE_INFINITY;
        }
        return delta / base;
    }

    private static double clamp01(double x) {
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
    }

    /** Standard normal CDF via Abramowitz & Stegun 7.1.26 erf approximation. */
    public static double standardNormalCdf(double z) {
        return 0.5 * (1.0 + erf(z / Math.sqrt(2.0)));
    }

    public static double standardNormalCritical(double p) {
        // Inverse CDF via rational approximation (Acklam).
        return normsinv(p);
    }

    /**
     * Regularized incomplete gamma upper for chi-square survival function:
     * P(Chi2_df > x) = gammaincc(df/2, x/2).
     */
    public static double chiSquareSf(double x, double df) {
        if (x <= 0.0) return 1.0;
        if (df <= 0.0) return Double.NaN;
        return gammaincc(df / 2.0, x / 2.0);
    }

    /** Two-sided-ready student-t CDF (lower tail). */
    public static double studentTCdf(double t, double df) {
        if (df <= 0.0) return Double.NaN;
        // t CDF via regularized incomplete beta
        double x = df / (df + t * t);
        double a = df / 2.0;
        double b = 0.5;
        double ib = regularizedIncompleteBeta(a, b, x);
        if (t >= 0.0) {
            return 1.0 - 0.5 * ib;
        } else {
            return 0.5 * ib;
        }
    }

    public static double studentTCritical(double p, double df) {
        // Binary search inverse CDF
        if (p <= 0.0 || p >= 1.0) {
            throw new IllegalArgumentException("p must be in (0,1)");
        }
        if (p < 0.5) {
            return -studentTCritical(1.0 - p, df);
        }
        double lo = 0.0;
        double hi = 1.0;
        while (studentTCdf(hi, df) < p) {
            hi *= 2.0;
            if (hi > 1e6) break;
        }
        for (int i = 0; i < 80; i++) {
            double mid = 0.5 * (lo + hi);
            if (studentTCdf(mid, df) < p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }

    // ---- special functions --------------------------------------------------

    private static double erf(double x) {
        // A&S 7.1.26
        double ax = Math.abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * ax);
        double[] c = {0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429};
        double poly = 0.0;
        double u = t;
        for (double coeff : c) {
            poly += coeff * u;
            u *= t;
        }
        double result = 1.0 - poly * Math.exp(-ax * ax);
        return x >= 0 ? result : -result;
    }

    /** Peter J. Acklam's inverse normal CDF approximation. */
    private static double normsinv(double p) {
        if (p <= 0.0) return Double.NEGATIVE_INFINITY;
        if (p >= 1.0) return Double.POSITIVE_INFINITY;
        double a1 = -3.969683028665376e+01;
        double a2 = 2.209460984245205e+02;
        double a3 = -2.759285104469687e+02;
        double a4 = 1.383577518672690e+02;
        double a5 = -3.066479806614736e+01;
        double a6 = 2.506628277459239e+00;
        double b1 = -5.447609879822406e+01;
        double b2 = 1.615858368580409e+02;
        double b3 = -1.556989798598866e+02;
        double b4 = 6.680131188771972e+01;
        double b5 = -1.328068155288572e+01;
        double c1 = -7.784894002430293e-03;
        double c2 = -3.223964580411365e-01;
        double c3 = -2.400758277161838e+00;
        double c4 = -2.549732539343734e+00;
        double c5 = 4.374664141464968e+00;
        double c6 = 2.938163982698783e+00;
        double d1 = 7.784695709041462e-03;
        double d2 = 3.224671290700398e-01;
        double d3 = 2.445134137142996e+00;
        double d4 = 3.754408661907416e+00;
        double plow = 0.02425;
        double phigh = 1.0 - plow;
        double q, r;
        if (p < plow) {
            q = Math.sqrt(-2.0 * Math.log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        if (phigh < p) {
            q = Math.sqrt(-2.0 * Math.log(1.0 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        q = p - 0.5;
        r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }

    /** Regularized incomplete gamma Q(a,x) = 1 - P(a,x). */
    private static double gammaincc(double a, double x) {
        if (x < 0.0 || a <= 0.0) return Double.NaN;
        if (x == 0.0) return 1.0;
        if (x < a + 1.0) {
            return 1.0 - gammaincSeries(a, x);
        }
        return gammaincCf(a, x);
    }

    private static double gammaincSeries(double a, double x) {
        final int maxIter = 200;
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int n = 1; n <= maxIter; n++) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (Math.abs(del) < Math.abs(sum) * 1e-12) {
                break;
            }
        }
        return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
    }

    private static double gammaincCf(double a, double x) {
        final int maxIter = 200;
        double b = x + 1.0 - a;
        double c = 1.0 / 1e-30;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i <= maxIter; i++) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1.0) < 1e-12) {
                break;
            }
        }
        return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
    }

    /** Lanczos approximation for log-gamma. */
    private static double logGamma(double x) {
        double[] cof = {
                76.18009172947146, -86.50532032941677,
                24.01409824083091, -1.231739572450155,
                0.1208650973866179e-2, -0.539841384917814e-5
        };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++) {
            ser += cof[j] / ++y;
        }
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    /** Regularized incomplete beta I_x(a,b) via continued fraction. */
    private static double regularizedIncompleteBeta(double a, double b, double x) {
        if (x < 0.0 || x > 1.0) return Double.NaN;
        if (x == 0.0 || x == 1.0) return x;
        double lbeta = logGamma(a) + logGamma(b) - logGamma(a + b);
        double front = Math.exp(Math.log(x) * a + Math.log(1.0 - x) * b - lbeta);
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return front * betaCf(a, b, x) / a;
        }
        return 1.0 - front * betaCf(b, a, 1.0 - x) / b;
    }

    private static double betaCf(double a, double b, double x) {
        final int maxIter = 200;
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (Math.abs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= maxIter; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1.0) < 1e-12) {
                break;
            }
        }
        return h;
    }
}
