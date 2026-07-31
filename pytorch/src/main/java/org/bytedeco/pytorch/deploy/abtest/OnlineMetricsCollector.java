/*
 * Streaming online metric collector for experiment arms.
 *
 * Production systems (Meta, ByteDance, Alibaba) typically:
 *   1. Join exposure log with action log (click / convert / dwell / pay)
 *   2. Aggregate per (experiment, variant) with HyperLogLog / sketch for UV
 *   3. Maintain sufficient statistics (n, sum, sumsq) for t-tests
 *   4. Emit to real-time dashboard (Druid / ClickHouse / Prometheus)
 *
 * This class is an in-process Welford accumulator suitable for:
 *   - unit tests / local simulation
 *   - single-node streaming join
 *   - embedding into a Flink / Spark map-side aggregator as reference logic
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Thread-safe per-variant metric accumulator.
 *
 * <p>Key layout: {@code experimentId -> variantId -> metricName -> Stats}.
 */
public final class OnlineMetricsCollector {

    private final ConcurrentHashMap<String, ConcurrentHashMap<String, ConcurrentHashMap<String, Stats>>>
            data = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, AtomicLong>> exposureCounts =
            new ConcurrentHashMap<>();

    /**
     * Record that a unit was exposed to a variant (for SRM denominators).
     */
    public void recordExposure(String experimentId, String variantId) {
        exposureCounts
                .computeIfAbsent(experimentId, k -> new ConcurrentHashMap<>())
                .computeIfAbsent(variantId, k -> new AtomicLong())
                .incrementAndGet();
    }

    /**
     * Observe a scalar metric value for one unit in a variant.
     * Uses Welford online mean / variance (numerically stable).
     */
    public void observe(String experimentId, String variantId, String metric, double value) {
        Objects.requireNonNull(experimentId, "experimentId");
        Objects.requireNonNull(variantId, "variantId");
        Objects.requireNonNull(metric, "metric");
        Stats stats = data
                .computeIfAbsent(experimentId, k -> new ConcurrentHashMap<>())
                .computeIfAbsent(variantId, k -> new ConcurrentHashMap<>())
                .computeIfAbsent(metric, k -> new Stats());
        stats.accept(value);
    }

    /**
     * Observe a binary outcome (click=1/0, convert=1/0) — stored as scalar.
     */
    public void observeBinary(String experimentId, String variantId, String metric, boolean success) {
        observe(experimentId, variantId, metric, success ? 1.0 : 0.0);
    }

    public long exposureCount(String experimentId, String variantId) {
        ConcurrentHashMap<String, AtomicLong> m = exposureCounts.get(experimentId);
        if (m == null) return 0L;
        AtomicLong c = m.get(variantId);
        return c == null ? 0L : c.get();
    }

    public Map<String, Long> exposureCounts(String experimentId) {
        ConcurrentHashMap<String, AtomicLong> m = exposureCounts.get(experimentId);
        if (m == null) return Collections.emptyMap();
        Map<String, Long> out = new LinkedHashMap<>();
        for (Map.Entry<String, AtomicLong> e : m.entrySet()) {
            out.put(e.getKey(), e.getValue().get());
        }
        return out;
    }

    public StatsSnapshot stats(String experimentId, String variantId, String metric) {
        ConcurrentHashMap<String, ConcurrentHashMap<String, Stats>> byVar = data.get(experimentId);
        if (byVar == null) return StatsSnapshot.empty();
        ConcurrentHashMap<String, Stats> byMetric = byVar.get(variantId);
        if (byMetric == null) return StatsSnapshot.empty();
        Stats s = byMetric.get(metric);
        return s == null ? StatsSnapshot.empty() : s.snapshot();
    }

    /**
     * Build a {@link Guardrail.ExperimentSnapshot} for control vs one treatment.
     */
    public Guardrail.ExperimentSnapshot guardrailSnapshot(
            String experimentId,
            String controlVariantId,
            String treatmentVariantId,
            Iterable<String> metricKeys,
            Double srmPValue) {
        long nC = exposureCount(experimentId, controlVariantId);
        long nT = exposureCount(experimentId, treatmentVariantId);
        Map<String, Guardrail.MetricArmPair> metrics = new LinkedHashMap<>();
        if (metricKeys != null) {
            for (String key : metricKeys) {
                StatsSnapshot c = stats(experimentId, controlVariantId, key);
                StatsSnapshot t = stats(experimentId, treatmentVariantId, key);
                Double p = null;
                if (c.n >= 2 && t.n >= 2) {
                    try {
                        StatisticalTest.MeanTestResult r = StatisticalTest.welchTTestFromStats(
                                c.n, c.mean, c.variance, t.n, t.mean, t.variance, 0.05);
                        p = r.pValue;
                    } catch (RuntimeException ignored) {
                        // keep p null
                    }
                }
                metrics.put(key, new Guardrail.MetricArmPair(c.mean, t.mean, p));
            }
        }
        return new Guardrail.ExperimentSnapshot(experimentId, nC, nT, srmPValue, metrics);
    }

    /**
     * Run SRM using exposure counts and expected weights from the experiment.
     */
    public StatisticalTest.SrmResult srm(Experiment experiment, double alpha) {
        Objects.requireNonNull(experiment, "experiment");
        long[] observed = new long[experiment.variants().size()];
        double[] weights = new double[experiment.variants().size()];
        int i = 0;
        for (Variant v : experiment.variants()) {
            observed[i] = exposureCount(experiment.id(), v.id());
            weights[i] = v.trafficWeight();
            i++;
        }
        // Scale expected by experiment traffic is unnecessary: ratios among variants matter.
        return StatisticalTest.srmTest(observed, weights, alpha);
    }

    /**
     * Compare control vs treatment on one metric via Welch t-test on aggregates.
     */
    public StatisticalTest.MeanTestResult compareMean(
            String experimentId,
            String controlVariantId,
            String treatmentVariantId,
            String metric,
            double alpha) {
        StatsSnapshot c = stats(experimentId, controlVariantId, metric);
        StatsSnapshot t = stats(experimentId, treatmentVariantId, metric);
        return StatisticalTest.welchTTestFromStats(
                c.n, c.mean, c.variance, t.n, t.mean, t.variance, alpha);
    }

    /**
     * Compare binary rate metric via two-proportion z-test.
     * Interprets mean as rate and n as trials (requires values in {0,1}).
     */
    public StatisticalTest.ProportionTestResult compareRate(
            String experimentId,
            String controlVariantId,
            String treatmentVariantId,
            String metric,
            double alpha) {
        StatsSnapshot c = stats(experimentId, controlVariantId, metric);
        StatsSnapshot t = stats(experimentId, treatmentVariantId, metric);
        long successC = Math.round(c.mean * c.n);
        long successT = Math.round(t.mean * t.n);
        return StatisticalTest.twoProportionZTest(successC, c.n, successT, t.n, alpha);
    }

    public void reset(String experimentId) {
        data.remove(experimentId);
        exposureCounts.remove(experimentId);
    }

    public void resetAll() {
        data.clear();
        exposureCounts.clear();
    }

    // ---- Welford stats ------------------------------------------------------

    /** Mutable thread-safe Welford accumulator. */
    public static final class Stats {
        private final ReentrantLock lock = new ReentrantLock();
        private long n;
        private double mean;
        private double m2;
        private double sum;

        void accept(double x) {
            lock.lock();
            try {
                n += 1;
                sum += x;
                double delta = x - mean;
                mean += delta / n;
                double delta2 = x - mean;
                m2 += delta * delta2;
            } finally {
                lock.unlock();
            }
        }

        StatsSnapshot snapshot() {
            lock.lock();
            try {
                double var = n > 1 ? m2 / (n - 1.0) : 0.0;
                return new StatsSnapshot(n, mean, var, sum);
            } finally {
                lock.unlock();
            }
        }
    }

    /** Immutable stats view. */
    public static final class StatsSnapshot {
        public final long n;
        public final double mean;
        public final double variance;
        public final double sum;

        public StatsSnapshot(long n, double mean, double variance, double sum) {
            this.n = n;
            this.mean = mean;
            this.variance = variance;
            this.sum = sum;
        }

        public static StatsSnapshot empty() {
            return new StatsSnapshot(0L, 0.0, 0.0, 0.0);
        }

        public double std() {
            return Math.sqrt(variance);
        }

        @Override
        public String toString() {
            return "Stats{n=" + n + ", mean=" + mean + ", var=" + variance + "}";
        }
    }
}
