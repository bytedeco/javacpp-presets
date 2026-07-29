/*
 * Shadow serving (dark launch): mirror production requests to a candidate
 * model without exposing its results to users.
 *
 * Used by Meta, Google, Uber Michelangelo, Alibaba, ByteDance before canary:
 *   1. Sample N% of prod traffic
 *   2. Score with candidate model asynchronously
 *   3. Log score deltas / ranking disagreements vs prod
 *   4. Gate promotion on disagreement + latency budgets
 */
package org.bytedeco.pytorch.utils.recommend.modelops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiFunction;

/** Shadow traffic scorer and disagreement aggregator. */
public final class ShadowServing {

    /**
     * Scores a list of item ids for a user/request context key.
     * Returns map itemId -> score.
     */
    public interface ModelScorer {
        Map<String, Double> score(String requestKey, List<String> itemIds) throws Exception;
    }

    public static final class DisagreementStats {
        public final long samples;
        public final long scoreComparisons;
        public final double meanAbsScoreDelta;
        public final double maxAbsScoreDelta;
        public final long rankMismatches;
        public final double rankMismatchRate;
        public final long shadowErrors;
        public final double meanShadowLatencyMs;

        public DisagreementStats(
                long samples,
                long scoreComparisons,
                double meanAbsScoreDelta,
                double maxAbsScoreDelta,
                long rankMismatches,
                double rankMismatchRate,
                long shadowErrors,
                double meanShadowLatencyMs) {
            this.samples = samples;
            this.scoreComparisons = scoreComparisons;
            this.meanAbsScoreDelta = meanAbsScoreDelta;
            this.maxAbsScoreDelta = maxAbsScoreDelta;
            this.rankMismatches = rankMismatches;
            this.rankMismatchRate = rankMismatchRate;
            this.shadowErrors = shadowErrors;
            this.meanShadowLatencyMs = meanShadowLatencyMs;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "ShadowStats{n=%d comparisons=%d mean|dScore|=%.6f max|dScore|=%.6f "
                            + "rankMismatch=%.2f%% errors=%d shadowLatency=%.1fms}",
                    samples, scoreComparisons, meanAbsScoreDelta, maxAbsScoreDelta,
                    rankMismatchRate * 100.0, shadowErrors, meanShadowLatencyMs);
        }
    }

    private final String name;
    private final ModelScorer production;
    private final ModelScorer shadow;
    private final double sampleRate;
    private final ExecutorService executor;
    private final boolean ownsExecutor;

    private final LongAdder samples = new LongAdder();
    private final LongAdder comparisons = new LongAdder();
    private final LongAdder rankMismatches = new LongAdder();
    private final LongAdder shadowErrors = new LongAdder();
    private final LongAdder latencySumMs = new LongAdder();
    private final LongAdder latencyCount = new LongAdder();
    // Welford for abs delta
    private final Object statsLock = new Object();
    private long deltaN;
    private double deltaMean;
    private double deltaMax;

    public ShadowServing(String name, ModelScorer production, ModelScorer shadow, double sampleRate) {
        this(name, production, shadow, sampleRate, null);
    }

    public ShadowServing(
            String name,
            ModelScorer production,
            ModelScorer shadow,
            double sampleRate,
            ExecutorService executor) {
        if (sampleRate < 0.0 || sampleRate > 1.0) {
            throw new IllegalArgumentException("sampleRate in [0,1]");
        }
        this.name = Objects.requireNonNull(name);
        this.production = Objects.requireNonNull(production);
        this.shadow = Objects.requireNonNull(shadow);
        this.sampleRate = sampleRate;
        if (executor != null) {
            this.executor = executor;
            this.ownsExecutor = false;
        } else {
            this.executor = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "shadow-" + name);
                t.setDaemon(true);
                return t;
            });
            this.ownsExecutor = true;
        }
    }

    public String name() {
        return name;
    }

    /**
     * Score with production (returned to user) and optionally mirror to shadow.
     *
     * @return production scores (always)
     */
    public Map<String, Double> score(String requestKey, List<String> itemIds) throws Exception {
        Map<String, Double> prodScores = production.score(requestKey, itemIds);
        if (sampleRate > 0.0 && (sampleRate >= 1.0 || Math.random() < sampleRate)) {
            List<String> itemsCopy = new ArrayList<>(itemIds);
            Map<String, Double> prodCopy = new LinkedHashMap<>(prodScores);
            executor.execute(() -> runShadow(requestKey, itemsCopy, prodCopy));
        }
        return prodScores;
    }

    /**
     * Synchronous shadow compare — useful in tests / offline replay.
     */
    public DisagreementStats compareSync(String requestKey, List<String> itemIds) throws Exception {
        Map<String, Double> prod = production.score(requestKey, itemIds);
        long t0 = System.currentTimeMillis();
        Map<String, Double> sh;
        try {
            sh = shadow.score(requestKey, itemIds);
        } catch (Exception ex) {
            shadowErrors.increment();
            throw ex;
        }
        long latency = System.currentTimeMillis() - t0;
        recordComparison(prod, sh, latency);
        return stats();
    }

    private void runShadow(String requestKey, List<String> itemIds, Map<String, Double> prodScores) {
        long t0 = System.currentTimeMillis();
        try {
            Map<String, Double> sh = shadow.score(requestKey, itemIds);
            recordComparison(prodScores, sh, System.currentTimeMillis() - t0);
        } catch (Exception ex) {
            shadowErrors.increment();
        }
    }

    private void recordComparison(
            Map<String, Double> prod, Map<String, Double> sh, long latencyMs) {
        samples.increment();
        latencySumMs.add(latencyMs);
        latencyCount.increment();

        // Score deltas on intersection
        for (Map.Entry<String, Double> e : prod.entrySet()) {
            Double sv = sh.get(e.getKey());
            if (sv == null) continue;
            double abs = Math.abs(e.getValue() - sv);
            comparisons.increment();
            synchronized (statsLock) {
                deltaN++;
                deltaMean += (abs - deltaMean) / deltaN;
                if (abs > deltaMax) deltaMax = abs;
            }
        }

        // Top-1 rank mismatch
        String prodTop = topKey(prod);
        String shTop = topKey(sh);
        if (prodTop != null && shTop != null && !prodTop.equals(shTop)) {
            rankMismatches.increment();
        }
    }

    private static String topKey(Map<String, Double> scores) {
        String best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (Map.Entry<String, Double> e : scores.entrySet()) {
            if (e.getValue() > bestScore) {
                bestScore = e.getValue();
                best = e.getKey();
            }
        }
        return best;
    }

    public DisagreementStats stats() {
        long s = samples.sum();
        long c = comparisons.sum();
        long rm = rankMismatches.sum();
        long lc = latencyCount.sum();
        double meanLat = lc == 0 ? 0.0 : latencySumMs.sum() / (double) lc;
        double meanDelta;
        double maxDelta;
        synchronized (statsLock) {
            meanDelta = deltaMean;
            maxDelta = deltaMax;
        }
        double mismatchRate = s == 0 ? 0.0 : rm / (double) s;
        return new DisagreementStats(s, c, meanDelta, maxDelta, rm, mismatchRate,
                shadowErrors.sum(), meanLat);
    }

    /**
     * Gate: pass if rank mismatch rate and mean abs delta below thresholds
     * and shadow error rate acceptable.
     */
    public boolean passGate(double maxRankMismatchRate, double maxMeanAbsDelta, double maxErrorRate) {
        DisagreementStats st = stats();
        if (st.samples < 100) {
            return false; // not enough data
        }
        double errRate = st.shadowErrors / (double) st.samples;
        return st.rankMismatchRate <= maxRankMismatchRate
                && st.meanAbsScoreDelta <= maxMeanAbsDelta
                && errRate <= maxErrorRate;
    }

    public void shutdown() {
        if (ownsExecutor) {
            executor.shutdownNow();
        }
    }

    /** Wrap a bi-function scorer. */
    public static ModelScorer fromFunction(BiFunction<String, List<String>, Map<String, Double>> fn) {
        return fn::apply;
    }
}
