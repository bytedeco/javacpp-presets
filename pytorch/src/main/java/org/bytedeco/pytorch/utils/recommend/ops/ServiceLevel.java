/*
 * SLI / SLO / error-budget helpers (Google SRE style) for recsys services.
 *
 * Typical recommendation SLOs:
 *   - Availability: successful non-5xx responses / total >= 99.9%
 *   - Latency: p99 end-to-end < 200ms (feed) or < 100ms (ads ranker)
 *   - Empty-rate: empty recall responses / total < 0.5%
 *   - Freshness: model feature lag < N minutes (separate pipeline SLO)
 */
package org.bytedeco.pytorch.utils.recommend.ops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

/** Service level indicator + objective tracking. */
public final class ServiceLevel {

    public enum SliType {
        AVAILABILITY,
        LATENCY,
        EMPTY_RATE,
        CUSTOM_RATIO
    }

    /** Definition of an SLO. */
    public static final class SloDefinition {
        public final String name;
        public final SliType type;
        /** Target ratio in [0,1] for availability-like, or max latency ms for LATENCY. */
        public final double objective;
        /** Window length for error budget calc (ms). */
        public final long windowMs;

        public SloDefinition(String name, SliType type, double objective, long windowMs) {
            this.name = Objects.requireNonNull(name);
            this.type = Objects.requireNonNull(type);
            this.objective = objective;
            this.windowMs = windowMs;
        }

        public static SloDefinition availability(String name, double objective, long windowMs) {
            return new SloDefinition(name, SliType.AVAILABILITY, objective, windowMs);
        }

        public static SloDefinition latencyP99(String name, double maxP99Ms, long windowMs) {
            return new SloDefinition(name, SliType.LATENCY, maxP99Ms, windowMs);
        }

        public static SloDefinition emptyRate(String name, double maxEmptyRatio, long windowMs) {
            return new SloDefinition(name, SliType.EMPTY_RATE, maxEmptyRatio, windowMs);
        }
    }

    /** Rolling window counters. */
    public static final class SliWindow {
        public final SloDefinition def;
        private final LongAdder good = new LongAdder();
        private final LongAdder total = new LongAdder();
        private final LongAdder latencySum = new LongAdder();
        private final LongAdder latencyCount = new LongAdder();
        private final MetricsRegistry.Timer latencyTimer;
        private final long startedAtMs = System.currentTimeMillis();

        public SliWindow(SloDefinition def) {
            this.def = def;
            this.latencyTimer = new MetricsRegistry.Timer(def.name + ".latency");
        }

        public void recordSuccess() {
            good.increment();
            total.increment();
        }

        public void recordFailure() {
            total.increment();
        }

        public void recordLatency(long ms) {
            latencySum.add(ms);
            latencyCount.increment();
            latencyTimer.record(ms);
            // count toward total for latency SLI evaluation samples
            total.increment();
            if (def.type == SliType.LATENCY) {
                if (ms <= def.objective) {
                    good.increment();
                }
            }
        }

        public void recordEmpty(boolean empty) {
            total.increment();
            if (!empty) {
                good.increment();
            }
        }

        public long good() {
            return good.sum();
        }

        public long total() {
            return total.sum();
        }

        public double ratio() {
            long t = total.sum();
            return t == 0 ? 1.0 : good.sum() / (double) t;
        }

        public double p99Latency() {
            return latencyTimer.percentile(0.99);
        }

        public SloStatus status() {
            double sli;
            boolean met;
            switch (def.type) {
                case LATENCY:
                    sli = p99Latency();
                    met = sli <= def.objective || latencyCount.sum() == 0;
                    break;
                case EMPTY_RATE:
                    // objective is max empty ratio; sli = empty ratio = 1 - good/total
                    sli = 1.0 - ratio();
                    met = sli <= def.objective;
                    break;
                case AVAILABILITY:
                case CUSTOM_RATIO:
                default:
                    sli = ratio();
                    met = sli >= def.objective;
                    break;
            }
            double errorBudgetRemaining = errorBudgetRemaining(met, sli);
            return new SloStatus(def.name, def.type, sli, def.objective, met, errorBudgetRemaining,
                    total.sum(), System.currentTimeMillis() - startedAtMs);
        }

        private double errorBudgetRemaining(boolean met, double sli) {
            // For availability: budget = 1 - objective; consumed = 1 - sli; remaining = 1 - consumed/budget
            if (def.type == SliType.AVAILABILITY || def.type == SliType.CUSTOM_RATIO) {
                double budget = 1.0 - def.objective;
                if (budget <= 0) return met ? 1.0 : 0.0;
                double consumed = Math.max(0.0, 1.0 - sli);
                return Math.max(0.0, 1.0 - consumed / budget);
            }
            if (def.type == SliType.EMPTY_RATE) {
                if (def.objective <= 0) return met ? 1.0 : 0.0;
                return Math.max(0.0, 1.0 - sli / def.objective);
            }
            // latency: rough budget on how far p99 is above objective
            if (def.objective <= 0) return met ? 1.0 : 0.0;
            if (sli <= def.objective) return 1.0;
            return Math.max(0.0, 1.0 - (sli - def.objective) / def.objective);
        }
    }

    public static final class SloStatus {
        public final String name;
        public final SliType type;
        public final double sliValue;
        public final double objective;
        public final boolean met;
        public final double errorBudgetRemaining;
        public final long sampleCount;
        public final long windowAgeMs;

        public SloStatus(
                String name,
                SliType type,
                double sliValue,
                double objective,
                boolean met,
                double errorBudgetRemaining,
                long sampleCount,
                long windowAgeMs) {
            this.name = name;
            this.type = type;
            this.sliValue = sliValue;
            this.objective = objective;
            this.met = met;
            this.errorBudgetRemaining = errorBudgetRemaining;
            this.sampleCount = sampleCount;
            this.windowAgeMs = windowAgeMs;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "SLO[%s] type=%s sli=%.6f obj=%.6f met=%s budgetLeft=%.2f%% n=%d",
                    name, type, sliValue, objective, met, errorBudgetRemaining * 100.0, sampleCount);
        }
    }

    private final Map<String, SliWindow> windows = new LinkedHashMap<>();

    public synchronized SliWindow register(SloDefinition def) {
        SliWindow w = new SliWindow(def);
        windows.put(def.name, w);
        return w;
    }

    public SliWindow get(String name) {
        return windows.get(name);
    }

    public synchronized List<SloStatus> evaluateAll() {
        List<SloStatus> list = new ArrayList<>();
        for (SliWindow w : windows.values()) {
            list.add(w.status());
        }
        return list;
    }

    public synchronized boolean allMet() {
        for (SliWindow w : windows.values()) {
            if (!w.status().met) return false;
        }
        return true;
    }

    /** Standard recsys SLO pack. */
    public static ServiceLevel standardRecsys() {
        ServiceLevel sl = new ServiceLevel();
        sl.register(SloDefinition.availability("availability", 0.999, TimeUnit.DAYS.toMillis(30)));
        sl.register(SloDefinition.latencyP99("latency_p99", 200.0, TimeUnit.DAYS.toMillis(30)));
        sl.register(SloDefinition.emptyRate("empty_rate", 0.005, TimeUnit.DAYS.toMillis(30)));
        return sl;
    }
}
