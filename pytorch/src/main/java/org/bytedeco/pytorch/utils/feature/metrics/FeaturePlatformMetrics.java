/*
 * In-process metrics for the feature platform (counters / timers).
 * Shape mirrors recommend.ops.MetricsRegistry — export to Prometheus later.
 */
package org.bytedeco.pytorch.utils.feature.metrics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/** Lightweight feature-platform metrics registry. */
public final class FeaturePlatformMetrics {

    public static final class Counter {
        private final String name;
        private final LongAdder adder = new LongAdder();

        Counter(String name) {
            this.name = name;
        }

        public void inc() {
            adder.increment();
        }

        public void add(long n) {
            adder.add(n);
        }

        public long get() {
            return adder.sum();
        }

        public String name() {
            return name;
        }
    }

    public static final class Timer {
        private final String name;
        private final LongAdder totalNanos = new LongAdder();
        private final LongAdder count = new LongAdder();
        private final AtomicLong maxNanos = new AtomicLong();

        Timer(String name) {
            this.name = name;
        }

        public void record(long nanos) {
            totalNanos.add(nanos);
            count.increment();
            maxNanos.accumulateAndGet(nanos, Math::max);
        }

        public long count() {
            return count.sum();
        }

        public long totalNanos() {
            return totalNanos.sum();
        }

        public double avgMs() {
            long c = count.sum();
            return c == 0 ? 0.0 : (totalNanos.sum() / 1_000_000.0) / c;
        }

        public double maxMs() {
            return maxNanos.get() / 1_000_000.0;
        }

        public String name() {
            return name;
        }
    }

    private final ConcurrentHashMap<String, Counter> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Timer> timers = new ConcurrentHashMap<>();

    public Counter counter(String name) {
        Objects.requireNonNull(name, "name");
        return counters.computeIfAbsent(name.toLowerCase(Locale.ROOT), Counter::new);
    }

    public Timer timer(String name) {
        Objects.requireNonNull(name, "name");
        return timers.computeIfAbsent(name.toLowerCase(Locale.ROOT), Timer::new);
    }

    public void inc(String name) {
        counter(name).inc();
    }

    public void record(String timerName, long nanos) {
        timer(timerName).record(nanos);
    }

    public Map<String, Long> snapshotCounters() {
        Map<String, Long> out = new LinkedHashMap<>();
        List<String> keys = new ArrayList<>(counters.keySet());
        Collections.sort(keys);
        for (String k : keys) out.put(k, counters.get(k).get());
        return out;
    }

    public Map<String, String> snapshotTimers() {
        Map<String, String> out = new LinkedHashMap<>();
        List<String> keys = new ArrayList<>(timers.keySet());
        Collections.sort(keys);
        for (String k : keys) {
            Timer t = timers.get(k);
            out.put(k, String.format(Locale.ROOT, "count=%d,avgMs=%.3f,maxMs=%.3f",
                    t.count(), t.avgMs(), t.maxMs()));
        }
        return out;
    }

    public void reset() {
        counters.clear();
        timers.clear();
    }

    // Canonical metric names
    public static final String REGISTRY_REGISTER = "feature.registry.register";
    public static final String MATERIALIZE_ROWS = "feature.materialize.rows_written";
    public static final String MATERIALIZE_LATENCY = "feature.materialize.latency";
    public static final String ONLINE_GET = "feature.online.get";
    public static final String ONLINE_LATENCY = "feature.online.latency";
    public static final String ONLINE_MISS = "feature.online.miss";
    public static final String PIT_JOIN_LATENCY = "feature.pit.latency";
    public static final String PIT_FUTURE_REJECT = "feature.pit.future_rejected";

    @Override
    public String toString() {
        return "FeaturePlatformMetrics{counters=" + snapshotCounters()
                + ", timers=" + snapshotTimers() + "}";
    }
}
