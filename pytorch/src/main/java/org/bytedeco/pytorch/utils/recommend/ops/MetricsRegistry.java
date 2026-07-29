/*
 * Runtime ops: metrics, SLO/SLI, health, inspection, circuit breaker,
 * degradation, fallback, rate limiting — for recommendation online services.
 *
 * Industry alignment:
 *   - Google SRE workbook (SLI/SLO/error budget)
 *   - Netflix Hystrix / resilience4j patterns
 *   - Alibaba Sentinel (flow control, degradation, system protect)
 *   - ByteDance / Tencent recsys multi-level fallback (策略降级)
 *   - Meta / Google service health + canary abort metrics
 */
package org.bytedeco.pytorch.utils.recommend.ops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Lightweight in-process metrics registry (counters, gauges, timers).
 * Production would export to Prometheus / StatsD / OpenTelemetry; this
 * encodes the metric *names and aggregation* recsys services typically emit.
 */
public final class MetricsRegistry {

    public static final class Counter {
        private final LongAdder adder = new LongAdder();
        private final String name;

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

    public static final class Gauge {
        private final AtomicLong value = new AtomicLong();
        private final String name;

        Gauge(String name) {
            this.name = name;
        }

        public void set(long v) {
            value.set(v);
        }

        public void setDouble(double v) {
            value.set(Double.doubleToLongBits(v));
        }

        public long get() {
            return value.get();
        }

        public double getDouble() {
            return Double.longBitsToDouble(value.get());
        }

        public String name() {
            return name;
        }
    }

    /**
     * HDR-lite latency recorder: fixed bell buckets in ms for p50/p90/p99 approx.
     */
    public static final class Timer {
        private final String name;
        private final LongAdder count = new LongAdder();
        private final LongAdder totalMs = new LongAdder();
        // buckets: 1,2,5,10,20,50,100,200,500,1000,2000,5000,+Inf
        private static final long[] BOUNDS = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
        private final LongAdder[] buckets = new LongAdder[BOUNDS.length + 1];

        Timer(String name) {
            this.name = name;
            for (int i = 0; i < buckets.length; i++) {
                buckets[i] = new LongAdder();
            }
        }

        public void record(long latencyMs) {
            if (latencyMs < 0) latencyMs = 0;
            count.increment();
            totalMs.add(latencyMs);
            int idx = BOUNDS.length;
            for (int i = 0; i < BOUNDS.length; i++) {
                if (latencyMs <= BOUNDS[i]) {
                    idx = i;
                    break;
                }
            }
            buckets[idx].increment();
        }

        public long count() {
            return count.sum();
        }

        public double mean() {
            long c = count.sum();
            return c == 0 ? 0.0 : totalMs.sum() / (double) c;
        }

        public double percentile(double p) {
            if (p <= 0 || p > 1) throw new IllegalArgumentException("p in (0,1]");
            long c = count.sum();
            if (c == 0) return 0.0;
            long target = (long) Math.ceil(c * p);
            long acc = 0;
            for (int i = 0; i < buckets.length; i++) {
                acc += buckets[i].sum();
                if (acc >= target) {
                    return i < BOUNDS.length ? BOUNDS[i] : BOUNDS[BOUNDS.length - 1] * 2.0;
                }
            }
            return BOUNDS[BOUNDS.length - 1] * 2.0;
        }

        public String name() {
            return name;
        }
    }

    private final ConcurrentHashMap<String, Counter> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Gauge> gauges = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Timer> timers = new ConcurrentHashMap<>();

    public Counter counter(String name) {
        return counters.computeIfAbsent(name, Counter::new);
    }

    public Gauge gauge(String name) {
        return gauges.computeIfAbsent(name, Gauge::new);
    }

    public Timer timer(String name) {
        return timers.computeIfAbsent(name, Timer::new);
    }

    /** Standard recsys metric names. */
    public static final class Names {
        public static final String REQUESTS = "recsys.requests";
        public static final String ERRORS = "recsys.errors";
        public static final String EMPTY_RESPONSES = "recsys.empty_responses";
        public static final String DEGRADED = "recsys.degraded";
        public static final String LATENCY = "recsys.latency_ms";
        public static final String STAGE_LATENCY_PREFIX = "recsys.stage.latency_ms.";
        public static final String CIRCUIT_OPEN = "recsys.circuit_open";
        public static final String FALLBACK = "recsys.fallback";
        public static final String QPS = "recsys.qps";

        private Names() {}
    }

    public Map<String, Long> snapshotCounters() {
        Map<String, Long> m = new LinkedHashMap<>();
        for (Map.Entry<String, Counter> e : counters.entrySet()) {
            m.put(e.getKey(), e.getValue().get());
        }
        return m;
    }

    public Map<String, Double> snapshotTimerMeans() {
        Map<String, Double> m = new LinkedHashMap<>();
        for (Map.Entry<String, Timer> e : timers.entrySet()) {
            m.put(e.getKey() + ".mean", e.getValue().mean());
            m.put(e.getKey() + ".p99", e.getValue().percentile(0.99));
            m.put(e.getKey() + ".count", (double) e.getValue().count());
        }
        return m;
    }
}
