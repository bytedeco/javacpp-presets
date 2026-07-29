/*
 * Multi-level degradation policy for recommendation services.
 *
 * When the system is under stress (high latency, dependency failures, CPU),
 * progressively shed non-critical work — a pattern universal across
 * Alibaba Sentinel, ByteDance, Tencent, Meta, Google:
 *
 *   L0 NORMAL     — full cascade
 *   L1 SOFT       — disable expensive re-rank / some recall channels
 *   L2 HARD       — skip fine-rank, use coarse scores only
 *   L3 EMERGENCY  — hot-list / operations pool only
 *   L4 CIRCUIT    — static fallback page / empty with error code
 *
 * Policy is driven by signals (error rate, p99, CPU, dependency health)
 * and hysteresis to avoid flapping.
 */
package org.bytedeco.pytorch.utils.recommend.ops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/** Degradation level controller. */
public final class DegradationPolicy {

    public enum Level {
        L0_NORMAL(0, "full pipeline"),
        L1_SOFT(1, "disable heavy rerank / optional recall channels"),
        L2_HARD(2, "skip fine-rank; coarse scores only"),
        L3_EMERGENCY(3, "hot-list / ops pool only"),
        L4_CIRCUIT(4, "static fallback / fail closed");

        public final int severity;
        public final String description;

        Level(int severity, String description) {
            this.severity = severity;
            this.description = description;
        }
    }

    /** Live system signals used to decide level. */
    public static final class Signal {
        public final double errorRate;
        public final double p99LatencyMs;
        public final double cpuUtilization;
        public final double dependencyHealthyRatio;
        public final double emptyRate;
        public final long timestampMs;

        public Signal(
                double errorRate,
                double p99LatencyMs,
                double cpuUtilization,
                double dependencyHealthyRatio,
                double emptyRate) {
            this(errorRate, p99LatencyMs, cpuUtilization, dependencyHealthyRatio, emptyRate,
                    System.currentTimeMillis());
        }

        public Signal(
                double errorRate,
                double p99LatencyMs,
                double cpuUtilization,
                double dependencyHealthyRatio,
                double emptyRate,
                long timestampMs) {
            this.errorRate = errorRate;
            this.p99LatencyMs = p99LatencyMs;
            this.cpuUtilization = cpuUtilization;
            this.dependencyHealthyRatio = dependencyHealthyRatio;
            this.emptyRate = emptyRate;
            this.timestampMs = timestampMs;
        }
    }

    /** Thresholds to enter each level (any match escalates). */
    public static final class Thresholds {
        public final double l1ErrorRate;
        public final double l1P99Ms;
        public final double l1Cpu;
        public final double l2ErrorRate;
        public final double l2P99Ms;
        public final double l2Cpu;
        public final double l2DependencyHealthy;
        public final double l3ErrorRate;
        public final double l3P99Ms;
        public final double l3EmptyRate;
        public final double l4ErrorRate;

        public Thresholds(
                double l1ErrorRate, double l1P99Ms, double l1Cpu,
                double l2ErrorRate, double l2P99Ms, double l2Cpu, double l2DependencyHealthy,
                double l3ErrorRate, double l3P99Ms, double l3EmptyRate,
                double l4ErrorRate) {
            this.l1ErrorRate = l1ErrorRate;
            this.l1P99Ms = l1P99Ms;
            this.l1Cpu = l1Cpu;
            this.l2ErrorRate = l2ErrorRate;
            this.l2P99Ms = l2P99Ms;
            this.l2Cpu = l2Cpu;
            this.l2DependencyHealthy = l2DependencyHealthy;
            this.l3ErrorRate = l3ErrorRate;
            this.l3P99Ms = l3P99Ms;
            this.l3EmptyRate = l3EmptyRate;
            this.l4ErrorRate = l4ErrorRate;
        }

        public static Thresholds defaults() {
            return new Thresholds(
                    0.02, 150.0, 0.75,
                    0.05, 250.0, 0.90, 0.5,
                    0.15, 500.0, 0.05,
                    0.40);
        }
    }

    /** Pipeline knobs derived from level — applied as experiment-like params. */
    public static final class PipelineKnobs {
        public final boolean enableFineRank;
        public final boolean enableRerank;
        public final boolean enableOptionalRecall;
        public final boolean hotListOnly;
        public final boolean failClosed;
        public final int recallQuota;
        public final int fineQuota;

        public PipelineKnobs(
                boolean enableFineRank,
                boolean enableRerank,
                boolean enableOptionalRecall,
                boolean hotListOnly,
                boolean failClosed,
                int recallQuota,
                int fineQuota) {
            this.enableFineRank = enableFineRank;
            this.enableRerank = enableRerank;
            this.enableOptionalRecall = enableOptionalRecall;
            this.hotListOnly = hotListOnly;
            this.failClosed = failClosed;
            this.recallQuota = recallQuota;
            this.fineQuota = fineQuota;
        }

        public Map<String, String> asExperimentParams() {
            Map<String, String> m = new LinkedHashMap<>();
            m.put("degrade.enable_fine", String.valueOf(enableFineRank));
            m.put("degrade.enable_rerank", String.valueOf(enableRerank));
            m.put("degrade.enable_optional_recall", String.valueOf(enableOptionalRecall));
            m.put("degrade.hot_list_only", String.valueOf(hotListOnly));
            m.put("degrade.fail_closed", String.valueOf(failClosed));
            m.put("recall.total_quota", String.valueOf(recallQuota));
            m.put("fine.quota", String.valueOf(fineQuota));
            m.put("rerank.Mmr.enabled", String.valueOf(enableRerank));
            return m;
        }

        public static PipelineKnobs forLevel(Level level) {
            switch (level) {
                case L1_SOFT:
                    return new PipelineKnobs(true, false, false, false, false, 800, 80);
                case L2_HARD:
                    return new PipelineKnobs(false, false, false, false, false, 400, 0);
                case L3_EMERGENCY:
                    return new PipelineKnobs(false, false, false, true, false, 50, 0);
                case L4_CIRCUIT:
                    return new PipelineKnobs(false, false, false, true, true, 0, 0);
                case L0_NORMAL:
                default:
                    return new PipelineKnobs(true, true, true, false, false, 1000, 100);
            }
        }
    }

    private final Thresholds thresholds;
    private final AtomicReference<Level> level = new AtomicReference<>(Level.L0_NORMAL);
    private final long upgradeHoldMs;
    private final long downgradeHoldMs;
    private long lastChangeMs;
    private final List<Consumer<LevelChange>> listeners = new ArrayList<>();

    public DegradationPolicy() {
        this(Thresholds.defaults(), 10_000L, 60_000L);
    }

    public DegradationPolicy(Thresholds thresholds, long upgradeHoldMs, long downgradeHoldMs) {
        this.thresholds = Objects.requireNonNull(thresholds);
        this.upgradeHoldMs = upgradeHoldMs;   // min time before escalating again
        this.downgradeHoldMs = downgradeHoldMs; // min time before recovering (hysteresis)
        this.lastChangeMs = 0L;
    }

    public void addListener(Consumer<LevelChange> listener) {
        listeners.add(Objects.requireNonNull(listener));
    }

    public Level currentLevel() {
        return level.get();
    }

    public PipelineKnobs currentKnobs() {
        return PipelineKnobs.forLevel(level.get());
    }

    /**
     * Evaluate signals and possibly change level.
     * Escalation is faster than recovery (asymmetric hysteresis).
     */
    public synchronized Level evaluate(Signal signal) {
        Objects.requireNonNull(signal);
        Level suggested = suggest(signal);
        Level current = level.get();
        long now = signal.timestampMs;
        if (suggested.severity > current.severity) {
            // escalate
            if (lastChangeMs == 0 || now - lastChangeMs >= upgradeHoldMs) {
                setLevel(suggested, signal, "escalate");
            }
        } else if (suggested.severity < current.severity) {
            // recover only if hold elapsed AND suggested is strictly healthier
            if (now - lastChangeMs >= downgradeHoldMs) {
                // step down one level at a time for safety
                Level next = stepDown(current);
                if (suggested.severity <= next.severity) {
                    setLevel(next, signal, "recover");
                }
            }
        }
        return level.get();
    }

    public synchronized void forceLevel(Level newLevel, String reason) {
        setLevel(newLevel, null, reason == null ? "force" : reason);
    }

    private Level suggest(Signal s) {
        Thresholds t = thresholds;
        if (s.errorRate >= t.l4ErrorRate) return Level.L4_CIRCUIT;
        if (s.errorRate >= t.l3ErrorRate || s.p99LatencyMs >= t.l3P99Ms || s.emptyRate >= t.l3EmptyRate) {
            return Level.L3_EMERGENCY;
        }
        if (s.errorRate >= t.l2ErrorRate || s.p99LatencyMs >= t.l2P99Ms
                || s.cpuUtilization >= t.l2Cpu
                || s.dependencyHealthyRatio < t.l2DependencyHealthy) {
            return Level.L2_HARD;
        }
        if (s.errorRate >= t.l1ErrorRate || s.p99LatencyMs >= t.l1P99Ms || s.cpuUtilization >= t.l1Cpu) {
            return Level.L1_SOFT;
        }
        return Level.L0_NORMAL;
    }

    private static Level stepDown(Level current) {
        switch (current) {
            case L4_CIRCUIT:
                return Level.L3_EMERGENCY;
            case L3_EMERGENCY:
                return Level.L2_HARD;
            case L2_HARD:
                return Level.L1_SOFT;
            case L1_SOFT:
                return Level.L0_NORMAL;
            default:
                return Level.L0_NORMAL;
        }
    }

    private void setLevel(Level newLevel, Signal signal, String reason) {
        Level old = level.getAndSet(newLevel);
        lastChangeMs = System.currentTimeMillis();
        if (old == newLevel) return;
        LevelChange change = new LevelChange(old, newLevel, reason, signal, lastChangeMs);
        for (Consumer<LevelChange> l : listeners) {
            try {
                l.accept(change);
            } catch (RuntimeException ignored) {
            }
        }
    }

    public static final class LevelChange {
        public final Level from;
        public final Level to;
        public final String reason;
        public final Signal signal;
        public final long timestampMs;

        public LevelChange(Level from, Level to, String reason, Signal signal, long timestampMs) {
            this.from = from;
            this.to = to;
            this.reason = reason;
            this.signal = signal;
            this.timestampMs = timestampMs;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "Degrade %s -> %s (%s)", from, to, reason);
        }
    }
}
