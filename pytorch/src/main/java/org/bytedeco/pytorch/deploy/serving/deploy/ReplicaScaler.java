/*
 * Replica autoscaling for recommendation serving.
 *
 * Industry:
 *   - Kubernetes HPA (CPU / custom metrics) + KEDA
 *   - Google / Meta predictive scaling for diurnal recsys traffic
 *   - Alibaba / ByteDance: QPS-per-pod + p99 latency dual objectives
 *   - Surge for flash sales / evening peaks
 *
 * Controller logic (reference implementation):
 *   desired = ceil(currentQps / targetQpsPerPod)
 *   clamp to [minReplicas, maxReplicas]
 *   optional: scale up immediately, scale down with cooldown
 */
package org.bytedeco.pytorch.deploy.serving.deploy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/** Horizontal replica scaler with cooldown and dual signals. */
public final class ReplicaScaler {

    public static final class Config {
        public final int minReplicas;
        public final int maxReplicas;
        public final double targetQpsPerPod;
        public final double targetP99LatencyMs;
        public final long scaleUpCooldownMs;
        public final long scaleDownCooldownMs;
        /** Max fraction of current replicas to change in one step (e.g. 0.5). */
        public final double maxStepFraction;

        public Config(
                int minReplicas,
                int maxReplicas,
                double targetQpsPerPod,
                double targetP99LatencyMs,
                long scaleUpCooldownMs,
                long scaleDownCooldownMs,
                double maxStepFraction) {
            if (minReplicas < 1 || maxReplicas < minReplicas) {
                throw new IllegalArgumentException("invalid replica bounds");
            }
            if (targetQpsPerPod <= 0.0) {
                throw new IllegalArgumentException("targetQpsPerPod must be > 0");
            }
            this.minReplicas = minReplicas;
            this.maxReplicas = maxReplicas;
            this.targetQpsPerPod = targetQpsPerPod;
            this.targetP99LatencyMs = targetP99LatencyMs;
            this.scaleUpCooldownMs = scaleUpCooldownMs;
            this.scaleDownCooldownMs = scaleDownCooldownMs;
            this.maxStepFraction = maxStepFraction <= 0 ? 1.0 : maxStepFraction;
        }

        public static Config defaults() {
            // Typical recsys ranker pod targets — tune per service.
            return new Config(2, 200, 500.0, 50.0, 30_000L, 300_000L, 0.5);
        }
    }

    public static final class Signal {
        public final double qps;
        public final double p99LatencyMs;
        public final double cpuUtilization;
        public final int currentReplicas;
        public final long timestampMs;

        public Signal(double qps, double p99LatencyMs, double cpuUtilization, int currentReplicas) {
            this(qps, p99LatencyMs, cpuUtilization, currentReplicas, System.currentTimeMillis());
        }

        public Signal(
                double qps,
                double p99LatencyMs,
                double cpuUtilization,
                int currentReplicas,
                long timestampMs) {
            this.qps = qps;
            this.p99LatencyMs = p99LatencyMs;
            this.cpuUtilization = cpuUtilization;
            this.currentReplicas = currentReplicas;
            this.timestampMs = timestampMs;
        }
    }

    public static final class Decision {
        public final int currentReplicas;
        public final int desiredReplicas;
        public final String reason;
        public final boolean scaled;
        public final long timestampMs;

        public Decision(int currentReplicas, int desiredReplicas, String reason, boolean scaled) {
            this.currentReplicas = currentReplicas;
            this.desiredReplicas = desiredReplicas;
            this.reason = reason;
            this.scaled = scaled;
            this.timestampMs = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "ScaleDecision{%d -> %d scaled=%s reason=%s}",
                    currentReplicas, desiredReplicas, scaled, reason);
        }
    }

    private final Config config;
    private long lastScaleUpMs;
    private long lastScaleDownMs;
    private final List<Decision> history = new ArrayList<>();

    public ReplicaScaler(Config config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    public Config config() {
        return config;
    }

    public List<Decision> history() {
        return Collections.unmodifiableList(new ArrayList<>(history));
    }

    /**
     * Compute next desired replica count from live signals.
     */
    public synchronized Decision evaluate(Signal signal) {
        Objects.requireNonNull(signal, "signal");
        int current = Math.max(1, signal.currentReplicas);

        // QPS is the primary capacity signal (can scale up or down).
        int byQps = (int) Math.ceil(signal.qps / config.targetQpsPerPod);
        // Latency / CPU only push UP when unhealthy — they must not floor desired
        // at current, otherwise healthy low-QPS clusters can never scale down.
        int byLatency = 0;
        if (signal.p99LatencyMs > config.targetP99LatencyMs && config.targetP99LatencyMs > 0) {
            double ratio = signal.p99LatencyMs / config.targetP99LatencyMs;
            byLatency = (int) Math.ceil(current * ratio);
        }
        int byCpu = 0;
        if (signal.cpuUtilization > 0.7) {
            byCpu = (int) Math.ceil(current * (signal.cpuUtilization / 0.7));
        }

        int rawDesired = Math.max(byQps, Math.max(byLatency, byCpu));
        // If all pressure signals are quiet and QPS is tiny, still honor min via clamp.
        rawDesired = clamp(rawDesired, config.minReplicas, config.maxReplicas);

        // Step limit
        int maxStep = Math.max(1, (int) Math.ceil(current * config.maxStepFraction));
        int desired = rawDesired;
        if (desired > current + maxStep) {
            desired = current + maxStep;
        } else if (desired < current - maxStep) {
            desired = current - maxStep;
        }

        long now = signal.timestampMs;
        String reason;
        boolean scaled;
        if (desired > current) {
            if (now - lastScaleUpMs < config.scaleUpCooldownMs && lastScaleUpMs > 0) {
                desired = current;
                reason = "scale_up_cooldown";
                scaled = false;
            } else {
                reason = String.format(Locale.ROOT,
                        "scale_up qps=%.1f p99=%.1f cpu=%.2f raw=%d",
                        signal.qps, signal.p99LatencyMs, signal.cpuUtilization, rawDesired);
                lastScaleUpMs = now;
                scaled = true;
            }
        } else if (desired < current) {
            if (now - lastScaleDownMs < config.scaleDownCooldownMs && lastScaleDownMs > 0) {
                desired = current;
                reason = "scale_down_cooldown";
                scaled = false;
            } else {
                reason = String.format(Locale.ROOT,
                        "scale_down qps=%.1f p99=%.1f raw=%d",
                        signal.qps, signal.p99LatencyMs, rawDesired);
                lastScaleDownMs = now;
                scaled = true;
            }
        } else {
            reason = "hold";
            scaled = false;
        }

        Decision d = new Decision(current, desired, reason, scaled);
        history.add(d);
        if (history.size() > 1000) {
            history.remove(0);
        }
        return d;
    }

    /**
     * Predictive hint from diurnal forecast (e.g. next 15 min expected QPS).
     * Takes max(reactive, predictive) so we pre-warm before peaks.
     */
    public synchronized Decision evaluateWithForecast(Signal signal, double forecastQps) {
        Decision reactive = evaluate(signal);
        int predicted = (int) Math.ceil(forecastQps / config.targetQpsPerPod);
        predicted = clamp(predicted, config.minReplicas, config.maxReplicas);
        if (predicted > reactive.desiredReplicas) {
            Decision d = new Decision(signal.currentReplicas, predicted,
                    reactive.reason + "+forecast", predicted != signal.currentReplicas);
            history.add(d);
            return d;
        }
        return reactive;
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
