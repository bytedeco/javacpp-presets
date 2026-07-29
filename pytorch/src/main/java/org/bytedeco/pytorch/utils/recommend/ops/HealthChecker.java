/*
 * Health checks and periodic inspection (巡检) for recommendation services.
 *
 * Levels:
 *   - Liveness: process up
 *   - Readiness: can accept traffic (model loaded, deps ok)
 *   - Dependency: feature store, ANN, rank model, AB config
 *   - Deep inspection: sample request replay, metric anomaly scan
 */
package org.bytedeco.pytorch.utils.recommend.ops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Supplier;

/** Health checker + inspector. */
public final class HealthChecker {

    public enum Status {
        UP,
        DEGRADED,
        DOWN
    }

    public static final class CheckResult {
        public final String name;
        public final Status status;
        public final String message;
        public final long latencyMs;
        public final long timestampMs;

        public CheckResult(String name, Status status, String message, long latencyMs) {
            this.name = name;
            this.status = status;
            this.message = message == null ? "" : message;
            this.latencyMs = latencyMs;
            this.timestampMs = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "Check[%s]=%s (%s) %dms",
                    name, status, message, latencyMs);
        }
    }

    public static final class Report {
        public final Status overall;
        public final List<CheckResult> checks;
        public final long timestampMs;

        public Report(Status overall, List<CheckResult> checks) {
            this.overall = overall;
            this.checks = Collections.unmodifiableList(new ArrayList<>(checks));
            this.timestampMs = System.currentTimeMillis();
        }

        public boolean ready() {
            return overall == Status.UP || overall == Status.DEGRADED;
        }

        public boolean live() {
            return overall != Status.DOWN;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Health overall=").append(overall).append('\n');
            for (CheckResult c : checks) {
                sb.append("  ").append(c).append('\n');
            }
            return sb.toString();
        }
    }

    /** A named health probe. */
    public interface Probe {
        String name();

        CheckResult check();
    }

    private final CopyOnWriteArrayList<Probe> probes = new CopyOnWriteArrayList<>();

    public HealthChecker addProbe(Probe probe) {
        probes.add(Objects.requireNonNull(probe));
        return this;
    }

    public HealthChecker addProbe(String name, Supplier<Status> supplier) {
        return addProbe(new Probe() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public CheckResult check() {
                long t0 = System.currentTimeMillis();
                try {
                    Status s = supplier.get();
                    return new CheckResult(name, s, s.name(), System.currentTimeMillis() - t0);
                } catch (RuntimeException ex) {
                    return new CheckResult(name, Status.DOWN, ex.getMessage(),
                            System.currentTimeMillis() - t0);
                }
            }
        });
    }

    public Report checkAll() {
        List<CheckResult> results = new ArrayList<>();
        Status overall = Status.UP;
        for (Probe p : probes) {
            CheckResult r;
            try {
                r = p.check();
            } catch (RuntimeException ex) {
                r = new CheckResult(p.name(), Status.DOWN, ex.getMessage(), 0L);
            }
            results.add(r);
            overall = worse(overall, r.status);
        }
        if (probes.isEmpty()) {
            overall = Status.UP;
            results.add(new CheckResult("default", Status.UP, "no probes", 0L));
        }
        return new Report(overall, results);
    }

    private static Status worse(Status a, Status b) {
        if (a == Status.DOWN || b == Status.DOWN) return Status.DOWN;
        if (a == Status.DEGRADED || b == Status.DEGRADED) return Status.DEGRADED;
        return Status.UP;
    }

    /**
     * Periodic inspection runner — evaluates probes + custom anomaly rules.
     */
    public static final class Inspector {
        public interface AnomalyRule {
            String name();

            /** @return anomaly message or null if healthy */
            String evaluate();
        }

        private final HealthChecker healthChecker;
        private final CopyOnWriteArrayList<AnomalyRule> rules = new CopyOnWriteArrayList<>();
        private final CopyOnWriteArrayList<InspectionReport> history = new CopyOnWriteArrayList<>();

        public Inspector(HealthChecker healthChecker) {
            this.healthChecker = Objects.requireNonNull(healthChecker);
        }

        public Inspector addRule(AnomalyRule rule) {
            rules.add(Objects.requireNonNull(rule));
            return this;
        }

        public Inspector addRule(String name, Supplier<String> eval) {
            return addRule(new AnomalyRule() {
                @Override
                public String name() {
                    return name;
                }

                @Override
                public String evaluate() {
                    return eval.get();
                }
            });
        }

        public InspectionReport run() {
            Report health = healthChecker.checkAll();
            Map<String, String> anomalies = new LinkedHashMap<>();
            for (AnomalyRule rule : rules) {
                try {
                    String msg = rule.evaluate();
                    if (msg != null && !msg.isEmpty()) {
                        anomalies.put(rule.name(), msg);
                    }
                } catch (RuntimeException ex) {
                    anomalies.put(rule.name(), "rule_error: " + ex.getMessage());
                }
            }
            InspectionReport report = new InspectionReport(health, anomalies);
            history.add(report);
            if (history.size() > 100) {
                history.remove(0);
            }
            return report;
        }

        public List<InspectionReport> history() {
            return Collections.unmodifiableList(new ArrayList<>(history));
        }
    }

    public static final class InspectionReport {
        public final Report health;
        public final Map<String, String> anomalies;
        public final long timestampMs;

        public InspectionReport(Report health, Map<String, String> anomalies) {
            this.health = health;
            this.anomalies = Collections.unmodifiableMap(new LinkedHashMap<>(anomalies));
            this.timestampMs = System.currentTimeMillis();
        }

        public boolean healthy() {
            return health.overall == Status.UP && anomalies.isEmpty();
        }

        @Override
        public String toString() {
            return "Inspection{health=" + health.overall
                    + ", anomalies=" + anomalies.size()
                    + ", detail=" + anomalies + "}";
        }
    }
}
