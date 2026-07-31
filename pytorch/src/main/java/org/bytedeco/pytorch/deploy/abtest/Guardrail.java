/*
 * Experiment guardrails — automatic kill / pause rules used by production
 * experiment platforms (Meta XP, Microsoft ExP, ByteDance Libra, Alibaba).
 *
 * Typical guardrails in recommendation:
 *   - SRM p-value below threshold
 *   - Primary / secondary metric relative drop beyond tolerance
 *   - Latency p99 / error-rate SLO breach on treatment
 *   - Business hard constraints (GMV, refund rate, complaint rate)
 *
 * Design:
 *   Guardrail is a pure predicate over ExperimentSnapshot.
 *   GuardrailEvaluator runs all rules and produces a decision:
 *     CONTINUE | PAUSE | KILL | ROLLBACK
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * Single guardrail rule + composite evaluator.
 */
public final class Guardrail {

    /** Action recommended when a guardrail fires. */
    public enum Action {
        /** Informational only; do not change experiment status. */
        WARN,
        /** Pause new assignments; keep sticky users. */
        PAUSE,
        /** Kill experiment; force all traffic to control. */
        KILL,
        /** Full rollback of associated model / config version. */
        ROLLBACK
    }

    /** Severity for dashboards / paging. */
    public enum Severity {
        INFO,
        WARNING,
        CRITICAL
    }

    /** Decision after evaluating all rules. */
    public enum Decision {
        CONTINUE,
        PAUSE,
        KILL,
        ROLLBACK
    }

    private final String id;
    private final String name;
    private final String metricKey;
    private final Action action;
    private final Severity severity;
    private final Function<ExperimentSnapshot, GuardrailHit> predicate;
    private final String description;

    private Guardrail(Builder b) {
        this.id = Objects.requireNonNull(b.id, "id");
        this.name = b.name != null ? b.name : b.id;
        this.metricKey = b.metricKey;
        this.action = b.action != null ? b.action : Action.KILL;
        this.severity = b.severity != null ? b.severity : Severity.CRITICAL;
        this.predicate = Objects.requireNonNull(b.predicate, "predicate");
        this.description = b.description != null ? b.description : "";
    }

    public static Builder builder(String id) {
        return new Builder(id);
    }

    public String id() {
        return id;
    }

    public String name() {
        return name;
    }

    public String metricKey() {
        return metricKey;
    }

    public Action action() {
        return action;
    }

    public Severity severity() {
        return severity;
    }

    public String description() {
        return description;
    }

    public GuardrailHit evaluate(ExperimentSnapshot snapshot) {
        GuardrailHit hit = predicate.apply(snapshot);
        if (hit == null) {
            return GuardrailHit.pass(id);
        }
        return hit;
    }

    // ---- factory helpers matching common recsys rules -----------------------

    /**
     * SRM guardrail: fire when chi-square p-value &lt; alpha (default 0.001).
     */
    public static Guardrail srm(String id, double alpha, Action action) {
        return builder(id)
                .name("SRM")
                .metricKey("srm_pvalue")
                .action(action)
                .severity(Severity.CRITICAL)
                .description("Sample Ratio Mismatch p < " + alpha)
                .predicate(snap -> {
                    if (snap.srmPValue == null) {
                        return GuardrailHit.pass(id);
                    }
                    if (snap.srmPValue < alpha) {
                        return GuardrailHit.fire(id, "srm_pvalue", snap.srmPValue,
                                "SRM detected p=" + snap.srmPValue + " < " + alpha, action);
                    }
                    return GuardrailHit.pass(id);
                })
                .build();
    }

    /**
     * Relative drop guardrail on a metric mean:
     * fire if (treat - control) / |control| &lt; -maxRelativeDrop.
     *
     * <p>Example: maxRelativeDrop=0.02 means kill if treatment is more than 2% worse.
     */
    public static Guardrail relativeDrop(
            String id, String metricKey, double maxRelativeDrop, Action action) {
        if (maxRelativeDrop < 0.0) {
            throw new IllegalArgumentException("maxRelativeDrop must be >= 0");
        }
        return builder(id)
                .name("relative_drop:" + metricKey)
                .metricKey(metricKey)
                .action(action)
                .severity(Severity.CRITICAL)
                .description(String.format(Locale.ROOT,
                        "relative drop of %s beyond %.2f%%", metricKey, maxRelativeDrop * 100.0))
                .predicate(snap -> {
                    MetricArmPair pair = snap.metrics.get(metricKey);
                    if (pair == null || pair.controlMean == 0.0) {
                        return GuardrailHit.pass(id);
                    }
                    double rel = (pair.treatmentMean - pair.controlMean) / Math.abs(pair.controlMean);
                    if (rel < -maxRelativeDrop) {
                        return GuardrailHit.fire(id, metricKey, rel,
                                String.format(Locale.ROOT,
                                        "%s relative delta=%.4f%% below -%.2f%%",
                                        metricKey, rel * 100.0, maxRelativeDrop * 100.0),
                                action);
                    }
                    return GuardrailHit.pass(id);
                })
                .build();
    }

    /**
     * Absolute threshold on treatment arm (e.g. error rate, latency).
     * Fires when treatment value &gt; maxValue (higher-is-worse metrics).
     */
    public static Guardrail treatmentAbove(
            String id, String metricKey, double maxValue, Action action) {
        return builder(id)
                .name("treatment_above:" + metricKey)
                .metricKey(metricKey)
                .action(action)
                .severity(Severity.CRITICAL)
                .description(metricKey + " treatment > " + maxValue)
                .predicate(snap -> {
                    MetricArmPair pair = snap.metrics.get(metricKey);
                    if (pair == null) {
                        return GuardrailHit.pass(id);
                    }
                    if (pair.treatmentMean > maxValue) {
                        return GuardrailHit.fire(id, metricKey, pair.treatmentMean,
                                metricKey + " treatment=" + pair.treatmentMean + " > " + maxValue,
                                action);
                    }
                    return GuardrailHit.pass(id);
                })
                .build();
    }

    /**
     * Absolute threshold on treatment arm for higher-is-better metrics
     * (fires when treatment &lt; minValue).
     */
    public static Guardrail treatmentBelow(
            String id, String metricKey, double minValue, Action action) {
        return builder(id)
                .name("treatment_below:" + metricKey)
                .metricKey(metricKey)
                .action(action)
                .severity(Severity.CRITICAL)
                .description(metricKey + " treatment < " + minValue)
                .predicate(snap -> {
                    MetricArmPair pair = snap.metrics.get(metricKey);
                    if (pair == null) {
                        return GuardrailHit.pass(id);
                    }
                    if (pair.treatmentMean < minValue) {
                        return GuardrailHit.fire(id, metricKey, pair.treatmentMean,
                                metricKey + " treatment=" + pair.treatmentMean + " < " + minValue,
                                action);
                    }
                    return GuardrailHit.pass(id);
                })
                .build();
    }

    /**
     * Minimum sample size before statistical decisions are trusted.
     * Does not kill — only WARNs until n is reached.
     */
    public static Guardrail minSample(String id, long minPerArm) {
        return builder(id)
                .name("min_sample")
                .metricKey("n")
                .action(Action.WARN)
                .severity(Severity.INFO)
                .description("require n>=" + minPerArm + " per arm")
                .predicate(snap -> {
                    if (snap.controlN < minPerArm || snap.treatmentN < minPerArm) {
                        return GuardrailHit.fire(id, "n",
                                Math.min(snap.controlN, snap.treatmentN),
                                "insufficient sample controlN=" + snap.controlN
                                        + " treatmentN=" + snap.treatmentN
                                        + " need>=" + minPerArm,
                                Action.WARN);
                    }
                    return GuardrailHit.pass(id);
                })
                .build();
    }

    // ---- evaluator ----------------------------------------------------------

    /**
     * Evaluate a list of guardrails; decision is the strongest action among fires.
     * Priority: ROLLBACK &gt; KILL &gt; PAUSE &gt; WARN(=CONTINUE with warnings).
     */
    public static EvaluationResult evaluateAll(List<Guardrail> rules, ExperimentSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot");
        List<GuardrailHit> hits = new ArrayList<>();
        List<GuardrailHit> fires = new ArrayList<>();
        Decision decision = Decision.CONTINUE;
        for (Guardrail g : rules) {
            GuardrailHit hit = g.evaluate(snapshot);
            hits.add(hit);
            if (!hit.fired) {
                continue;
            }
            fires.add(hit);
            Decision d = toDecision(hit.action);
            if (rank(d) > rank(decision)) {
                decision = d;
            }
        }
        return new EvaluationResult(decision, hits, fires, snapshot.experimentId);
    }

    private static Decision toDecision(Action action) {
        switch (action) {
            case ROLLBACK:
                return Decision.ROLLBACK;
            case KILL:
                return Decision.KILL;
            case PAUSE:
                return Decision.PAUSE;
            case WARN:
            default:
                return Decision.CONTINUE;
        }
    }

    private static int rank(Decision d) {
        switch (d) {
            case ROLLBACK:
                return 4;
            case KILL:
                return 3;
            case PAUSE:
                return 2;
            case CONTINUE:
            default:
                return 1;
        }
    }

    // ---- data types ---------------------------------------------------------

    /** Point-in-time view of experiment health for guardrail evaluation. */
    public static final class ExperimentSnapshot {
        public final String experimentId;
        public final long controlN;
        public final long treatmentN;
        public final Double srmPValue;
        public final Map<String, MetricArmPair> metrics;
        public final long timestampMs;

        public ExperimentSnapshot(
                String experimentId,
                long controlN,
                long treatmentN,
                Double srmPValue,
                Map<String, MetricArmPair> metrics) {
            this.experimentId = experimentId;
            this.controlN = controlN;
            this.treatmentN = treatmentN;
            this.srmPValue = srmPValue;
            this.metrics = metrics == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(metrics));
            this.timestampMs = System.currentTimeMillis();
        }
    }

    /** Control vs treatment mean for one metric. */
    public static final class MetricArmPair {
        public final double controlMean;
        public final double treatmentMean;
        public final Double pValue;

        public MetricArmPair(double controlMean, double treatmentMean) {
            this(controlMean, treatmentMean, null);
        }

        public MetricArmPair(double controlMean, double treatmentMean, Double pValue) {
            this.controlMean = controlMean;
            this.treatmentMean = treatmentMean;
            this.pValue = pValue;
        }
    }

    /** Result of one guardrail evaluation. */
    public static final class GuardrailHit {
        public final String guardrailId;
        public final boolean fired;
        public final String metricKey;
        public final double observedValue;
        public final String message;
        public final Action action;

        private GuardrailHit(
                String guardrailId,
                boolean fired,
                String metricKey,
                double observedValue,
                String message,
                Action action) {
            this.guardrailId = guardrailId;
            this.fired = fired;
            this.metricKey = metricKey;
            this.observedValue = observedValue;
            this.message = message;
            this.action = action;
        }

        public static GuardrailHit pass(String id) {
            return new GuardrailHit(id, false, null, Double.NaN, "pass", Action.WARN);
        }

        public static GuardrailHit fire(
                String id, String metricKey, double value, String message, Action action) {
            return new GuardrailHit(id, true, metricKey, value, message, action);
        }

        @Override
        public String toString() {
            return fired
                    ? "FIRE[" + guardrailId + "] " + message + " action=" + action
                    : "PASS[" + guardrailId + "]";
        }
    }

    /** Aggregate evaluation result. */
    public static final class EvaluationResult {
        public final Decision decision;
        public final List<GuardrailHit> allHits;
        public final List<GuardrailHit> fires;
        public final String experimentId;

        public EvaluationResult(
                Decision decision,
                List<GuardrailHit> allHits,
                List<GuardrailHit> fires,
                String experimentId) {
            this.decision = decision;
            this.allHits = Collections.unmodifiableList(new ArrayList<>(allHits));
            this.fires = Collections.unmodifiableList(new ArrayList<>(fires));
            this.experimentId = experimentId;
        }

        public boolean shouldAct() {
            return decision != Decision.CONTINUE;
        }

        @Override
        public String toString() {
            return "GuardrailEval{exp=" + experimentId + ", decision=" + decision
                    + ", fires=" + fires.size() + "}";
        }
    }

    public static final class Builder {
        private final String id;
        private String name;
        private String metricKey;
        private Action action = Action.KILL;
        private Severity severity = Severity.CRITICAL;
        private Function<ExperimentSnapshot, GuardrailHit> predicate;
        private String description;

        private Builder(String id) {
            this.id = id;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder metricKey(String metricKey) {
            this.metricKey = metricKey;
            return this;
        }

        public Builder action(Action action) {
            this.action = action;
            return this;
        }

        public Builder severity(Severity severity) {
            this.severity = severity;
            return this;
        }

        public Builder predicate(Function<ExperimentSnapshot, GuardrailHit> predicate) {
            this.predicate = predicate;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Guardrail build() {
            return new Guardrail(this);
        }
    }
}
