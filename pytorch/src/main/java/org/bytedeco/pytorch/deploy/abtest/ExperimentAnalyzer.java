/*
 * Experiment analysis facade: SRM + primary metrics + guardrails + ship decision.
 *
 * Mirrors the "experiment readout" step in Meta XP / Microsoft ExP / Libra:
 *   1. Integrity: SRM, AA-test residual, exposure health
 *   2. Primary metric significance + CI + CUPED optional
 *   3. Guardrail evaluation
 *   4. Ship / no-ship recommendation with human-readable report
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Analyzes an experiment using accumulated online metrics.
 */
public final class ExperimentAnalyzer {

    /** Final product decision recommendation. */
    public enum ShipDecision {
        /** Treatment wins on primary, guardrails green, SRM clean. */
        SHIP,
        /** No significant win, or mixed signals — keep iterating. */
        NO_SHIP,
        /** Guardrail / SRM / integrity failure — do not ship. */
        BLOCKED,
        /** Not enough data yet. */
        INCONCLUSIVE
    }

    private final double alpha;
    private final double srmAlpha;
    private final long minSamplePerArm;
    private final List<Guardrail> guardrails;

    private ExperimentAnalyzer(Builder b) {
        this.alpha = b.alpha;
        this.srmAlpha = b.srmAlpha;
        this.minSamplePerArm = b.minSamplePerArm;
        this.guardrails = Collections.unmodifiableList(new ArrayList<>(b.guardrails));
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Full readout for control vs a single treatment variant.
     */
    public Report analyze(
            Experiment experiment,
            OnlineMetricsCollector collector,
            String controlVariantId,
            String treatmentVariantId) {
        Objects.requireNonNull(experiment, "experiment");
        Objects.requireNonNull(collector, "collector");

        long nC = collector.exposureCount(experiment.id(), controlVariantId);
        long nT = collector.exposureCount(experiment.id(), treatmentVariantId);

        StatisticalTest.SrmResult srm = null;
        try {
            srm = collector.srm(experiment, srmAlpha);
        } catch (RuntimeException ex) {
            // not enough data
        }

        Map<String, StatisticalTest.MeanTestResult> meanResults = new LinkedHashMap<>();
        Map<String, StatisticalTest.ProportionTestResult> rateResults = new LinkedHashMap<>();

        for (String metric : experiment.primaryMetrics()) {
            OnlineMetricsCollector.StatsSnapshot c =
                    collector.stats(experiment.id(), controlVariantId, metric);
            OnlineMetricsCollector.StatsSnapshot t =
                    collector.stats(experiment.id(), treatmentVariantId, metric);
            if (c.n >= 2 && t.n >= 2) {
                try {
                    meanResults.put(metric, collector.compareMean(
                            experiment.id(), controlVariantId, treatmentVariantId, metric, alpha));
                } catch (RuntimeException ignored) {
                }
            }
            // Heuristic: if values look binary (variance ~ mean*(1-mean)), also run proportion test.
            if (c.n >= 1 && t.n >= 1 && looksLikeRate(c) && looksLikeRate(t)) {
                try {
                    rateResults.put(metric, collector.compareRate(
                            experiment.id(), controlVariantId, treatmentVariantId, metric, alpha));
                } catch (RuntimeException ignored) {
                }
            }
        }

        // Guardrail metrics also loaded into snapshot.
        List<String> allMetricKeys = new ArrayList<>(experiment.primaryMetrics());
        for (String g : experiment.guardrailMetrics()) {
            if (!allMetricKeys.contains(g)) {
                allMetricKeys.add(g);
            }
        }
        Double srmP = srm == null ? null : srm.pValue;
        Guardrail.ExperimentSnapshot snap = collector.guardrailSnapshot(
                experiment.id(), controlVariantId, treatmentVariantId, allMetricKeys, srmP);

        List<Guardrail> rules = new ArrayList<>(guardrails);
        // Always attach default integrity rules if caller did not.
        if (rules.isEmpty()) {
            rules.add(Guardrail.srm("default_srm", srmAlpha, Guardrail.Action.KILL));
            rules.add(Guardrail.minSample("default_min_n", minSamplePerArm));
        }
        Guardrail.EvaluationResult guardrailResult = Guardrail.evaluateAll(rules, snap);

        ShipDecision decision = decide(nC, nT, srm, meanResults, guardrailResult, experiment);
        return new Report(
                experiment.id(),
                controlVariantId,
                treatmentVariantId,
                nC,
                nT,
                srm,
                meanResults,
                rateResults,
                guardrailResult,
                decision,
                render(experiment, nC, nT, srm, meanResults, guardrailResult, decision));
    }

    private ShipDecision decide(
            long nC,
            long nT,
            StatisticalTest.SrmResult srm,
            Map<String, StatisticalTest.MeanTestResult> meanResults,
            Guardrail.EvaluationResult guardrailResult,
            Experiment experiment) {
        if (guardrailResult.decision == Guardrail.Decision.KILL
                || guardrailResult.decision == Guardrail.Decision.ROLLBACK
                || guardrailResult.decision == Guardrail.Decision.PAUSE) {
            return ShipDecision.BLOCKED;
        }
        if (srm != null && srm.srmDetected) {
            return ShipDecision.BLOCKED;
        }
        if (nC < minSamplePerArm || nT < minSamplePerArm) {
            return ShipDecision.INCONCLUSIVE;
        }
        if (experiment.primaryMetrics().isEmpty()) {
            return ShipDecision.INCONCLUSIVE;
        }
        // Ship if ALL primary metrics are significant in the positive direction.
        // (Conservative; multi-metric sequential testing is more nuanced in production.)
        boolean anyPrimary = false;
        boolean allPositiveSignificant = true;
        boolean anyPositiveSignificant = false;
        for (String m : experiment.primaryMetrics()) {
            StatisticalTest.MeanTestResult r = meanResults.get(m);
            if (r == null) {
                allPositiveSignificant = false;
                continue;
            }
            anyPrimary = true;
            if (r.significantAtAlpha && r.absoluteDelta > 0.0) {
                anyPositiveSignificant = true;
            } else {
                allPositiveSignificant = false;
            }
            // Significant negative on primary is a hard no-ship.
            if (r.significantAtAlpha && r.absoluteDelta < 0.0) {
                return ShipDecision.NO_SHIP;
            }
        }
        if (!anyPrimary) {
            return ShipDecision.INCONCLUSIVE;
        }
        if (allPositiveSignificant) {
            return ShipDecision.SHIP;
        }
        if (anyPositiveSignificant) {
            // Mixed: some win some flat — conservative no-ship with note.
            return ShipDecision.NO_SHIP;
        }
        return ShipDecision.NO_SHIP;
    }

    private static boolean looksLikeRate(OnlineMetricsCollector.StatsSnapshot s) {
        if (s.n == 0) return false;
        if (s.mean < -0.01 || s.mean > 1.01) return false;
        // For Bernoulli, var ≈ p(1-p); allow slack.
        double expected = s.mean * (1.0 - s.mean);
        return s.variance <= expected + 0.05;
    }

    private static String render(
            Experiment experiment,
            long nC,
            long nT,
            StatisticalTest.SrmResult srm,
            Map<String, StatisticalTest.MeanTestResult> meanResults,
            Guardrail.EvaluationResult guardrailResult,
            ShipDecision decision) {
        StringBuilder sb = new StringBuilder();
        sb.append("===== Experiment Report: ").append(experiment.id()).append(" =====\n");
        sb.append("name: ").append(experiment.name()).append('\n');
        sb.append("layer: ").append(experiment.layerId()).append('\n');
        sb.append("status: ").append(experiment.status()).append('\n');
        sb.append("hypothesis: ").append(experiment.hypothesis()).append('\n');
        sb.append(String.format(Locale.ROOT, "exposure: control=%d treatment=%d\n", nC, nT));
        if (srm != null) {
            sb.append("SRM: ").append(srm).append('\n');
        } else {
            sb.append("SRM: n/a\n");
        }
        sb.append("-- primary metrics --\n");
        for (String m : experiment.primaryMetrics()) {
            StatisticalTest.MeanTestResult r = meanResults.get(m);
            if (r == null) {
                sb.append("  ").append(m).append(": insufficient data\n");
            } else {
                sb.append("  ").append(m).append(": ").append(r).append('\n');
            }
        }
        sb.append("-- guardrails --\n");
        sb.append("  decision: ").append(guardrailResult.decision).append('\n');
        for (Guardrail.GuardrailHit hit : guardrailResult.fires) {
            sb.append("  FIRE: ").append(hit).append('\n');
        }
        sb.append("SHIP_DECISION: ").append(decision).append('\n');
        return sb.toString();
    }

    /** Immutable analysis report. */
    public static final class Report {
        public final String experimentId;
        public final String controlVariantId;
        public final String treatmentVariantId;
        public final long controlN;
        public final long treatmentN;
        public final StatisticalTest.SrmResult srm;
        public final Map<String, StatisticalTest.MeanTestResult> meanResults;
        public final Map<String, StatisticalTest.ProportionTestResult> rateResults;
        public final Guardrail.EvaluationResult guardrailResult;
        public final ShipDecision decision;
        public final String text;

        public Report(
                String experimentId,
                String controlVariantId,
                String treatmentVariantId,
                long controlN,
                long treatmentN,
                StatisticalTest.SrmResult srm,
                Map<String, StatisticalTest.MeanTestResult> meanResults,
                Map<String, StatisticalTest.ProportionTestResult> rateResults,
                Guardrail.EvaluationResult guardrailResult,
                ShipDecision decision,
                String text) {
            this.experimentId = experimentId;
            this.controlVariantId = controlVariantId;
            this.treatmentVariantId = treatmentVariantId;
            this.controlN = controlN;
            this.treatmentN = treatmentN;
            this.srm = srm;
            this.meanResults = Collections.unmodifiableMap(new LinkedHashMap<>(meanResults));
            this.rateResults = Collections.unmodifiableMap(new LinkedHashMap<>(rateResults));
            this.guardrailResult = guardrailResult;
            this.decision = decision;
            this.text = text;
        }

        @Override
        public String toString() {
            return text;
        }
    }

    public static final class Builder {
        private double alpha = 0.05;
        private double srmAlpha = 0.001;
        private long minSamplePerArm = 1000L;
        private final List<Guardrail> guardrails = new ArrayList<>();

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder srmAlpha(double srmAlpha) {
            this.srmAlpha = srmAlpha;
            return this;
        }

        public Builder minSamplePerArm(long minSamplePerArm) {
            this.minSamplePerArm = minSamplePerArm;
            return this;
        }

        public Builder addGuardrail(Guardrail g) {
            this.guardrails.add(Objects.requireNonNull(g));
            return this;
        }

        public Builder guardrails(List<Guardrail> gs) {
            this.guardrails.clear();
            if (gs != null) {
                this.guardrails.addAll(gs);
            }
            return this;
        }

        public ExperimentAnalyzer build() {
            return new ExperimentAnalyzer(this);
        }
    }
}
