/*
 * Aggregated quality report for a feature view (validation + freshness + drift).
 */
package org.bytedeco.pytorch.feature.lifecycle;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Immutable quality snapshot. */
public final class FeatureQualityReport {

    private final String project;
    private final String viewName;
    private final FeatureValidator.Report validation;
    private final FreshnessMonitor.Status freshness;
    private final List<FeatureDriftMonitor.PsiResult> drift;
    private final long generatedAtMs;
    private final Map<String, String> meta;

    private FeatureQualityReport(Builder b) {
        this.project = b.project != null ? b.project : "default";
        this.viewName = Objects.requireNonNull(b.viewName, "viewName");
        this.validation = b.validation;
        this.freshness = b.freshness;
        this.drift = b.drift != null ? List.copyOf(b.drift) : List.of();
        this.generatedAtMs = b.generatedAtMs > 0 ? b.generatedAtMs : System.currentTimeMillis();
        this.meta = Collections.unmodifiableMap(new LinkedHashMap<>(b.meta));
    }

    public static Builder builder(String viewName) {
        return new Builder(viewName);
    }

    public String project() { return project; }
    public String viewName() { return viewName; }
    public FeatureValidator.Report validation() { return validation; }
    public FreshnessMonitor.Status freshness() { return freshness; }
    public List<FeatureDriftMonitor.PsiResult> drift() { return drift; }
    public long generatedAtMs() { return generatedAtMs; }
    public Map<String, String> meta() { return meta; }

    public boolean healthy() {
        boolean valOk = validation == null || validation.ok;
        boolean freshOk = freshness == null || !freshness.alert;
        boolean driftOk = true;
        for (FeatureDriftMonitor.PsiResult p : drift) {
            if (p.alert) {
                driftOk = false;
                break;
            }
        }
        return valOk && freshOk && driftOk;
    }

    @Override
    public String toString() {
        return "FeatureQualityReport{" + project + "/" + viewName
                + ", healthy=" + healthy()
                + ", validation=" + validation
                + ", freshness=" + freshness
                + ", drift=" + drift.size()
                + "}";
    }

    public static final class Builder {
        private String project = "default";
        private final String viewName;
        private FeatureValidator.Report validation;
        private FreshnessMonitor.Status freshness;
        private List<FeatureDriftMonitor.PsiResult> drift;
        private long generatedAtMs;
        private final Map<String, String> meta = new LinkedHashMap<>();

        private Builder(String viewName) {
            this.viewName = viewName;
        }

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder validation(FeatureValidator.Report validation) {
            this.validation = validation;
            return this;
        }

        public Builder freshness(FreshnessMonitor.Status freshness) {
            this.freshness = freshness;
            return this;
        }

        public Builder drift(List<FeatureDriftMonitor.PsiResult> drift) {
            this.drift = drift;
            return this;
        }

        public Builder generatedAtMs(long generatedAtMs) {
            this.generatedAtMs = generatedAtMs;
            return this;
        }

        public Builder meta(String k, String v) {
            if (k != null && v != null) meta.put(k, v);
            return this;
        }

        public FeatureQualityReport build() {
            return new FeatureQualityReport(this);
        }
    }
}
