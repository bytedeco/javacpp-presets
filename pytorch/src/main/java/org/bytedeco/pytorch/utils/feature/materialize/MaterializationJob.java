/*
 * Materialization job descriptor (views + time window + incremental flag).
 */
package org.bytedeco.pytorch.utils.feature.materialize;

import org.bytedeco.pytorch.utils.feature.core.FeatureView;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

/** Immutable materialization request. */
public final class MaterializationJob {

    private final String jobId;
    private final String project;
    private final List<FeatureView> views;
    private final Instant start;
    private final Instant end;
    private final boolean incremental;

    private MaterializationJob(Builder b) {
        this.jobId = b.jobId != null && !b.jobId.isEmpty() ? b.jobId : "mat-" + UUID.randomUUID().toString().substring(0, 8);
        this.project = b.project != null ? b.project : "default";
        this.views = Collections.unmodifiableList(new ArrayList<>(b.views));
        this.start = b.start;
        this.end = b.end != null ? b.end : Instant.now();
        this.incremental = b.incremental;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String jobId() { return jobId; }
    public String project() { return project; }
    public List<FeatureView> views() { return views; }
    public Instant start() { return start; }
    public Instant end() { return end; }
    public boolean incremental() { return incremental; }

    @Override
    public String toString() {
        return "MaterializationJob{" + jobId + ", views=" + views.size()
                + ", incremental=" + incremental + "}";
    }

    public static final class Builder {
        private String jobId;
        private String project = "default";
        private final List<FeatureView> views = new ArrayList<>();
        private Instant start;
        private Instant end;
        private boolean incremental;

        public Builder jobId(String jobId) { this.jobId = jobId; return this; }
        public Builder project(String project) { this.project = project; return this; }
        public Builder views(List<FeatureView> views) {
            this.views.clear();
            if (views != null) this.views.addAll(views);
            return this;
        }
        public Builder view(FeatureView view) {
            if (view != null) views.add(view);
            return this;
        }
        public Builder start(Instant start) { this.start = start; return this; }
        public Builder end(Instant end) { this.end = end; return this; }
        public Builder incremental(boolean incremental) { this.incremental = incremental; return this; }

        public MaterializationJob build() {
            if (views.isEmpty()) throw new IllegalStateException("materialization job requires views");
            return new MaterializationJob(this);
        }
    }
}
