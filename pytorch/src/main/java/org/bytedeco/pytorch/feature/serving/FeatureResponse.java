/*
 * Feature retrieval response — one or many FeatureVectors plus diagnostics.
 */
package org.bytedeco.pytorch.feature.serving;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Immutable serving response. */
public final class FeatureResponse {

    private final String featureService;
    private final String project;
    private final List<FeatureVector> vectors;
    private final long elapsedNanos;
    private final int viewsHit;
    private final int viewsMiss;
    private final int onDemandComputed;
    private final Map<String, String> meta;
    private final boolean success;
    private final String error;

    private FeatureResponse(Builder b) {
        this.featureService = b.featureService != null ? b.featureService : "";
        this.project = b.project != null ? b.project : "default";
        this.vectors = Collections.unmodifiableList(new ArrayList<>(b.vectors));
        this.elapsedNanos = b.elapsedNanos;
        this.viewsHit = b.viewsHit;
        this.viewsMiss = b.viewsMiss;
        this.onDemandComputed = b.onDemandComputed;
        this.meta = Collections.unmodifiableMap(new LinkedHashMap<>(b.meta));
        this.success = b.success;
        this.error = b.error != null ? b.error : "";
    }

    public static Builder builder() {
        return new Builder();
    }

    public String featureService() {
        return featureService;
    }

    public String project() {
        return project;
    }

    public List<FeatureVector> vectors() {
        return vectors;
    }

    /** First vector (single-entity online path). */
    public FeatureVector vector() {
        return vectors.isEmpty() ? FeatureVector.builder().build() : vectors.get(0);
    }

    public int size() {
        return vectors.size();
    }

    public long elapsedNanos() {
        return elapsedNanos;
    }

    public double elapsedMs() {
        return elapsedNanos / 1_000_000.0;
    }

    public int viewsHit() {
        return viewsHit;
    }

    public int viewsMiss() {
        return viewsMiss;
    }

    public int onDemandComputed() {
        return onDemandComputed;
    }

    public Map<String, String> meta() {
        return meta;
    }

    public boolean success() {
        return success;
    }

    public String error() {
        return error;
    }

    @Override
    public String toString() {
        return "FeatureResponse{svc=" + featureService
                + ", n=" + vectors.size()
                + ", hit=" + viewsHit
                + ", miss=" + viewsMiss
                + ", ms=" + elapsedMs()
                + ", ok=" + success
                + "}";
    }

    public static final class Builder {
        private String featureService;
        private String project = "default";
        private final List<FeatureVector> vectors = new ArrayList<>();
        private long elapsedNanos;
        private int viewsHit;
        private int viewsMiss;
        private int onDemandComputed;
        private final Map<String, String> meta = new LinkedHashMap<>();
        private boolean success = true;
        private String error;

        public Builder featureService(String featureService) {
            this.featureService = featureService;
            return this;
        }

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder vector(FeatureVector v) {
            if (v != null) vectors.add(v);
            return this;
        }

        public Builder vectors(List<FeatureVector> vs) {
            if (vs != null) vectors.addAll(vs);
            return this;
        }

        public Builder elapsedNanos(long elapsedNanos) {
            this.elapsedNanos = elapsedNanos;
            return this;
        }

        public Builder viewsHit(int viewsHit) {
            this.viewsHit = viewsHit;
            return this;
        }

        public Builder viewsMiss(int viewsMiss) {
            this.viewsMiss = viewsMiss;
            return this;
        }

        public Builder onDemandComputed(int onDemandComputed) {
            this.onDemandComputed = onDemandComputed;
            return this;
        }

        public Builder meta(String k, String v) {
            if (k != null && v != null) meta.put(k, v);
            return this;
        }

        public Builder success(boolean success) {
            this.success = success;
            return this;
        }

        public Builder error(String error) {
            this.error = error;
            this.success = false;
            return this;
        }

        public FeatureResponse build() {
            return new FeatureResponse(this);
        }
    }
}
