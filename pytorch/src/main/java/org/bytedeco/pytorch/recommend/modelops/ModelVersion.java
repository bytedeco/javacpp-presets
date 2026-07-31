/*
 * Model operations lifecycle for recommendation models.
 *
 * Covers the MLOps path used by Meta, Google, ByteDance, Alibaba, Netflix:
 *   - Model registry (version, stage, artifacts, metrics)
 *   - Stage transitions: TRAINED -> OFFLINE_PASS -> SHADOW -> CANARY -> PROD -> ARCHIVED
 *   - Shadow serving (dark launch / mirror traffic, compare scores)
 *   - Feature / prediction drift detection
 *   - Instant model rollback
 *   - Online learning hooks (nearline updates)
 */
package org.bytedeco.pytorch.recommend.modelops;

import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Immutable model version metadata.
 */
public final class ModelVersion {

    private final String modelName;
    private final String versionId;
    private final String artifactUri;
    private final String framework; // pytorch / onnx / tensorrt / ...
    private final ModelStage stage;
    private final Map<String, Double> offlineMetrics;
    private final Map<String, String> tags;
    private final String parentVersionId;
    private final String trainingJobId;
    private final Instant createdAt;
    private final Instant updatedAt;
    private final String description;

    private ModelVersion(Builder b) {
        if (b.modelName == null || b.modelName.isEmpty()) {
            throw new IllegalArgumentException("modelName required");
        }
        if (b.versionId == null || b.versionId.isEmpty()) {
            throw new IllegalArgumentException("versionId required");
        }
        this.modelName = b.modelName;
        this.versionId = b.versionId;
        this.artifactUri = b.artifactUri != null ? b.artifactUri : "";
        this.framework = b.framework != null ? b.framework : "pytorch";
        this.stage = b.stage != null ? b.stage : ModelStage.TRAINED;
        this.offlineMetrics = Collections.unmodifiableMap(new LinkedHashMap<>(b.offlineMetrics));
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        this.parentVersionId = b.parentVersionId;
        this.trainingJobId = b.trainingJobId;
        this.createdAt = b.createdAt != null ? b.createdAt : Instant.now();
        this.updatedAt = b.updatedAt != null ? b.updatedAt : this.createdAt;
        this.description = b.description != null ? b.description : "";
    }

    public static Builder builder(String modelName, String versionId) {
        return new Builder(modelName, versionId);
    }

    public String modelName() {
        return modelName;
    }

    public String versionId() {
        return versionId;
    }

    public String artifactUri() {
        return artifactUri;
    }

    public String framework() {
        return framework;
    }

    public ModelStage stage() {
        return stage;
    }

    public Map<String, Double> offlineMetrics() {
        return offlineMetrics;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public String parentVersionId() {
        return parentVersionId;
    }

    public String trainingJobId() {
        return trainingJobId;
    }

    public Instant createdAt() {
        return createdAt;
    }

    public Instant updatedAt() {
        return updatedAt;
    }

    public String description() {
        return description;
    }

    public String fullyQualifiedId() {
        return modelName + ":" + versionId;
    }

    public ModelVersion withStage(ModelStage newStage) {
        return builder(modelName, versionId)
                .artifactUri(artifactUri)
                .framework(framework)
                .stage(newStage)
                .offlineMetrics(offlineMetrics)
                .tags(tags)
                .parentVersionId(parentVersionId)
                .trainingJobId(trainingJobId)
                .createdAt(createdAt)
                .updatedAt(Instant.now())
                .description(description)
                .build();
    }

    public ModelVersion withOfflineMetrics(Map<String, Double> metrics) {
        return builder(modelName, versionId)
                .artifactUri(artifactUri)
                .framework(framework)
                .stage(stage)
                .offlineMetrics(metrics)
                .tags(tags)
                .parentVersionId(parentVersionId)
                .trainingJobId(trainingJobId)
                .createdAt(createdAt)
                .updatedAt(Instant.now())
                .description(description)
                .build();
    }

    @Override
    public String toString() {
        return "ModelVersion{" + fullyQualifiedId() + ", stage=" + stage
                + ", artifact=" + artifactUri + "}";
    }

    public static final class Builder {
        private final String modelName;
        private final String versionId;
        private String artifactUri;
        private String framework;
        private ModelStage stage = ModelStage.TRAINED;
        private final Map<String, Double> offlineMetrics = new LinkedHashMap<>();
        private final Map<String, String> tags = new LinkedHashMap<>();
        private String parentVersionId;
        private String trainingJobId;
        private Instant createdAt;
        private Instant updatedAt;
        private String description;

        private Builder(String modelName, String versionId) {
            this.modelName = modelName;
            this.versionId = versionId;
        }

        public Builder artifactUri(String artifactUri) {
            this.artifactUri = artifactUri;
            return this;
        }

        public Builder framework(String framework) {
            this.framework = framework;
            return this;
        }

        public Builder stage(ModelStage stage) {
            this.stage = stage;
            return this;
        }

        public Builder offlineMetric(String key, double value) {
            this.offlineMetrics.put(key, value);
            return this;
        }

        public Builder offlineMetrics(Map<String, Double> metrics) {
            this.offlineMetrics.clear();
            if (metrics != null) this.offlineMetrics.putAll(metrics);
            return this;
        }

        public Builder tag(String key, String value) {
            this.tags.put(key, value);
            return this;
        }

        public Builder tags(Map<String, String> tags) {
            this.tags.clear();
            if (tags != null) this.tags.putAll(tags);
            return this;
        }

        public Builder parentVersionId(String parentVersionId) {
            this.parentVersionId = parentVersionId;
            return this;
        }

        public Builder trainingJobId(String trainingJobId) {
            this.trainingJobId = trainingJobId;
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder updatedAt(Instant updatedAt) {
            this.updatedAt = updatedAt;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public ModelVersion build() {
            return new ModelVersion(this);
        }
    }
}
