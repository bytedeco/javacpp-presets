/*
 * Versioned registration envelope for any feature resource
 * (view / service / on-demand / stream).
 */
package org.bytedeco.pytorch.feature.registry;

import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/** Immutable version record stored in the Feature Registry. */
public final class FeatureVersion {

    public enum ResourceType {
        ENTITY,
        FEATURE_VIEW,
        ON_DEMAND_FEATURE_VIEW,
        STREAM_FEATURE_VIEW,
        FEATURE_SERVICE,
        PROJECT
    }

    private final String versionId;
    private final String resourceName;
    private final String project;
    private final ResourceType resourceType;
    private final LifecycleStage stage;
    private final String schemaHash;
    private final String description;
    private final Instant createdAt;
    private final String createdBy;
    private final Map<String, String> meta;
    private final Object payload;

    private FeatureVersion(Builder b) {
        this.versionId = b.versionId != null && !b.versionId.isEmpty()
                ? b.versionId
                : "v-" + UUID.randomUUID().toString().substring(0, 8);
        this.resourceName = Objects.requireNonNull(b.resourceName, "resourceName");
        this.project = b.project != null && !b.project.isEmpty() ? b.project : "default";
        this.resourceType = Objects.requireNonNull(b.resourceType, "resourceType");
        this.stage = b.stage != null ? b.stage : LifecycleStage.DRAFT;
        this.schemaHash = b.schemaHash != null ? b.schemaHash : "";
        this.description = b.description != null ? b.description : "";
        this.createdAt = b.createdAt != null ? b.createdAt : Instant.now();
        this.createdBy = b.createdBy != null ? b.createdBy : "";
        this.meta = Collections.unmodifiableMap(new LinkedHashMap<>(b.meta));
        this.payload = b.payload;
    }

    public static Builder builder(String resourceName, ResourceType type) {
        return new Builder(resourceName, type);
    }

    public String versionId() {
        return versionId;
    }

    public String resourceName() {
        return resourceName;
    }

    public String project() {
        return project;
    }

    public ResourceType resourceType() {
        return resourceType;
    }

    public LifecycleStage stage() {
        return stage;
    }

    public String schemaHash() {
        return schemaHash;
    }

    public String description() {
        return description;
    }

    public Instant createdAt() {
        return createdAt;
    }

    public String createdBy() {
        return createdBy;
    }

    public Map<String, String> meta() {
        return meta;
    }

    @SuppressWarnings("unchecked")
    public <T> T payloadAs(Class<T> type) {
        if (payload == null) return null;
        if (type.isInstance(payload)) return (T) payload;
        throw new ClassCastException("payload is " + payload.getClass().getName()
                + ", not " + type.getName());
    }

    public Object payload() {
        return payload;
    }

    public String fullyQualifiedId() {
        return project + "/" + resourceType.name().toLowerCase() + "/" + resourceName + ":" + versionId;
    }

    public FeatureVersion withStage(LifecycleStage stage) {
        return builder(resourceName, resourceType)
                .versionId(versionId)
                .project(project)
                .stage(stage)
                .schemaHash(schemaHash)
                .description(description)
                .createdAt(createdAt)
                .createdBy(createdBy)
                .meta(meta)
                .payload(payload)
                .build();
    }

    @Override
    public String toString() {
        return "FeatureVersion{" + fullyQualifiedId() + ", stage=" + stage + "}";
    }

    public static final class Builder {
        private String versionId;
        private final String resourceName;
        private String project = "default";
        private final ResourceType resourceType;
        private LifecycleStage stage = LifecycleStage.DRAFT;
        private String schemaHash;
        private String description;
        private Instant createdAt;
        private String createdBy;
        private final Map<String, String> meta = new LinkedHashMap<>();
        private Object payload;

        private Builder(String resourceName, ResourceType resourceType) {
            this.resourceName = resourceName;
            this.resourceType = resourceType;
        }

        public Builder versionId(String versionId) {
            this.versionId = versionId;
            return this;
        }

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder stage(LifecycleStage stage) {
            this.stage = stage;
            return this;
        }

        public Builder schemaHash(String schemaHash) {
            this.schemaHash = schemaHash;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder createdBy(String createdBy) {
            this.createdBy = createdBy;
            return this;
        }

        public Builder meta(String k, String v) {
            if (k != null && v != null) meta.put(k, v);
            return this;
        }

        public Builder meta(Map<String, String> more) {
            if (more != null) meta.putAll(more);
            return this;
        }

        public Builder payload(Object payload) {
            this.payload = payload;
            return this;
        }

        public FeatureVersion build() {
            return new FeatureVersion(this);
        }
    }
}
