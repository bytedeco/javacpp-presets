/*
 * Stream FeatureView — Feast StreamFeatureView / Feathub stream feature descriptor.
 *
 * Binds a Kafka (or similar) source + optional window aggregation specs.
 * Materialization may be simulated offline in-process for demos/benchmarks;
 * production would push aggregations into a stream engine then online write.
 */
package org.bytedeco.pytorch.feature.core;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/** Streaming feature view with source + schema + optional aggregations. */
public final class StreamFeatureView {

    private final String name;
    private final String project;
    private final List<Entity> entities;
    private final List<Field> schema;
    private final FeatureTable source;
    private final Duration ttl;
    private final boolean online;
    private final List<String> aggregationSpecs;
    private final String description;
    private final String owner;
    private final Map<String, String> tags;
    private final long createdAtMs;

    private StreamFeatureView(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        if (this.name.isEmpty()) throw new IllegalArgumentException("stream feature view name empty");
        this.project = b.project != null && !b.project.isEmpty() ? b.project : Project.DEFAULT;
        this.entities = Collections.unmodifiableList(new ArrayList<>(b.entities));
        this.schema = Collections.unmodifiableList(new ArrayList<>(b.schema));
        this.source = b.source != null ? b.source : FeatureTable.kafka(this.name, this.name);
        this.ttl = b.ttl != null ? b.ttl : Duration.ofHours(1);
        this.online = b.online;
        this.aggregationSpecs = Collections.unmodifiableList(new ArrayList<>(b.aggregationSpecs));
        this.description = b.description != null ? b.description : "";
        this.owner = b.owner != null ? b.owner : "";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        this.createdAtMs = b.createdAtMs > 0 ? b.createdAtMs : System.currentTimeMillis();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() {
        return name;
    }

    public String project() {
        return project;
    }

    public List<Entity> entities() {
        return entities;
    }

    public List<String> entityNames() {
        return entities.stream().map(Entity::name).collect(Collectors.toList());
    }

    public List<String> joinKeys() {
        return entities.stream().map(Entity::joinKey).collect(Collectors.toList());
    }

    public List<Field> schema() {
        return schema;
    }

    public List<String> featureNames() {
        return schema.stream().map(Field::name).collect(Collectors.toList());
    }

    public FeatureTable source() {
        return source;
    }

    public Duration ttl() {
        return ttl;
    }

    public long ttlMillis() {
        return ttl.isZero() || ttl.isNegative() ? 0L : ttl.toMillis();
    }

    public boolean online() {
        return online;
    }

    /** Opaque aggregation descriptors (e.g. "COUNT(click) OVER TUMBLE 5m"). */
    public List<String> aggregationSpecs() {
        return aggregationSpecs;
    }

    public String description() {
        return description;
    }

    public String owner() {
        return owner;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public long createdAtMs() {
        return createdAtMs;
    }

    public String qualifiedName() {
        return project + "/" + name;
    }

    /** Project as a batch FeatureView for materialize/PIT paths that share schema. */
    public FeatureView asBatchView() {
        return FeatureView.builder(name)
                .project(project)
                .entities(entities)
                .schema(schema)
                .source(source)
                .ttl(ttl)
                .online(online)
                .description(description)
                .owner(owner)
                .tags(tags)
                .createdAtMs(createdAtMs)
                .build();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof StreamFeatureView)) return false;
        StreamFeatureView that = (StreamFeatureView) o;
        return name.equals(that.name) && project.equals(that.project);
    }

    @Override
    public int hashCode() {
        return Objects.hash(project, name);
    }

    @Override
    public String toString() {
        return "StreamFeatureView{" + qualifiedName()
                + ", source=" + source.sourceType()
                + ", aggs=" + aggregationSpecs.size()
                + "}";
    }

    public static final class Builder {
        private final String name;
        private String project = Project.DEFAULT;
        private final List<Entity> entities = new ArrayList<>();
        private final List<Field> schema = new ArrayList<>();
        private FeatureTable source;
        private Duration ttl = Duration.ofHours(1);
        private boolean online = true;
        private final List<String> aggregationSpecs = new ArrayList<>();
        private String description;
        private String owner;
        private final Map<String, String> tags = new LinkedHashMap<>();
        private long createdAtMs;

        private Builder(String name) {
            this.name = name;
        }

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder entities(Entity... entities) {
            if (entities != null) this.entities.addAll(Arrays.asList(entities));
            return this;
        }

        public Builder entities(List<Entity> entities) {
            if (entities != null) this.entities.addAll(entities);
            return this;
        }

        public Builder schema(Field... fields) {
            if (fields != null) this.schema.addAll(Arrays.asList(fields));
            return this;
        }

        public Builder schema(List<Field> fields) {
            if (fields != null) this.schema.addAll(fields);
            return this;
        }

        public Builder source(FeatureTable source) {
            this.source = source;
            return this;
        }

        public Builder ttl(Duration ttl) {
            this.ttl = ttl;
            return this;
        }

        public Builder online(boolean online) {
            this.online = online;
            return this;
        }

        public Builder aggregation(String spec) {
            if (spec != null && !spec.isEmpty()) aggregationSpecs.add(spec);
            return this;
        }

        public Builder aggregations(List<String> specs) {
            if (specs != null) aggregationSpecs.addAll(specs);
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder owner(String owner) {
            this.owner = owner;
            return this;
        }

        public Builder tag(String k, String v) {
            if (k != null && v != null) tags.put(k, v);
            return this;
        }

        public Builder tags(Map<String, String> more) {
            if (more != null) tags.putAll(more);
            return this;
        }

        public Builder createdAtMs(long createdAtMs) {
            this.createdAtMs = createdAtMs;
            return this;
        }

        public StreamFeatureView build() {
            return new StreamFeatureView(this);
        }
    }
}
