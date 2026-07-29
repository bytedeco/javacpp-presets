/*
 * Project namespace for multi-team feature isolation (Feast project / Databricks catalog schema).
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Logical project / workspace. */
public final class Project {

    public static final String DEFAULT = "default";

    private final String name;
    private final String description;
    private final String owner;
    private final Map<String, String> tags;
    private final long createdAtMs;

    private Project(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        if (this.name.isEmpty()) throw new IllegalArgumentException("project name empty");
        this.description = b.description != null ? b.description : "";
        this.owner = b.owner != null ? b.owner : "";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        this.createdAtMs = b.createdAtMs > 0 ? b.createdAtMs : System.currentTimeMillis();
    }

    public static Project of(String name) {
        return builder(name).build();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() {
        return name;
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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Project)) return false;
        return name.equals(((Project) o).name);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public String toString() {
        return "Project{" + name + "}";
    }

    public static final class Builder {
        private final String name;
        private String description;
        private String owner;
        private final Map<String, String> tags = new LinkedHashMap<>();
        private long createdAtMs;

        private Builder(String name) {
            this.name = name;
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

        public Project build() {
            return new Project(this);
        }
    }
}
