/*
 * Entity — primary join key in Feast / Featureform / Databricks Feature Store.
 * Examples: user_id, item_id, sku_id, account_id, compound_id.
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Join-key entity definition. */
public final class Entity {

    private final String name;
    private final ValueType valueType;
    private final String joinKey;
    private final String description;
    private final String project;
    private final Map<String, String> tags;

    private Entity(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        if (this.name.isEmpty()) throw new IllegalArgumentException("entity name empty");
        this.valueType = b.valueType != null ? b.valueType : ValueType.INT64;
        this.joinKey = b.joinKey != null && !b.joinKey.isEmpty() ? b.joinKey : this.name;
        this.description = b.description != null ? b.description : "";
        this.project = b.project != null ? b.project : "default";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
    }

    public static Entity of(String name, ValueType type) {
        return builder(name).valueType(type).build();
    }

    public static Entity of(String name) {
        return of(name, ValueType.INT64);
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() {
        return name;
    }

    public ValueType valueType() {
        return valueType;
    }

    /** Physical column name used in offline/online tables (defaults to name). */
    public String joinKey() {
        return joinKey;
    }

    public String description() {
        return description;
    }

    public String project() {
        return project;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public Entity withProject(String project) {
        return builder(name)
                .valueType(valueType)
                .joinKey(joinKey)
                .description(description)
                .project(project)
                .tags(tags)
                .build();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Entity)) return false;
        Entity entity = (Entity) o;
        return name.equals(entity.name) && project.equals(entity.project);
    }

    @Override
    public int hashCode() {
        return Objects.hash(project, name);
    }

    @Override
    public String toString() {
        return "Entity{" + project + "/" + name + ":" + valueType + ", joinKey=" + joinKey + "}";
    }

    public static final class Builder {
        private final String name;
        private ValueType valueType = ValueType.INT64;
        private String joinKey;
        private String description;
        private String project = "default";
        private final Map<String, String> tags = new LinkedHashMap<>();

        private Builder(String name) {
            this.name = name;
        }

        public Builder valueType(ValueType valueType) {
            this.valueType = valueType;
            return this;
        }

        public Builder joinKey(String joinKey) {
            this.joinKey = joinKey;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder project(String project) {
            this.project = project;
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

        public Entity build() {
            return new Entity(this);
        }
    }
}
