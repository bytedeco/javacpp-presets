/*
 * Schema field for FeatureView / FeatureTable columns (Feast Field).
 */
package org.bytedeco.pytorch.feature.core;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Immutable column schema entry. */
public final class Field {

    private final String name;
    private final ValueType valueType;
    private final String description;
    private final int embeddingDim;
    private final Map<String, String> tags;

    private Field(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        if (this.name.isEmpty()) throw new IllegalArgumentException("field name empty");
        this.valueType = b.valueType != null ? b.valueType : ValueType.UNKNOWN;
        this.description = b.description != null ? b.description : "";
        this.embeddingDim = b.embeddingDim;
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        if (this.valueType == ValueType.EMBEDDING && this.embeddingDim <= 0) {
            throw new IllegalArgumentException("EMBEDDING field requires positive embeddingDim: " + name);
        }
    }

    public static Field of(String name, ValueType type) {
        return builder(name).valueType(type).build();
    }

    public static Field embedding(String name, int dim) {
        return builder(name).valueType(ValueType.EMBEDDING).embeddingDim(dim).build();
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

    public String description() {
        return description;
    }

    public int embeddingDim() {
        return embeddingDim;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public String tag(String key) {
        return tags.get(key);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Field)) return false;
        Field field = (Field) o;
        return embeddingDim == field.embeddingDim
                && name.equals(field.name)
                && valueType == field.valueType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, valueType, embeddingDim);
    }

    @Override
    public String toString() {
        return "Field{" + name + ":" + valueType
                + (valueType == ValueType.EMBEDDING ? "[" + embeddingDim + "]" : "")
                + "}";
    }

    public static final class Builder {
        private final String name;
        private ValueType valueType = ValueType.UNKNOWN;
        private String description;
        private int embeddingDim;
        private final Map<String, String> tags = new LinkedHashMap<>();

        private Builder(String name) {
            this.name = name;
        }

        public Builder valueType(ValueType valueType) {
            this.valueType = valueType;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder embeddingDim(int embeddingDim) {
            this.embeddingDim = embeddingDim;
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

        public Field build() {
            return new Field(this);
        }
    }
}
