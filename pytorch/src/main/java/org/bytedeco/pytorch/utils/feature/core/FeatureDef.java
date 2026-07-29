/*
 * FeatureDef — registry-level feature metadata (name, type, owner, tags).
 * Not the same as recommend.basic.features.Feature (model embedding input).
 *
 * Feast: Feature; Databricks: feature column in Feature Table; Featureform: feature.
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Immutable feature metadata bound to a view. */
public final class FeatureDef {

    private final String name;
    private final ValueType valueType;
    private final String viewName;
    private final String description;
    private final int embeddingDim;
    private final long vocabSize;
    private final int embedDim;
    private final String owner;
    private final Map<String, String> tags;

    private FeatureDef(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.valueType = b.valueType != null ? b.valueType : ValueType.UNKNOWN;
        this.viewName = b.viewName != null ? b.viewName : "";
        this.description = b.description != null ? b.description : "";
        this.embeddingDim = b.embeddingDim;
        this.vocabSize = b.vocabSize;
        this.embedDim = b.embedDim;
        this.owner = b.owner != null ? b.owner : "";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
    }

    public static FeatureDef of(String name, ValueType type) {
        return builder(name).valueType(type).build();
    }

    public static FeatureDef fromField(Field field) {
        Objects.requireNonNull(field, "field");
        return builder(field.name())
                .valueType(field.valueType())
                .description(field.description())
                .embeddingDim(field.embeddingDim())
                .tags(field.tags())
                .build();
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

    public String viewName() {
        return viewName;
    }

    public String description() {
        return description;
    }

    public int embeddingDim() {
        return embeddingDim;
    }

    /** Optional vocab for sparse id features (bridge to SparseFeature). */
    public long vocabSize() {
        return vocabSize;
    }

    /** Optional model embedding dim (bridge to recommend Feature.embedDim). */
    public int embedDim() {
        return embedDim;
    }

    public String owner() {
        return owner;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public FeatureDef withViewName(String viewName) {
        return builder(name)
                .valueType(valueType)
                .viewName(viewName)
                .description(description)
                .embeddingDim(embeddingDim)
                .vocabSize(vocabSize)
                .embedDim(embedDim)
                .owner(owner)
                .tags(tags)
                .build();
    }

    public Field toField() {
        Field.Builder fb = Field.builder(name).valueType(valueType).description(description).tags(tags);
        if (valueType == ValueType.EMBEDDING) {
            fb.embeddingDim(Math.max(1, embeddingDim));
        }
        return fb.build();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FeatureDef)) return false;
        FeatureDef that = (FeatureDef) o;
        return name.equals(that.name) && Objects.equals(viewName, that.viewName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(viewName, name);
    }

    @Override
    public String toString() {
        return "FeatureDef{" + (viewName.isEmpty() ? "" : viewName + ".") + name + ":" + valueType + "}";
    }

    public static final class Builder {
        private final String name;
        private ValueType valueType = ValueType.UNKNOWN;
        private String viewName;
        private String description;
        private int embeddingDim;
        private long vocabSize;
        private int embedDim;
        private String owner;
        private final Map<String, String> tags = new LinkedHashMap<>();

        private Builder(String name) {
            this.name = name;
        }

        public Builder valueType(ValueType valueType) {
            this.valueType = valueType;
            return this;
        }

        public Builder viewName(String viewName) {
            this.viewName = viewName;
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

        public Builder vocabSize(long vocabSize) {
            this.vocabSize = vocabSize;
            return this;
        }

        public Builder embedDim(int embedDim) {
            this.embedDim = embedDim;
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

        public FeatureDef build() {
            return new FeatureDef(this);
        }
    }
}
