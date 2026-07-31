/*
 * Text feature specification (raw text + optional embedding).
 */
package org.bytedeco.pytorch.feature.multimodal;

import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.ValueType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Text column + optional embedding sibling. */
public final class TextFeatureSpec {

    private final String name;
    private final int maxLen;
    private final EmbeddingFeatureSpec embedding;
    private final String description;

    private TextFeatureSpec(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.maxLen = b.maxLen > 0 ? b.maxLen : 512;
        this.embedding = b.embedding;
        this.description = b.description != null ? b.description : "";
    }

    public static TextFeatureSpec of(String name, int embDim) {
        return builder(name).embedding(EmbeddingFeatureSpec.of(name + "_emb", embDim)).build();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() { return name; }
    public int maxLen() { return maxLen; }
    public EmbeddingFeatureSpec embedding() { return embedding; }
    public String description() { return description; }

    public List<Field> fields() {
        List<Field> f = new ArrayList<>();
        f.add(Field.builder(name).valueType(ValueType.STRING)
                .description(description)
                .tag("modality", Modality.TEXT.name())
                .tag("max_len", String.valueOf(maxLen))
                .build());
        if (embedding != null) f.add(embedding.toField());
        return f;
    }

    public static final class Builder {
        private final String name;
        private int maxLen = 512;
        private EmbeddingFeatureSpec embedding;
        private String description;

        private Builder(String name) { this.name = name; }
        public Builder maxLen(int maxLen) { this.maxLen = maxLen; return this; }
        public Builder embedding(EmbeddingFeatureSpec embedding) { this.embedding = embedding; return this; }
        public Builder description(String description) { this.description = description; return this; }
        public TextFeatureSpec build() { return new TextFeatureSpec(this); }
    }
}
