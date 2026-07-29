/*
 * Image feature specification (URI + optional embedding).
 */
package org.bytedeco.pytorch.utils.feature.multimodal;

import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.ValueType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Image URI column + optional embedding. */
public final class ImageFeatureSpec {

    private final String name;
    private final EmbeddingFeatureSpec embedding;
    private final String description;

    private ImageFeatureSpec(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.embedding = b.embedding;
        this.description = b.description != null ? b.description : "";
    }

    public static ImageFeatureSpec of(String name, int embDim) {
        return builder(name).embedding(EmbeddingFeatureSpec.of(name + "_emb", embDim)).build();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() { return name; }
    public EmbeddingFeatureSpec embedding() { return embedding; }
    public String description() { return description; }

    public List<Field> fields() {
        List<Field> f = new ArrayList<>();
        f.add(Field.builder(name + "_uri").valueType(ValueType.STRING)
                .description(description)
                .tag("modality", Modality.IMAGE.name())
                .build());
        if (embedding != null) f.add(embedding.toField());
        return f;
    }

    public static final class Builder {
        private final String name;
        private EmbeddingFeatureSpec embedding;
        private String description;

        private Builder(String name) { this.name = name; }
        public Builder embedding(EmbeddingFeatureSpec embedding) { this.embedding = embedding; return this; }
        public Builder description(String description) { this.description = description; return this; }
        public ImageFeatureSpec build() { return new ImageFeatureSpec(this); }
    }
}
