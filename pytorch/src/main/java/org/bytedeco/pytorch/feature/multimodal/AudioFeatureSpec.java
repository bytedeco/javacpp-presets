/*
 * Audio feature specification (URI + sample rate + optional embedding).
 */
package org.bytedeco.pytorch.feature.multimodal;

import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.ValueType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Audio URI column + sample rate + optional embedding. */
public final class AudioFeatureSpec {

    private final String name;
    private final int sampleRate;
    private final EmbeddingFeatureSpec embedding;
    private final String description;

    private AudioFeatureSpec(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.sampleRate = b.sampleRate > 0 ? b.sampleRate : 16000;
        this.embedding = b.embedding;
        this.description = b.description != null ? b.description : "";
    }

    public static AudioFeatureSpec of(String name, int embDim) {
        return builder(name).embedding(EmbeddingFeatureSpec.of(name + "_emb", embDim)).build();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() { return name; }
    public int sampleRate() { return sampleRate; }
    public EmbeddingFeatureSpec embedding() { return embedding; }
    public String description() { return description; }

    public List<Field> fields() {
        List<Field> f = new ArrayList<>();
        f.add(Field.builder(name + "_uri").valueType(ValueType.STRING)
                .description(description)
                .tag("modality", Modality.AUDIO.name())
                .build());
        f.add(Field.builder(name + "_sr").valueType(ValueType.INT32)
                .tag("modality", Modality.AUDIO.name())
                .tag("sample_rate", String.valueOf(sampleRate))
                .build());
        if (embedding != null) f.add(embedding.toField());
        return f;
    }

    public static final class Builder {
        private final String name;
        private int sampleRate = 16000;
        private EmbeddingFeatureSpec embedding;
        private String description;

        private Builder(String name) { this.name = name; }
        public Builder sampleRate(int sampleRate) { this.sampleRate = sampleRate; return this; }
        public Builder embedding(EmbeddingFeatureSpec embedding) { this.embedding = embedding; return this; }
        public Builder description(String description) { this.description = description; return this; }
        public AudioFeatureSpec build() { return new AudioFeatureSpec(this); }
    }
}
