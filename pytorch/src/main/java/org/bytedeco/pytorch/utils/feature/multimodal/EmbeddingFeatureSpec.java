/*
 * Embedding feature specification (dim, distance, model id).
 */
package org.bytedeco.pytorch.utils.feature.multimodal;

import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.ValueType;

import java.util.Objects;

/** Spec for fixed-dim float embedding features. */
public final class EmbeddingFeatureSpec {

    public enum Distance {
        L2, COSINE, IP
    }

    private final String name;
    private final int dim;
    private final Distance distance;
    private final String modelId;
    private final String description;

    private EmbeddingFeatureSpec(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.dim = b.dim;
        if (dim <= 0) throw new IllegalArgumentException("embedding dim must be > 0");
        this.distance = b.distance != null ? b.distance : Distance.COSINE;
        this.modelId = b.modelId != null ? b.modelId : "";
        this.description = b.description != null ? b.description : "";
    }

    public static EmbeddingFeatureSpec of(String name, int dim) {
        return builder(name).dim(dim).build();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() { return name; }
    public int dim() { return dim; }
    public Distance distance() { return distance; }
    public String modelId() { return modelId; }
    public String description() { return description; }

    public Field toField() {
        return Field.builder(name)
                .valueType(ValueType.EMBEDDING)
                .embeddingDim(dim)
                .description(description)
                .tag("modality", Modality.EMBEDDING.name())
                .tag("distance", distance.name())
                .tag("model_id", modelId)
                .build();
    }

    /** Validate embedding vector length / finiteness. */
    public boolean validate(float[] v) {
        if (v == null || v.length != dim) return false;
        for (float x : v) {
            if (!Float.isFinite(x)) return false;
        }
        return true;
    }

    public double l2Norm(float[] v) {
        if (v == null) return Double.NaN;
        double s = 0;
        for (float x : v) s += (double) x * x;
        return Math.sqrt(s);
    }

    public static final class Builder {
        private final String name;
        private int dim;
        private Distance distance = Distance.COSINE;
        private String modelId;
        private String description;

        private Builder(String name) { this.name = name; }
        public Builder dim(int dim) { this.dim = dim; return this; }
        public Builder distance(Distance distance) { this.distance = distance; return this; }
        public Builder modelId(String modelId) { this.modelId = modelId; return this; }
        public Builder description(String description) { this.description = description; return this; }
        public EmbeddingFeatureSpec build() { return new EmbeddingFeatureSpec(this); }
    }
}
