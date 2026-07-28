package org.bytedeco.pytorch.dataframe.ai;

import java.util.List;

import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;

/**
 * Pluggable embedding model (Daft {@code functions.ai.embed_*} backends).
 *
 * <p>Implementations may be pure-Java feature hashers (always available),
 * TorchScript / safetensors-backed encoders, or remote API clients.
 *
 * <pre>
 *   EmbeddingModel m = EmbeddingRegistry.get("clip-vit-base-patch32");
 *   float[][] vectors = m.embedBatch(List.of(img1, img2), Modality.IMAGE);
 * </pre>
 */
public interface EmbeddingModel extends AutoCloseable {

    /** Model identity / hyperparameters. */
    ModelSpec spec();

    /** Output embedding dimension. */
    default int dimension() { return spec().defaultDim(); }

    /** Human-readable backend name (hash, torch, onnx, http, …). */
    String backend();

    /** Whether this model can embed the given modality. */
    default boolean supports(Modality modality) {
        return spec().supports(modality);
    }

    /**
     * Embed a single input cell value.
     * Accepted types depend on modality: {@code String}, {@code ImageData},
     * {@code AudioData}, {@code VideoData}, {@code float[]}, {@code EmbeddingData}.
     */
    float[] embed(Object input, Modality modality);

    /**
     * Batch embed — default loops {@link #embed}; backends should override for throughput.
     */
    default float[][] embedBatch(List<?> inputs, Modality modality) {
        if (inputs == null || inputs.isEmpty()) return new float[0][];
        float[][] out = new float[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) {
            Object v = inputs.get(i);
            out[i] = v == null ? null : embed(v, modality);
        }
        return out;
    }

    /** Wrap a vector as {@link EmbeddingData} tagged with this model id. */
    default EmbeddingData toEmbeddingData(float[] vector) {
        if (vector == null) return null;
        float[] v = vector;
        if (spec().l2Normalize()) v = EmbeddingMath.l2Normalize(v);
        return new EmbeddingData(v, spec().id());
    }

    /** Optional warm-up (load weights). Default no-op. */
    default void warmup() {}

    /** True if underlying weights/resources are loaded. */
    default boolean isReady() { return true; }

    @Override
    default void close() {}
}
