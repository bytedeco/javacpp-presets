/*
 * Embedding store SPI — multimodal / tower / item embeddings for ANN retrieval
 * and online feature serving of EMBEDDING fields.
 *
 * Backends: MEMORY, SQLITE (SQLiteEmbeddingStore), MILVUS, REDIS_VECTOR, LANCE.
 */
package org.bytedeco.pytorch.utils.feature.store;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Namespace-scoped embedding KV + optional top-K search. */
public interface EmbeddingStore extends AutoCloseable {

    /** Backend label for diagnostics. */
    String backend();

    /** Declared default dim (0 if mixed). */
    int dim();

    void put(String namespace, String id, float[] vector);

    default void put(String namespace, String id, float[] vector, Map<String, Object> meta) {
        put(namespace, id, vector);
    }

    default void putBatch(String namespace, Map<String, float[]> vectors) {
        if (vectors == null) return;
        for (Map.Entry<String, float[]> e : vectors.entrySet()) {
            put(namespace, e.getKey(), e.getValue());
        }
    }

    Optional<float[]> get(String namespace, String id);

    default Map<String, float[]> getBatch(String namespace, Collection<String> ids) {
        Map<String, float[]> out = new LinkedHashMap<>();
        if (ids == null) return out;
        for (String id : ids) {
            get(namespace, id).ifPresent(v -> out.put(id, v));
        }
        return out;
    }

    void delete(String namespace, String id);

    long count(String namespace);

    /**
     * Optional ANN / brute-force top-K. Empty list if backend has no search
     * (pure KV still OK for feature serving).
     */
    default List<EmbeddingHit> search(String namespace, float[] query, int topK) {
        return List.of();
    }

    @Override
    default void close() {}

    /** Search hit. */
    final class EmbeddingHit {
        public final String id;
        public final double score;
        public final float[] vector;

        public EmbeddingHit(String id, double score, float[] vector) {
            this.id = id;
            this.score = score;
            this.vector = vector;
        }

        @Override
        public String toString() {
            return "EmbeddingHit{id=" + id + ", score=" + score + "}";
        }
    }
}
