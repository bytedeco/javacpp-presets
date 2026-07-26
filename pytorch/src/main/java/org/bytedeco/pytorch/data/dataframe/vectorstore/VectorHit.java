package org.bytedeco.pytorch.data.dataframe.vectorstore;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One neighbor returned by {@link VectorStore#search(VectorQuery)}.
 *
 * <p>{@link #score()} is backend-native (similarity or distance). Use
 * {@link #distance()} when the adapter was able to normalize to "lower is better".
 */
public final class VectorHit {
    private final String id;
    private final long numericId;
    private final boolean hasNumericId;
    private final float score;
    private final Float distance;
    private final float[] vector;
    private final Map<String, Object> payload;

    public VectorHit(String id, float score, float[] vector, Map<String, Object> payload) {
        this(id, -1L, false, score, null, vector, payload);
    }

    public VectorHit(String id, long numericId, boolean hasNumericId,
                     float score, Float distance, float[] vector, Map<String, Object> payload) {
        this.id = id;
        this.numericId = numericId;
        this.hasNumericId = hasNumericId;
        this.score = score;
        this.distance = distance;
        this.vector = vector;
        this.payload = payload == null || payload.isEmpty()
            ? Map.of()
            : Collections.unmodifiableMap(new LinkedHashMap<>(payload));
    }

    public static VectorHit of(String id, float score) {
        return new VectorHit(id, score, null, null);
    }

    public static VectorHit of(long id, float score) {
        return new VectorHit(Long.toString(id), id, true, score, null, null, null);
    }

    public String id() { return id; }
    public long numericId() { return numericId; }
    public boolean hasNumericId() { return hasNumericId; }
    public float score() { return score; }
    /** Lower-is-better distance when known; may be {@code null}. */
    public Float distance() { return distance; }
    public float[] vector() { return vector; }
    public Map<String, Object> payload() { return payload; }

    public VectorHit withDistance(float d) {
        return new VectorHit(id, numericId, hasNumericId, score, d, vector, payload);
    }

    public VectorHit withVector(float[] v) {
        return new VectorHit(id, numericId, hasNumericId, score, distance, v, payload);
    }

    public VectorHit withPayload(Map<String, Object> p) {
        return new VectorHit(id, numericId, hasNumericId, score, distance, vector, p);
    }

    @Override
    public String toString() {
        return "VectorHit{id=" + id + ", score=" + score
            + (distance != null ? ", distance=" + distance : "")
            + ", payloadKeys=" + payload.keySet() + "}";
    }
}
