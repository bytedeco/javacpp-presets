package org.bytedeco.pytorch.data.dataframe.vectorstore;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * One vector point with optional string id, numeric id, payload metadata, and dense embedding.
 *
 * <p>Either {@link #id()} (string) or {@link #numericId()} may be set; adapters pick whichever
 * the backend requires (Qdrant accepts both, Redis uses string keys, pgvector uses bigint, …).
 */
public final class VectorRecord {
    private final String id;
    private final long numericId;
    private final boolean hasNumericId;
    private final float[] vector;
    private final Map<String, Object> payload;

    private VectorRecord(String id, long numericId, boolean hasNumericId,
                         float[] vector, Map<String, Object> payload) {
        this.id = id;
        this.numericId = numericId;
        this.hasNumericId = hasNumericId;
        this.vector = Objects.requireNonNull(vector, "vector");
        this.payload = payload == null || payload.isEmpty()
            ? Map.of()
            : Collections.unmodifiableMap(new LinkedHashMap<>(payload));
    }

    public static VectorRecord of(String id, float[] vector) {
        return new VectorRecord(id, -1L, false, vector, null);
    }

    public static VectorRecord of(String id, float[] vector, Map<String, Object> payload) {
        return new VectorRecord(id, -1L, false, vector, payload);
    }

    public static VectorRecord of(long id, float[] vector) {
        return new VectorRecord(Long.toString(id), id, true, vector, null);
    }

    public static VectorRecord of(long id, float[] vector, Map<String, Object> payload) {
        return new VectorRecord(Long.toString(id), id, true, vector, payload);
    }

    public static Builder builder() { return new Builder(); }

    public String id() { return id; }
    public long numericId() { return numericId; }
    public boolean hasNumericId() { return hasNumericId; }
    public float[] vector() { return vector; }
    public int dim() { return vector.length; }
    public Map<String, Object> payload() { return payload; }

    /** Resolved id preferred by most HTTP backends (string form of numeric if only that is set). */
    public String resolvedId() {
        if (id != null && !id.isEmpty()) return id;
        if (hasNumericId) return Long.toString(numericId);
        throw new IllegalStateException("VectorRecord has no id");
    }

    public VectorRecord withPayload(Map<String, Object> extra) {
        if (extra == null || extra.isEmpty()) return this;
        Map<String, Object> m = new LinkedHashMap<>(payload);
        m.putAll(extra);
        return new VectorRecord(id, numericId, hasNumericId, vector, m);
    }

    public static final class Builder {
        private String id;
        private long numericId = -1L;
        private boolean hasNumericId;
        private float[] vector;
        private final Map<String, Object> payload = new LinkedHashMap<>();

        public Builder id(String id) { this.id = id; return this; }
        public Builder id(long id) {
            this.numericId = id;
            this.hasNumericId = true;
            if (this.id == null) this.id = Long.toString(id);
            return this;
        }
        public Builder vector(float[] v) { this.vector = v; return this; }
        public Builder put(String key, Object value) {
            if (key != null) payload.put(key, value);
            return this;
        }
        public Builder payload(Map<String, ?> m) {
            if (m != null) {
                for (Map.Entry<String, ?> e : m.entrySet()) payload.put(e.getKey(), e.getValue());
            }
            return this;
        }
        public VectorRecord build() {
            if (vector == null) throw new IllegalStateException("vector required");
            return new VectorRecord(id, numericId, hasNumericId, vector, payload);
        }
    }

    @Override
    public String toString() {
        return "VectorRecord{id=" + resolvedIdSafe() + ", dim=" + vector.length
            + ", payloadKeys=" + payload.keySet() + "}";
    }

    private String resolvedIdSafe() {
        try { return resolvedId(); } catch (RuntimeException e) { return "?"; }
    }
}
