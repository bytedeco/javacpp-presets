package org.bytedeco.pytorch.data.dataframe.vectorstore;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * k-NN search request.
 *
 * <p>{@link #filter()} is a free-form backend filter expression / document:
 * <ul>
 *   <li>Qdrant — map matching Filter JSON ({@code must}/{@code should}/…)</li>
 *   <li>Milvus — boolean expr string ({@code "year > 2020"})</li>
 *   <li>OpenSearch — query DSL map fragment</li>
 *   <li>Redis — extra RediSearch filter tokens</li>
 *   <li>pgvector — SQL WHERE fragment (bound separately via params if needed)</li>
 *   <li>Mongo — MQL filter document</li>
 * </ul>
 */
public final class VectorQuery {
    private final float[] vector;
    private final int topK;
    private final Object filter;
    private final boolean includeVector;
    private final boolean includePayload;
    private final Map<String, Object> options;
    private final String vectorName;

    private VectorQuery(float[] vector, int topK, Object filter,
                        boolean includeVector, boolean includePayload,
                        Map<String, Object> options, String vectorName) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("query vector required");
        }
        if (topK <= 0) throw new IllegalArgumentException("topK must be > 0");
        this.vector = vector;
        this.topK = topK;
        this.filter = filter;
        this.includeVector = includeVector;
        this.includePayload = includePayload;
        this.options = options == null || options.isEmpty()
            ? Map.of()
            : Collections.unmodifiableMap(new LinkedHashMap<>(options));
        this.vectorName = vectorName;
    }

    public static VectorQuery of(float[] vector, int topK) {
        return new VectorQuery(vector, topK, null, false, true, null, null);
    }

    public static Builder builder(float[] vector, int topK) {
        return new Builder(vector, topK);
    }

    public float[] vector() { return vector; }
    public int topK() { return topK; }
    public Object filter() { return filter; }
    public boolean includeVector() { return includeVector; }
    public boolean includePayload() { return includePayload; }
    public Map<String, Object> options() { return options; }
    /** Named vector field when the collection has multiple vector columns. */
    public String vectorName() { return vectorName; }

    @SuppressWarnings("unchecked")
    public <T> T option(String key, T defaultValue) {
        Object v = options.get(key);
        if (v == null) return defaultValue;
        return (T) v;
    }

    public static final class Builder {
        private final float[] vector;
        private final int topK;
        private Object filter;
        private boolean includeVector;
        private boolean includePayload = true;
        private final Map<String, Object> options = new LinkedHashMap<>();
        private String vectorName;

        Builder(float[] vector, int topK) {
            this.vector = Objects.requireNonNull(vector, "vector");
            this.topK = topK;
        }

        public Builder filter(Object filter) { this.filter = filter; return this; }
        public Builder includeVector(boolean v) { this.includeVector = v; return this; }
        public Builder includePayload(boolean v) { this.includePayload = v; return this; }
        public Builder vectorName(String name) { this.vectorName = name; return this; }
        public Builder option(String key, Object value) {
            if (key != null) options.put(key, value);
            return this;
        }
        /** ef / nprobe / num_candidates style search-time parameter. */
        public Builder ef(int ef) { return option("ef", ef); }
        public Builder nprobe(int n) { return option("nprobe", n); }

        public VectorQuery build() {
            return new VectorQuery(vector, topK, filter, includeVector, includePayload, options, vectorName);
        }
    }
}
