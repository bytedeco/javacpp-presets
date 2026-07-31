/*
 * Feature vector bag for one entity key set — dense / sparse / sequence / embedding.
 */
package org.bytedeco.pytorch.feature.serving;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Mutable builder + immutable snapshot of served features. */
public final class FeatureVector {

    private final Map<String, Object> entities;
    private final Map<String, Double> dense;
    private final Map<String, Long> sparse;
    private final Map<String, long[]> sequences;
    private final Map<String, float[]> embeddings;
    private final Map<String, Object> raw;
    private final Map<String, String> meta;

    private FeatureVector(Builder b) {
        this.entities = Collections.unmodifiableMap(new LinkedHashMap<>(b.entities));
        this.dense = Collections.unmodifiableMap(new LinkedHashMap<>(b.dense));
        this.sparse = Collections.unmodifiableMap(new LinkedHashMap<>(b.sparse));
        this.sequences = Collections.unmodifiableMap(new LinkedHashMap<>(b.sequences));
        this.embeddings = Collections.unmodifiableMap(new LinkedHashMap<>(b.embeddings));
        this.raw = Collections.unmodifiableMap(new LinkedHashMap<>(b.raw));
        this.meta = Collections.unmodifiableMap(new LinkedHashMap<>(b.meta));
    }

    public static Builder builder() {
        return new Builder();
    }

    public Map<String, Object> entities() { return entities; }
    public Map<String, Double> dense() { return dense; }
    public Map<String, Long> sparse() { return sparse; }
    public Map<String, long[]> sequences() { return sequences; }
    public Map<String, float[]> embeddings() { return embeddings; }
    public Map<String, Object> raw() { return raw; }
    public Map<String, String> meta() { return meta; }

    public double denseOr(String key, double d) {
        return dense.getOrDefault(key, d);
    }

    public long sparseOr(String key, long d) {
        return sparse.getOrDefault(key, d);
    }

    public Object raw(String key) {
        return raw.get(key);
    }

    public int size() {
        return dense.size() + sparse.size() + sequences.size() + embeddings.size();
    }

    /** Merge another vector (other wins on key conflict for typed maps). */
    public FeatureVector merge(FeatureVector other) {
        Objects.requireNonNull(other, "other");
        Builder b = builder();
        b.entities.putAll(this.entities);
        b.entities.putAll(other.entities);
        b.dense.putAll(this.dense);
        b.dense.putAll(other.dense);
        b.sparse.putAll(this.sparse);
        b.sparse.putAll(other.sparse);
        b.sequences.putAll(this.sequences);
        b.sequences.putAll(other.sequences);
        b.embeddings.putAll(this.embeddings);
        b.embeddings.putAll(other.embeddings);
        b.raw.putAll(this.raw);
        b.raw.putAll(other.raw);
        b.meta.putAll(this.meta);
        b.meta.putAll(other.meta);
        return b.build();
    }

    /**
     * Classify a raw feature value into typed bags.
     */
    public static void putTyped(Builder b, String name, Object value) {
        if (value == null) {
            b.raw.put(name, null);
            return;
        }
        b.raw.put(name, value);
        if (value instanceof float[]) {
            b.embeddings.put(name, (float[]) value);
            return;
        }
        if (value instanceof double[]) {
            double[] d = (double[]) value;
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            b.embeddings.put(name, f);
            return;
        }
        if (value instanceof long[]) {
            b.sequences.put(name, (long[]) value);
            return;
        }
        if (value instanceof int[]) {
            int[] a = (int[]) value;
            long[] l = new long[a.length];
            for (int i = 0; i < a.length; i++) l[i] = a[i];
            b.sequences.put(name, l);
            return;
        }
        if (value instanceof Number) {
            Number n = (Number) value;
            if (n instanceof Double || n instanceof Float) {
                b.dense.put(name, n.doubleValue());
            } else {
                b.sparse.put(name, n.longValue());
                b.dense.put(name, n.doubleValue());
            }
            return;
        }
        if (value instanceof Boolean) {
            b.sparse.put(name, ((Boolean) value) ? 1L : 0L);
            b.dense.put(name, ((Boolean) value) ? 1.0 : 0.0);
            return;
        }
        // strings / others stay in raw only
    }

    @Override
    public String toString() {
        return "FeatureVector{dense=" + dense.size()
                + ", sparse=" + sparse.size()
                + ", seq=" + sequences.size()
                + ", emb=" + embeddings.size()
                + ", raw=" + raw.size() + "}";
    }

    public static final class Builder {
        private final Map<String, Object> entities = new LinkedHashMap<>();
        private final Map<String, Double> dense = new LinkedHashMap<>();
        private final Map<String, Long> sparse = new LinkedHashMap<>();
        private final Map<String, long[]> sequences = new LinkedHashMap<>();
        private final Map<String, float[]> embeddings = new LinkedHashMap<>();
        private final Map<String, Object> raw = new LinkedHashMap<>();
        private final Map<String, String> meta = new LinkedHashMap<>();

        public Builder entity(String k, Object v) {
            entities.put(k, v);
            return this;
        }

        public Builder entities(Map<String, Object> more) {
            if (more != null) entities.putAll(more);
            return this;
        }

        public Builder dense(String k, double v) {
            dense.put(k, v);
            raw.put(k, v);
            return this;
        }

        public Builder sparse(String k, long v) {
            sparse.put(k, v);
            raw.put(k, v);
            return this;
        }

        public Builder sequence(String k, long[] v) {
            sequences.put(k, v);
            raw.put(k, v);
            return this;
        }

        public Builder embedding(String k, float[] v) {
            embeddings.put(k, v);
            raw.put(k, v);
            return this;
        }

        public Builder raw(String k, Object v) {
            putTyped(this, k, v);
            return this;
        }

        public Builder rawAll(Map<String, Object> values) {
            if (values != null) {
                for (Map.Entry<String, Object> e : values.entrySet()) {
                    putTyped(this, e.getKey(), e.getValue());
                }
            }
            return this;
        }

        public Builder meta(String k, String v) {
            if (k != null && v != null) meta.put(k, v);
            return this;
        }

        public FeatureVector build() {
            return new FeatureVector(this);
        }
    }
}
