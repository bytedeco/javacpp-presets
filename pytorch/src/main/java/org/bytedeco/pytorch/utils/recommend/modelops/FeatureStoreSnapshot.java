/*
 * Feature store snapshot contract for ranking feature consistency.
 *
 * Online/offline training-serving skew is a top production incident class
 * at Meta, Google, Alibaba, ByteDance. This utility captures:
 *   - feature schema version
 *   - point-in-time feature values for a request
 *   - join keys / event time for replay
 * so offline training and online serving can be audited against the same view.
 */
package org.bytedeco.pytorch.utils.recommend.modelops;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Point-in-time feature snapshot for one ranking request. */
public final class FeatureStoreSnapshot {

    private final String snapshotId;
    private final String schemaVersion;
    private final String userId;
    private final long eventTimeMs;
    private final long ingestTimeMs;
    private final Map<String, Double> denseFeatures;
    private final Map<String, Long> sparseFeatures;
    private final Map<String, long[]> sequenceFeatures;
    private final Map<String, String> meta;

    private FeatureStoreSnapshot(Builder b) {
        this.snapshotId = Objects.requireNonNull(b.snapshotId, "snapshotId");
        this.schemaVersion = b.schemaVersion != null ? b.schemaVersion : "v1";
        this.userId = b.userId != null ? b.userId : "";
        this.eventTimeMs = b.eventTimeMs;
        this.ingestTimeMs = b.ingestTimeMs > 0 ? b.ingestTimeMs : System.currentTimeMillis();
        this.denseFeatures = Collections.unmodifiableMap(new LinkedHashMap<>(b.denseFeatures));
        this.sparseFeatures = Collections.unmodifiableMap(new LinkedHashMap<>(b.sparseFeatures));
        this.sequenceFeatures = Collections.unmodifiableMap(new LinkedHashMap<>(b.sequenceFeatures));
        this.meta = Collections.unmodifiableMap(new LinkedHashMap<>(b.meta));
    }

    public static Builder builder(String snapshotId) {
        return new Builder(snapshotId);
    }

    public String snapshotId() {
        return snapshotId;
    }

    public String schemaVersion() {
        return schemaVersion;
    }

    public String userId() {
        return userId;
    }

    public long eventTimeMs() {
        return eventTimeMs;
    }

    public long ingestTimeMs() {
        return ingestTimeMs;
    }

    /** Feature freshness lag: ingest - event (ms). */
    public long freshnessLagMs() {
        return Math.max(0L, ingestTimeMs - eventTimeMs);
    }

    public Map<String, Double> denseFeatures() {
        return denseFeatures;
    }

    public Map<String, Long> sparseFeatures() {
        return sparseFeatures;
    }

    public Map<String, long[]> sequenceFeatures() {
        return sequenceFeatures;
    }

    public Map<String, String> meta() {
        return meta;
    }

    public double dense(String key, double defaultValue) {
        return denseFeatures.getOrDefault(key, defaultValue);
    }

    public long sparse(String key, long defaultValue) {
        return sparseFeatures.getOrDefault(key, defaultValue);
    }

    /**
     * Detect training-serving skew on dense features vs a training-time snapshot.
     *
     * @return map of feature -> absolute relative delta; empty if aligned
     */
    public Map<String, Double> denseSkewAgainst(FeatureStoreSnapshot trainingView, double relTolerance) {
        Objects.requireNonNull(trainingView, "trainingView");
        Map<String, Double> skew = new LinkedHashMap<>();
        for (Map.Entry<String, Double> e : denseFeatures.entrySet()) {
            Double t = trainingView.denseFeatures.get(e.getKey());
            if (t == null) {
                skew.put(e.getKey(), Double.NaN); // missing in training view
                continue;
            }
            double denom = Math.max(Math.abs(t), 1e-9);
            double rel = Math.abs(e.getValue() - t) / denom;
            if (rel > relTolerance) {
                skew.put(e.getKey(), rel);
            }
        }
        for (String key : trainingView.denseFeatures.keySet()) {
            if (!denseFeatures.containsKey(key)) {
                skew.put(key, Double.NaN); // missing online
            }
        }
        return skew;
    }

    @Override
    public String toString() {
        return "FeatureStoreSnapshot{id=" + snapshotId + ", schema=" + schemaVersion
                + ", user=" + userId + ", dense=" + denseFeatures.size()
                + ", sparse=" + sparseFeatures.size() + ", lagMs=" + freshnessLagMs() + "}";
    }

    public static final class Builder {
        private final String snapshotId;
        private String schemaVersion;
        private String userId;
        private long eventTimeMs;
        private long ingestTimeMs;
        private final Map<String, Double> denseFeatures = new LinkedHashMap<>();
        private final Map<String, Long> sparseFeatures = new LinkedHashMap<>();
        private final Map<String, long[]> sequenceFeatures = new LinkedHashMap<>();
        private final Map<String, String> meta = new LinkedHashMap<>();

        private Builder(String snapshotId) {
            this.snapshotId = snapshotId;
        }

        public Builder schemaVersion(String schemaVersion) {
            this.schemaVersion = schemaVersion;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder eventTimeMs(long eventTimeMs) {
            this.eventTimeMs = eventTimeMs;
            return this;
        }

        public Builder ingestTimeMs(long ingestTimeMs) {
            this.ingestTimeMs = ingestTimeMs;
            return this;
        }

        public Builder dense(String key, double value) {
            denseFeatures.put(key, value);
            return this;
        }

        public Builder dense(Map<String, Double> values) {
            if (values != null) denseFeatures.putAll(values);
            return this;
        }

        public Builder sparse(String key, long value) {
            sparseFeatures.put(key, value);
            return this;
        }

        public Builder sparse(Map<String, Long> values) {
            if (values != null) sparseFeatures.putAll(values);
            return this;
        }

        public Builder sequence(String key, long[] value) {
            sequenceFeatures.put(key, value);
            return this;
        }

        public Builder meta(String key, String value) {
            meta.put(key, value);
            return this;
        }

        public FeatureStoreSnapshot build() {
            return new FeatureStoreSnapshot(this);
        }
    }
}
