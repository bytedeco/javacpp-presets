/*
 * Single online feature row keyed by entity.
 */
package org.bytedeco.pytorch.utils.feature.online;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Immutable online KV value. */
public final class OnlineFeatureRow {

    private final String project;
    private final String viewName;
    private final String entityKey;
    private final Map<String, Object> values;
    private final long eventTimestampMs;
    private final long writtenAtMs;
    private final long ttlMs;

    private OnlineFeatureRow(Builder b) {
        this.project = b.project != null ? b.project : "default";
        this.viewName = Objects.requireNonNull(b.viewName, "viewName");
        this.entityKey = Objects.requireNonNull(b.entityKey, "entityKey");
        this.values = Collections.unmodifiableMap(new LinkedHashMap<>(b.values));
        this.eventTimestampMs = b.eventTimestampMs;
        this.writtenAtMs = b.writtenAtMs > 0 ? b.writtenAtMs : System.currentTimeMillis();
        this.ttlMs = b.ttlMs;
    }

    public static Builder builder(String viewName, String entityKey) {
        return new Builder(viewName, entityKey);
    }

    public String project() {
        return project;
    }

    public String viewName() {
        return viewName;
    }

    public String entityKey() {
        return entityKey;
    }

    public Map<String, Object> values() {
        return values;
    }

    public Object get(String feature) {
        return values.get(feature);
    }

    public long eventTimestampMs() {
        return eventTimestampMs;
    }

    public long writtenAtMs() {
        return writtenAtMs;
    }

    public long ttlMs() {
        return ttlMs;
    }

    public boolean isExpired(long nowMs) {
        if (ttlMs <= 0) return false;
        return nowMs - eventTimestampMs > ttlMs;
    }

    public String storageKey() {
        return project + "#" + viewName + "#" + entityKey;
    }

    @Override
    public String toString() {
        return "OnlineFeatureRow{" + storageKey() + ", n=" + values.size() + "}";
    }

    public static final class Builder {
        private String project = "default";
        private final String viewName;
        private final String entityKey;
        private final Map<String, Object> values = new LinkedHashMap<>();
        private long eventTimestampMs;
        private long writtenAtMs;
        private long ttlMs;

        private Builder(String viewName, String entityKey) {
            this.viewName = viewName;
            this.entityKey = entityKey;
        }

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder values(Map<String, Object> values) {
            if (values != null) this.values.putAll(values);
            return this;
        }

        public Builder put(String feature, Object value) {
            values.put(feature, value);
            return this;
        }

        public Builder eventTimestampMs(long eventTimestampMs) {
            this.eventTimestampMs = eventTimestampMs;
            return this;
        }

        public Builder writtenAtMs(long writtenAtMs) {
            this.writtenAtMs = writtenAtMs;
            return this;
        }

        public Builder ttlMs(long ttlMs) {
            this.ttlMs = ttlMs;
            return this;
        }

        public OnlineFeatureRow build() {
            return new OnlineFeatureRow(this);
        }
    }
}
