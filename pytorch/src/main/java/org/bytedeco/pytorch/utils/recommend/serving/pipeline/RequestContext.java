/*
 * Per-request context carried through the ranking cascade.
 *
 * Contains:
 *   - user / device / session identity
 *   - scene / page / request meta
 *   - experiment parameter overlays
 *   - deadline / remaining budget
 *   - debug flags
 */
package org.bytedeco.pytorch.utils.recommend.serving.pipeline;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/** Mutable request-scoped context for the ranking pipeline. */
public final class RequestContext {

    private final String requestId;
    private final String userId;
    private final String deviceId;
    private final String scene;
    private final long startEpochMs;
    private final long deadlineEpochMs;
    private final Map<String, String> experimentParams;
    private final Map<String, String> features;
    private final Map<String, Object> attributes;
    private final boolean debug;

    private RequestContext(Builder b) {
        this.requestId = Objects.requireNonNull(b.requestId, "requestId");
        this.userId = b.userId != null ? b.userId : "";
        this.deviceId = b.deviceId != null ? b.deviceId : "";
        this.scene = b.scene != null ? b.scene : "default";
        this.startEpochMs = b.startEpochMs > 0 ? b.startEpochMs : System.currentTimeMillis();
        this.deadlineEpochMs = b.deadlineEpochMs > 0
                ? b.deadlineEpochMs
                : this.startEpochMs + b.timeoutMs;
        this.experimentParams = Collections.unmodifiableMap(new LinkedHashMap<>(b.experimentParams));
        this.features = Collections.unmodifiableMap(new LinkedHashMap<>(b.features));
        this.attributes = new ConcurrentHashMap<>(b.attributes);
        this.debug = b.debug;
    }

    public static Builder builder(String requestId) {
        return new Builder(requestId);
    }

    public String requestId() {
        return requestId;
    }

    public String userId() {
        return userId;
    }

    public String deviceId() {
        return deviceId;
    }

    /** Primary diversion key: prefer userId, fall back to deviceId. */
    public String diversionKey() {
        if (userId != null && !userId.isEmpty()) {
            return userId;
        }
        return deviceId;
    }

    public String scene() {
        return scene;
    }

    public long startEpochMs() {
        return startEpochMs;
    }

    public long deadlineEpochMs() {
        return deadlineEpochMs;
    }

    public long remainingBudgetMs() {
        return Math.max(0L, deadlineEpochMs - System.currentTimeMillis());
    }

    public boolean deadlineExceeded() {
        return System.currentTimeMillis() >= deadlineEpochMs;
    }

    public Map<String, String> experimentParams() {
        return experimentParams;
    }

    public String expParam(String key, String defaultValue) {
        return experimentParams.getOrDefault(key, defaultValue);
    }

    public int expParamInt(String key, int defaultValue) {
        String v = experimentParams.get(key);
        if (v == null) return defaultValue;
        try {
            return Integer.parseInt(v);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public double expParamDouble(String key, double defaultValue) {
        String v = experimentParams.get(key);
        if (v == null) return defaultValue;
        try {
            return Double.parseDouble(v);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public Map<String, String> features() {
        return features;
    }

    public void setAttribute(String key, Object value) {
        if (value == null) {
            attributes.remove(key);
        } else {
            attributes.put(key, value);
        }
    }

    @SuppressWarnings("unchecked")
    public <T> T getAttribute(String key) {
        return (T) attributes.get(key);
    }

    public boolean debug() {
        return debug;
    }

    public static final class Builder {
        private final String requestId;
        private String userId;
        private String deviceId;
        private String scene;
        private long startEpochMs;
        private long deadlineEpochMs;
        private long timeoutMs = 200L; // typical recsys end-to-end budget
        private final Map<String, String> experimentParams = new LinkedHashMap<>();
        private final Map<String, String> features = new LinkedHashMap<>();
        private final Map<String, Object> attributes = new LinkedHashMap<>();
        private boolean debug;

        private Builder(String requestId) {
            this.requestId = requestId;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder deviceId(String deviceId) {
            this.deviceId = deviceId;
            return this;
        }

        public Builder scene(String scene) {
            this.scene = scene;
            return this;
        }

        public Builder startEpochMs(long startEpochMs) {
            this.startEpochMs = startEpochMs;
            return this;
        }

        public Builder deadlineEpochMs(long deadlineEpochMs) {
            this.deadlineEpochMs = deadlineEpochMs;
            return this;
        }

        public Builder timeoutMs(long timeoutMs) {
            this.timeoutMs = timeoutMs;
            return this;
        }

        public Builder experimentParam(String key, String value) {
            this.experimentParams.put(key, value);
            return this;
        }

        public Builder experimentParams(Map<String, String> params) {
            if (params != null) {
                this.experimentParams.putAll(params);
            }
            return this;
        }

        public Builder feature(String key, String value) {
            this.features.put(key, value);
            return this;
        }

        public Builder features(Map<String, String> features) {
            if (features != null) {
                this.features.putAll(features);
            }
            return this;
        }

        public Builder attribute(String key, Object value) {
            this.attributes.put(key, value);
            return this;
        }

        public Builder debug(boolean debug) {
            this.debug = debug;
            return this;
        }

        public RequestContext build() {
            return new RequestContext(this);
        }
    }
}
