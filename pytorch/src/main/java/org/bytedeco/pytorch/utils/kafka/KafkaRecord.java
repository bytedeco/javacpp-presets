package org.bytedeco.pytorch.utils.kafka;

import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable Kafka message model used across produce / consume / offline dump paths.
 *
 * <p>Value may be raw {@code byte[]}, a UTF-8 {@link String}, a decoded
 * {@code Map<String,Object>} (JSON object), or any object that {@link KafkaSerde}
 * can encode.
 */
public final class KafkaRecord {

    private final String topic;
    private final Integer partition;
    private final Long offset;
    private final Long timestamp;
    private final String key;
    private final Object value;
    private final Map<String, String> headers;

    private KafkaRecord(Builder b) {
        this.topic = b.topic;
        this.partition = b.partition;
        this.offset = b.offset;
        this.timestamp = b.timestamp;
        this.key = b.key;
        this.value = b.value;
        this.headers = b.headers == null || b.headers.isEmpty()
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.headers));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static KafkaRecord of(String topic, String key, Object value) {
        return builder().topic(topic).key(key).value(value).build();
    }

    public static KafkaRecord of(String topic, Object value) {
        return builder().topic(topic).value(value).build();
    }

    public String topic() {
        return topic;
    }

    public Integer partition() {
        return partition;
    }

    public Long offset() {
        return offset;
    }

    public Long timestamp() {
        return timestamp;
    }

    public String key() {
        return key;
    }

    public Object value() {
        return value;
    }

    public Map<String, String> headers() {
        return headers;
    }

    /** UTF-8 view of value when it is a String or byte[]. */
    public String valueAsString() {
        if (value == null) return null;
        if (value instanceof String s) return s;
        if (value instanceof byte[] b) return new String(b, StandardCharsets.UTF_8);
        return String.valueOf(value);
    }

    public byte[] valueAsBytes() {
        return KafkaSerde.encodeValue(value);
    }

    /**
     * Flatten to a row map suitable for {@code DataFrame.fromRecords}.
     * Metadata columns are prefixed with {@code __} when {@code includeMetadata} is true.
     */
    public Map<String, Object> toRowMap(boolean includeMetadata) {
        Map<String, Object> row = new LinkedHashMap<>();
        if (includeMetadata) {
            if (topic != null) row.put("__topic", topic);
            if (partition != null) row.put("__partition", partition.longValue());
            if (offset != null) row.put("__offset", offset);
            if (timestamp != null) row.put("__timestamp", timestamp);
            if (key != null) row.put("__key", key);
            if (!headers.isEmpty()) row.put("__headers", new LinkedHashMap<>(headers));
        }
        Object decoded = KafkaSerde.decodeToObject(value);
        if (decoded instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (e.getKey() != null) {
                    row.put(String.valueOf(e.getKey()), e.getValue());
                }
            }
        } else if (decoded != null) {
            row.put("value", decoded);
        }
        if (!includeMetadata && key != null && !row.containsKey("key")) {
            // keep key accessible when metadata is off but row has no "key" field
            row.putIfAbsent("_key", key);
        }
        return row;
    }

    @Override
    public String toString() {
        return "KafkaRecord{topic=" + topic
                + ", partition=" + partition
                + ", offset=" + offset
                + ", key=" + key
                + ", valueType=" + (value == null ? "null" : value.getClass().getSimpleName())
                + ", headers=" + headers.size()
                + '}';
    }

    public static final class Builder {
        private String topic;
        private Integer partition;
        private Long offset;
        private Long timestamp;
        private String key;
        private Object value;
        private Map<String, String> headers;

        public Builder topic(String topic) {
            this.topic = topic;
            return this;
        }

        public Builder partition(Integer partition) {
            this.partition = partition;
            return this;
        }

        public Builder partition(int partition) {
            this.partition = partition;
            return this;
        }

        public Builder offset(Long offset) {
            this.offset = offset;
            return this;
        }

        public Builder offset(long offset) {
            this.offset = offset;
            return this;
        }

        public Builder timestamp(Long timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder timestamp(long timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder key(String key) {
            this.key = key;
            return this;
        }

        public Builder value(Object value) {
            this.value = value;
            return this;
        }

        public Builder headers(Map<String, String> headers) {
            this.headers = headers;
            return this;
        }

        public Builder header(String name, String value) {
            if (this.headers == null) this.headers = new LinkedHashMap<>();
            this.headers.put(Objects.requireNonNull(name, "header name"), value);
            return this;
        }

        public KafkaRecord build() {
            return new KafkaRecord(this);
        }
    }
}
