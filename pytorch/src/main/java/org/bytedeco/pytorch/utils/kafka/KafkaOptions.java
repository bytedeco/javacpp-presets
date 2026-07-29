package org.bytedeco.pytorch.utils.kafka;

import org.apache.kafka.clients.CommonClientConfigs;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.config.SaslConfigs;
import org.apache.kafka.common.config.SslConfigs;
import org.apache.kafka.common.serialization.ByteArrayDeserializer;
import org.apache.kafka.common.serialization.ByteArraySerializer;

import java.time.Duration;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**
 * Immutable options for Kafka produce / consume / admin / DataFrame I/O.
 *
 * <p>Enterprise defaults target high-QPS feature-log and online-training pipelines:
 * idempotent producer + {@code acks=all}, LZ4 compression, manual consumer commit,
 * cooperative-sticky assignment, {@code read_committed} isolation.
 *
 * <pre>{@code
 * KafkaOptions opts = KafkaOptions.builder()
 *     .bootstrapServers("kafka-1:9092,kafka-2:9092")
 *     .clientId("jnitorch-rec")
 *     .topic(TopicOpts.builder().name("rec.feature.log").partitions(64).replicationFactor((short) 3).build())
 *     .keyColumn("user_id")
 *     .valueFormat(ValueFormat.JSON)
 *     .producer(ProducerOpts.defaults())
 *     .consumer(ConsumerOpts.builder().groupId("fe-online-v1").maxPollRecords(2000).build())
 *     .build();
 * }</pre>
 */
public final class KafkaOptions {

    public enum ValueFormat {
        /** UTF-8 JSON object per message (DataFrame row default). */
        JSON,
        /** Alias of JSON used when writing JSONL offline dumps. */
        JSONL_ROW,
        /** Raw UTF-8 string payload. */
        STRING,
        /** Opaque bytes. */
        BYTES,
        /** Comma-separated row values. */
        CSV_ROW
    }

    public enum Acks {
        ZERO("0"),
        ONE("1"),
        ALL("all");

        private final String value;

        Acks(String value) {
            this.value = value;
        }

        public String value() {
            return value;
        }
    }

    public enum Compression {
        NONE("none"),
        GZIP("gzip"),
        SNAPPY("snappy"),
        LZ4("lz4"),
        ZSTD("zstd");

        private final String value;

        Compression(String value) {
            this.value = value;
        }

        public String value() {
            return value;
        }
    }

    public enum AutoOffsetReset {
        EARLIEST("earliest"),
        LATEST("latest"),
        NONE("none");

        private final String value;

        AutoOffsetReset(String value) {
            this.value = value;
        }

        public String value() {
            return value;
        }
    }

    public enum Isolation {
        READ_UNCOMMITTED("read_uncommitted"),
        READ_COMMITTED("read_committed");

        private final String value;

        Isolation(String value) {
            this.value = value;
        }

        public String value() {
            return value;
        }
    }

    public enum SecurityProtocol {
        PLAINTEXT,
        SSL,
        SASL_PLAINTEXT,
        SASL_SSL
    }

    // ── fields ───────────────────────────────────────────────────────────────

    private final String bootstrapServers;
    private final String clientId;
    private final SecurityOpts security;
    private final ProducerOpts producer;
    private final ConsumerOpts consumer;
    private final TopicOpts topic;
    private final String keyColumn;
    private final String partitionColumn;
    private final String timestampColumn;
    private final ValueFormat valueFormat;
    private final Map<String, String> headers;
    private final String dlqTopic;
    private final boolean includeMetadata;
    private final Duration pollTimeout;
    private final int maxRecords;
    private final Map<String, String> extra;
    private final boolean ifNotExists;

    private KafkaOptions(Builder b) {
        this.bootstrapServers = b.bootstrapServers == null || b.bootstrapServers.isBlank()
                ? "localhost:9092" : b.bootstrapServers.trim();
        this.clientId = b.clientId;
        this.security = b.security == null ? SecurityOpts.plaintext() : b.security;
        this.producer = b.producer == null ? ProducerOpts.defaults() : b.producer;
        this.consumer = b.consumer == null ? ConsumerOpts.defaults() : b.consumer;
        this.topic = b.topic;
        this.keyColumn = b.keyColumn;
        this.partitionColumn = b.partitionColumn;
        this.timestampColumn = b.timestampColumn;
        this.valueFormat = b.valueFormat == null ? ValueFormat.JSON : b.valueFormat;
        this.headers = b.headers == null || b.headers.isEmpty()
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.headers));
        this.dlqTopic = b.dlqTopic;
        this.includeMetadata = b.includeMetadata;
        this.pollTimeout = b.pollTimeout == null ? Duration.ofMillis(1000) : b.pollTimeout;
        this.maxRecords = Math.max(0, b.maxRecords);
        this.extra = b.extra == null || b.extra.isEmpty()
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
        this.ifNotExists = b.ifNotExists;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static KafkaOptions defaults() {
        return builder().build();
    }

    public static KafkaOptions producer(String bootstrap, String topic) {
        return builder()
                .bootstrapServers(bootstrap)
                .topic(TopicOpts.of(topic))
                .build();
    }

    public static KafkaOptions consumer(String bootstrap, String topic, String groupId) {
        return builder()
                .bootstrapServers(bootstrap)
                .topic(TopicOpts.of(topic))
                .consumer(ConsumerOpts.builder().groupId(groupId).build())
                .build();
    }

    public static KafkaOptions topic(String name, int partitions, short replicationFactor) {
        return builder()
                .topic(TopicOpts.builder()
                        .name(name)
                        .partitions(partitions)
                        .replicationFactor(replicationFactor)
                        .build())
                .build();
    }

    // ── accessors ────────────────────────────────────────────────────────────

    public String bootstrapServers() {
        return bootstrapServers;
    }

    public String clientId() {
        return clientId;
    }

    public SecurityOpts security() {
        return security;
    }

    public ProducerOpts producer() {
        return producer;
    }

    public ConsumerOpts consumer() {
        return consumer;
    }

    public TopicOpts topic() {
        return topic;
    }

    public String topicName() {
        return topic == null ? null : topic.name();
    }

    public String keyColumn() {
        return keyColumn;
    }

    public String partitionColumn() {
        return partitionColumn;
    }

    public String timestampColumn() {
        return timestampColumn;
    }

    public ValueFormat valueFormat() {
        return valueFormat;
    }

    public Map<String, String> headers() {
        return headers;
    }

    public String dlqTopic() {
        return dlqTopic;
    }

    public boolean includeMetadata() {
        return includeMetadata;
    }

    public Duration pollTimeout() {
        return pollTimeout;
    }

    public int maxRecords() {
        return maxRecords;
    }

    public Map<String, String> extra() {
        return extra;
    }

    public boolean ifNotExists() {
        return ifNotExists;
    }

    public Builder toBuilder() {
        return builder()
                .bootstrapServers(bootstrapServers)
                .clientId(clientId)
                .security(security)
                .producer(producer)
                .consumer(consumer)
                .topic(topic)
                .keyColumn(keyColumn)
                .partitionColumn(partitionColumn)
                .timestampColumn(timestampColumn)
                .valueFormat(valueFormat)
                .headers(headers.isEmpty() ? null : new LinkedHashMap<>(headers))
                .dlqTopic(dlqTopic)
                .includeMetadata(includeMetadata)
                .pollTimeout(pollTimeout)
                .maxRecords(maxRecords)
                .extra(extra.isEmpty() ? null : new LinkedHashMap<>(extra))
                .ifNotExists(ifNotExists);
    }

    // ── Properties builders ──────────────────────────────────────────────────

    public Properties commonProperties() {
        Properties p = new Properties();
        p.put(CommonClientConfigs.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        if (clientId != null && !clientId.isBlank()) {
            p.put(CommonClientConfigs.CLIENT_ID_CONFIG, clientId);
        }
        security.apply(p);
        for (Map.Entry<String, String> e : extra.entrySet()) {
            p.put(e.getKey(), e.getValue());
        }
        return p;
    }

    public Properties producerProperties() {
        Properties p = commonProperties();
        producer.apply(p, clientId);
        p.putIfAbsent(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, ByteArraySerializer.class.getName());
        p.putIfAbsent(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, ByteArraySerializer.class.getName());
        return p;
    }

    public Properties consumerProperties() {
        Properties p = commonProperties();
        consumer.apply(p, clientId);
        p.putIfAbsent(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, ByteArrayDeserializer.class.getName());
        p.putIfAbsent(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, ByteArrayDeserializer.class.getName());
        return p;
    }

    public Properties adminProperties() {
        return commonProperties();
    }

    // ── Builder ──────────────────────────────────────────────────────────────

    public static final class Builder {
        private String bootstrapServers = "localhost:9092";
        private String clientId;
        private SecurityOpts security;
        private ProducerOpts producer;
        private ConsumerOpts consumer;
        private TopicOpts topic;
        private String keyColumn;
        private String partitionColumn;
        private String timestampColumn;
        private ValueFormat valueFormat = ValueFormat.JSON;
        private Map<String, String> headers;
        private String dlqTopic;
        private boolean includeMetadata = true;
        private Duration pollTimeout = Duration.ofMillis(1000);
        private int maxRecords = 0;
        private Map<String, String> extra;
        private boolean ifNotExists = true;

        public Builder bootstrapServers(String bootstrapServers) {
            this.bootstrapServers = bootstrapServers;
            return this;
        }

        public Builder clientId(String clientId) {
            this.clientId = clientId;
            return this;
        }

        public Builder security(SecurityOpts security) {
            this.security = security;
            return this;
        }

        public Builder producer(ProducerOpts producer) {
            this.producer = producer;
            return this;
        }

        public Builder consumer(ConsumerOpts consumer) {
            this.consumer = consumer;
            return this;
        }

        public Builder topic(TopicOpts topic) {
            this.topic = topic;
            return this;
        }

        public Builder topic(String name) {
            this.topic = TopicOpts.of(name);
            return this;
        }

        public Builder topic(String name, int partitions, short replicationFactor) {
            this.topic = TopicOpts.builder()
                    .name(name)
                    .partitions(partitions)
                    .replicationFactor(replicationFactor)
                    .build();
            return this;
        }

        public Builder keyColumn(String keyColumn) {
            this.keyColumn = keyColumn;
            return this;
        }

        public Builder partitionColumn(String partitionColumn) {
            this.partitionColumn = partitionColumn;
            return this;
        }

        public Builder timestampColumn(String timestampColumn) {
            this.timestampColumn = timestampColumn;
            return this;
        }

        public Builder valueFormat(ValueFormat valueFormat) {
            this.valueFormat = valueFormat;
            return this;
        }

        public Builder headers(Map<String, String> headers) {
            this.headers = headers;
            return this;
        }

        public Builder header(String name, String value) {
            if (this.headers == null) this.headers = new LinkedHashMap<>();
            this.headers.put(name, value);
            return this;
        }

        public Builder dlqTopic(String dlqTopic) {
            this.dlqTopic = dlqTopic;
            return this;
        }

        public Builder includeMetadata(boolean includeMetadata) {
            this.includeMetadata = includeMetadata;
            return this;
        }

        public Builder pollTimeout(Duration pollTimeout) {
            this.pollTimeout = pollTimeout;
            return this;
        }

        public Builder maxRecords(int maxRecords) {
            this.maxRecords = maxRecords;
            return this;
        }

        public Builder extra(Map<String, String> extra) {
            this.extra = extra;
            return this;
        }

        public Builder extra(String key, String value) {
            if (this.extra == null) this.extra = new LinkedHashMap<>();
            this.extra.put(key, value);
            return this;
        }

        public Builder ifNotExists(boolean ifNotExists) {
            this.ifNotExists = ifNotExists;
            return this;
        }

        public Builder groupId(String groupId) {
            ConsumerOpts base = this.consumer == null ? ConsumerOpts.defaults() : this.consumer;
            this.consumer = base.toBuilder().groupId(groupId).build();
            return this;
        }

        public KafkaOptions build() {
            return new KafkaOptions(this);
        }
    }

    // ── nested: ProducerOpts ─────────────────────────────────────────────────

    /**
     * Producer knobs. Defaults: {@code acks=all}, idempotent, LZ4, linger 10ms,
     * batch 256KiB, buffer 64MiB, retries MAX, max.in.flight=5.
     */
    public static final class ProducerOpts {
        private final Acks acks;
        private final boolean idempotent;
        private final Compression compression;
        private final int lingerMs;
        private final int batchSize;
        private final long bufferMemory;
        private final int retries;
        private final int maxInFlight;
        private final String transactionalId;
        private final int requestTimeoutMs;
        private final int deliveryTimeoutMs;
        private final Map<String, String> extra;

        private ProducerOpts(Builder b) {
            this.acks = b.acks == null ? Acks.ALL : b.acks;
            this.idempotent = b.idempotent;
            this.compression = b.compression == null ? Compression.LZ4 : b.compression;
            this.lingerMs = Math.max(0, b.lingerMs);
            this.batchSize = Math.max(0, b.batchSize);
            this.bufferMemory = Math.max(0L, b.bufferMemory);
            this.retries = b.retries < 0 ? Integer.MAX_VALUE : b.retries;
            this.maxInFlight = Math.max(1, b.maxInFlight);
            this.transactionalId = b.transactionalId;
            this.requestTimeoutMs = Math.max(1, b.requestTimeoutMs);
            this.deliveryTimeoutMs = Math.max(1, b.deliveryTimeoutMs);
            this.extra = b.extra == null || b.extra.isEmpty()
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
        }

        public static Builder builder() {
            return new Builder();
        }

        public static ProducerOpts defaults() {
            return builder().build();
        }

        public Acks acks() {
            return acks;
        }

        public boolean idempotent() {
            return idempotent;
        }

        public Compression compression() {
            return compression;
        }

        public int lingerMs() {
            return lingerMs;
        }

        public int batchSize() {
            return batchSize;
        }

        public long bufferMemory() {
            return bufferMemory;
        }

        public int retries() {
            return retries;
        }

        public int maxInFlight() {
            return maxInFlight;
        }

        public String transactionalId() {
            return transactionalId;
        }

        public boolean transactional() {
            return transactionalId != null && !transactionalId.isBlank();
        }

        public int requestTimeoutMs() {
            return requestTimeoutMs;
        }

        public int deliveryTimeoutMs() {
            return deliveryTimeoutMs;
        }

        public Builder toBuilder() {
            return builder()
                    .acks(acks)
                    .idempotent(idempotent)
                    .compression(compression)
                    .lingerMs(lingerMs)
                    .batchSize(batchSize)
                    .bufferMemory(bufferMemory)
                    .retries(retries)
                    .maxInFlight(maxInFlight)
                    .transactionalId(transactionalId)
                    .requestTimeoutMs(requestTimeoutMs)
                    .deliveryTimeoutMs(deliveryTimeoutMs)
                    .extra(extra.isEmpty() ? null : new LinkedHashMap<>(extra));
        }

        void apply(Properties p, String clientId) {
            p.put(ProducerConfig.ACKS_CONFIG, acks.value());
            p.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, Boolean.toString(idempotent));
            p.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, compression.value());
            p.put(ProducerConfig.LINGER_MS_CONFIG, Integer.toString(lingerMs));
            p.put(ProducerConfig.BATCH_SIZE_CONFIG, Integer.toString(batchSize));
            p.put(ProducerConfig.BUFFER_MEMORY_CONFIG, Long.toString(bufferMemory));
            p.put(ProducerConfig.RETRIES_CONFIG, Integer.toString(retries));
            p.put(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, Integer.toString(maxInFlight));
            p.put(ProducerConfig.REQUEST_TIMEOUT_MS_CONFIG, Integer.toString(requestTimeoutMs));
            p.put(ProducerConfig.DELIVERY_TIMEOUT_MS_CONFIG, Integer.toString(deliveryTimeoutMs));
            if (transactionalId != null && !transactionalId.isBlank()) {
                p.put(ProducerConfig.TRANSACTIONAL_ID_CONFIG, transactionalId);
            }
            if (clientId != null && !clientId.isBlank()) {
                p.putIfAbsent(ProducerConfig.CLIENT_ID_CONFIG, clientId + "-producer");
            }
            for (Map.Entry<String, String> e : extra.entrySet()) {
                p.put(e.getKey(), e.getValue());
            }
        }

        public static final class Builder {
            private Acks acks = Acks.ALL;
            private boolean idempotent = true;
            private Compression compression = Compression.LZ4;
            private int lingerMs = 10;
            private int batchSize = 262144;
            private long bufferMemory = 64L << 20;
            private int retries = Integer.MAX_VALUE;
            private int maxInFlight = 5;
            private String transactionalId;
            private int requestTimeoutMs = 30_000;
            private int deliveryTimeoutMs = 120_000;
            private Map<String, String> extra;

            public Builder acks(Acks acks) {
                this.acks = acks;
                return this;
            }

            public Builder idempotent(boolean idempotent) {
                this.idempotent = idempotent;
                return this;
            }

            public Builder compression(Compression compression) {
                this.compression = compression;
                return this;
            }

            public Builder lingerMs(int lingerMs) {
                this.lingerMs = lingerMs;
                return this;
            }

            public Builder batchSize(int batchSize) {
                this.batchSize = batchSize;
                return this;
            }

            public Builder bufferMemory(long bufferMemory) {
                this.bufferMemory = bufferMemory;
                return this;
            }

            public Builder retries(int retries) {
                this.retries = retries;
                return this;
            }

            public Builder maxInFlight(int maxInFlight) {
                this.maxInFlight = maxInFlight;
                return this;
            }

            public Builder transactionalId(String transactionalId) {
                this.transactionalId = transactionalId;
                return this;
            }

            public Builder requestTimeoutMs(int requestTimeoutMs) {
                this.requestTimeoutMs = requestTimeoutMs;
                return this;
            }

            public Builder deliveryTimeoutMs(int deliveryTimeoutMs) {
                this.deliveryTimeoutMs = deliveryTimeoutMs;
                return this;
            }

            public Builder extra(Map<String, String> extra) {
                this.extra = extra;
                return this;
            }

            public ProducerOpts build() {
                // idempotence requires acks=all and max.in.flight <= 5
                if (idempotent) {
                    if (acks != Acks.ALL) acks = Acks.ALL;
                    if (maxInFlight > 5) maxInFlight = 5;
                }
                return new ProducerOpts(this);
            }
        }
    }

    // ── nested: ConsumerOpts ─────────────────────────────────────────────────

    /**
     * Consumer knobs. Defaults: manual commit, earliest reset, max.poll.records=2000,
     * read_committed, cooperative-sticky assignor, fetch.min.bytes=1, fetch.max.wait=500ms.
     */
    public static final class ConsumerOpts {
        private final String groupId;
        private final boolean enableAutoCommit;
        private final AutoOffsetReset autoOffsetReset;
        private final int maxPollRecords;
        private final int maxPollIntervalMs;
        private final Isolation isolation;
        private final boolean cooperativeSticky;
        private final int fetchMinBytes;
        private final int fetchMaxWaitMs;
        private final int sessionTimeoutMs;
        private final int heartbeatIntervalMs;
        private final Map<String, String> extra;

        private ConsumerOpts(Builder b) {
            this.groupId = b.groupId;
            this.enableAutoCommit = b.enableAutoCommit;
            this.autoOffsetReset = b.autoOffsetReset == null ? AutoOffsetReset.EARLIEST : b.autoOffsetReset;
            this.maxPollRecords = Math.max(1, b.maxPollRecords);
            this.maxPollIntervalMs = Math.max(1000, b.maxPollIntervalMs);
            this.isolation = b.isolation == null ? Isolation.READ_COMMITTED : b.isolation;
            this.cooperativeSticky = b.cooperativeSticky;
            this.fetchMinBytes = Math.max(1, b.fetchMinBytes);
            this.fetchMaxWaitMs = Math.max(0, b.fetchMaxWaitMs);
            this.sessionTimeoutMs = Math.max(1000, b.sessionTimeoutMs);
            this.heartbeatIntervalMs = Math.max(100, b.heartbeatIntervalMs);
            this.extra = b.extra == null || b.extra.isEmpty()
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
        }

        public static Builder builder() {
            return new Builder();
        }

        public static ConsumerOpts defaults() {
            return builder().build();
        }

        public String groupId() {
            return groupId;
        }

        public boolean enableAutoCommit() {
            return enableAutoCommit;
        }

        public AutoOffsetReset autoOffsetReset() {
            return autoOffsetReset;
        }

        public int maxPollRecords() {
            return maxPollRecords;
        }

        public Isolation isolation() {
            return isolation;
        }

        public boolean cooperativeSticky() {
            return cooperativeSticky;
        }

        public Builder toBuilder() {
            return builder()
                    .groupId(groupId)
                    .enableAutoCommit(enableAutoCommit)
                    .autoOffsetReset(autoOffsetReset)
                    .maxPollRecords(maxPollRecords)
                    .maxPollIntervalMs(maxPollIntervalMs)
                    .isolation(isolation)
                    .cooperativeSticky(cooperativeSticky)
                    .fetchMinBytes(fetchMinBytes)
                    .fetchMaxWaitMs(fetchMaxWaitMs)
                    .sessionTimeoutMs(sessionTimeoutMs)
                    .heartbeatIntervalMs(heartbeatIntervalMs)
                    .extra(extra.isEmpty() ? null : new LinkedHashMap<>(extra));
        }

        void apply(Properties p, String clientId) {
            if (groupId != null && !groupId.isBlank()) {
                p.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
            }
            p.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, Boolean.toString(enableAutoCommit));
            p.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, autoOffsetReset.value());
            p.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, Integer.toString(maxPollRecords));
            p.put(ConsumerConfig.MAX_POLL_INTERVAL_MS_CONFIG, Integer.toString(maxPollIntervalMs));
            p.put(ConsumerConfig.ISOLATION_LEVEL_CONFIG, isolation.value());
            p.put(ConsumerConfig.FETCH_MIN_BYTES_CONFIG, Integer.toString(fetchMinBytes));
            p.put(ConsumerConfig.FETCH_MAX_WAIT_MS_CONFIG, Integer.toString(fetchMaxWaitMs));
            p.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, Integer.toString(sessionTimeoutMs));
            p.put(ConsumerConfig.HEARTBEAT_INTERVAL_MS_CONFIG, Integer.toString(heartbeatIntervalMs));
            if (cooperativeSticky) {
                p.put(ConsumerConfig.PARTITION_ASSIGNMENT_STRATEGY_CONFIG,
                        "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
            }
            if (clientId != null && !clientId.isBlank()) {
                p.putIfAbsent(ConsumerConfig.CLIENT_ID_CONFIG, clientId + "-consumer");
            }
            for (Map.Entry<String, String> e : extra.entrySet()) {
                p.put(e.getKey(), e.getValue());
            }
        }

        public static final class Builder {
            private String groupId = "jnitorch-default";
            private boolean enableAutoCommit = false;
            private AutoOffsetReset autoOffsetReset = AutoOffsetReset.EARLIEST;
            private int maxPollRecords = 2000;
            private int maxPollIntervalMs = 300_000;
            private Isolation isolation = Isolation.READ_COMMITTED;
            private boolean cooperativeSticky = true;
            private int fetchMinBytes = 1;
            private int fetchMaxWaitMs = 500;
            private int sessionTimeoutMs = 45_000;
            private int heartbeatIntervalMs = 3000;
            private Map<String, String> extra;

            public Builder groupId(String groupId) {
                this.groupId = groupId;
                return this;
            }

            public Builder enableAutoCommit(boolean enableAutoCommit) {
                this.enableAutoCommit = enableAutoCommit;
                return this;
            }

            public Builder autoOffsetReset(AutoOffsetReset autoOffsetReset) {
                this.autoOffsetReset = autoOffsetReset;
                return this;
            }

            public Builder maxPollRecords(int maxPollRecords) {
                this.maxPollRecords = maxPollRecords;
                return this;
            }

            public Builder maxPollIntervalMs(int maxPollIntervalMs) {
                this.maxPollIntervalMs = maxPollIntervalMs;
                return this;
            }

            public Builder isolation(Isolation isolation) {
                this.isolation = isolation;
                return this;
            }

            public Builder cooperativeSticky(boolean cooperativeSticky) {
                this.cooperativeSticky = cooperativeSticky;
                return this;
            }

            public Builder fetchMinBytes(int fetchMinBytes) {
                this.fetchMinBytes = fetchMinBytes;
                return this;
            }

            public Builder fetchMaxWaitMs(int fetchMaxWaitMs) {
                this.fetchMaxWaitMs = fetchMaxWaitMs;
                return this;
            }

            public Builder sessionTimeoutMs(int sessionTimeoutMs) {
                this.sessionTimeoutMs = sessionTimeoutMs;
                return this;
            }

            public Builder heartbeatIntervalMs(int heartbeatIntervalMs) {
                this.heartbeatIntervalMs = heartbeatIntervalMs;
                return this;
            }

            public Builder extra(Map<String, String> extra) {
                this.extra = extra;
                return this;
            }

            public ConsumerOpts build() {
                return new ConsumerOpts(this);
            }
        }
    }

    // ── nested: TopicOpts ────────────────────────────────────────────────────

    /**
     * Topic declaration: name, partitions, replication factor, retention, min.ISR.
     * Default RF=3 with min.insync.replicas=2 when RF≥3 (production durability).
     */
    public static final class TopicOpts {
        private final String name;
        private final int partitions;
        private final short replicationFactor;
        private final int minInSyncReplicas;
        private final long retentionMs;
        private final Map<String, String> configs;

        private TopicOpts(Builder b) {
            this.name = Objects.requireNonNull(b.name, "topic name").trim();
            if (this.name.isEmpty()) throw new IllegalArgumentException("topic name blank");
            this.partitions = Math.max(1, b.partitions);
            this.replicationFactor = b.replicationFactor <= 0 ? 1 : b.replicationFactor;
            int defaultMisr = this.replicationFactor >= 3 ? 2 : 1;
            this.minInSyncReplicas = b.minInSyncReplicas <= 0 ? defaultMisr : b.minInSyncReplicas;
            this.retentionMs = b.retentionMs;
            Map<String, String> cfg = new LinkedHashMap<>();
            if (b.configs != null) cfg.putAll(b.configs);
            cfg.putIfAbsent("min.insync.replicas", Integer.toString(this.minInSyncReplicas));
            if (this.retentionMs > 0) {
                cfg.putIfAbsent("retention.ms", Long.toString(this.retentionMs));
            }
            this.configs = Collections.unmodifiableMap(cfg);
        }

        public static Builder builder() {
            return new Builder();
        }

        public static TopicOpts of(String name) {
            return builder().name(name).build();
        }

        public static TopicOpts of(String name, int partitions, short replicationFactor) {
            return builder().name(name).partitions(partitions).replicationFactor(replicationFactor).build();
        }

        public String name() {
            return name;
        }

        public int partitions() {
            return partitions;
        }

        public short replicationFactor() {
            return replicationFactor;
        }

        public int minInSyncReplicas() {
            return minInSyncReplicas;
        }

        public long retentionMs() {
            return retentionMs;
        }

        public Map<String, String> configs() {
            return configs;
        }

        public static final class Builder {
            private String name;
            private int partitions = 1;
            private short replicationFactor = 1;
            private int minInSyncReplicas = -1;
            private long retentionMs = 7L * 24 * 3600 * 1000; // 7d
            private Map<String, String> configs;

            public Builder name(String name) {
                this.name = name;
                return this;
            }

            public Builder partitions(int partitions) {
                this.partitions = partitions;
                return this;
            }

            public Builder replicationFactor(short replicationFactor) {
                this.replicationFactor = replicationFactor;
                return this;
            }

            public Builder replicationFactor(int replicationFactor) {
                this.replicationFactor = (short) replicationFactor;
                return this;
            }

            public Builder minInSyncReplicas(int minInSyncReplicas) {
                this.minInSyncReplicas = minInSyncReplicas;
                return this;
            }

            public Builder retentionMs(long retentionMs) {
                this.retentionMs = retentionMs;
                return this;
            }

            public Builder configs(Map<String, String> configs) {
                this.configs = configs;
                return this;
            }

            public Builder config(String key, String value) {
                if (this.configs == null) this.configs = new LinkedHashMap<>();
                this.configs.put(key, value);
                return this;
            }

            public TopicOpts build() {
                return new TopicOpts(this);
            }
        }
    }

    // ── nested: SecurityOpts ─────────────────────────────────────────────────

    public static final class SecurityOpts {
        private final SecurityProtocol protocol;
        private final String saslMechanism;
        private final String saslJaas;
        private final String sslTruststoreLocation;
        private final String sslTruststorePassword;
        private final String sslKeystoreLocation;
        private final String sslKeystorePassword;
        private final String sslKeyPassword;
        private final Map<String, String> extra;

        private SecurityOpts(Builder b) {
            this.protocol = b.protocol == null ? SecurityProtocol.PLAINTEXT : b.protocol;
            this.saslMechanism = b.saslMechanism;
            this.saslJaas = b.saslJaas;
            this.sslTruststoreLocation = b.sslTruststoreLocation;
            this.sslTruststorePassword = b.sslTruststorePassword;
            this.sslKeystoreLocation = b.sslKeystoreLocation;
            this.sslKeystorePassword = b.sslKeystorePassword;
            this.sslKeyPassword = b.sslKeyPassword;
            this.extra = b.extra == null || b.extra.isEmpty()
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
        }

        public static Builder builder() {
            return new Builder();
        }

        public static SecurityOpts plaintext() {
            return builder().protocol(SecurityProtocol.PLAINTEXT).build();
        }

        public static SecurityOpts ssl(String truststoreLocation, String truststorePassword) {
            return builder()
                    .protocol(SecurityProtocol.SSL)
                    .sslTruststoreLocation(truststoreLocation)
                    .sslTruststorePassword(truststorePassword)
                    .build();
        }

        public static SecurityOpts saslPlain(String username, String password) {
            String jaas = "org.apache.kafka.common.security.plain.PlainLoginModule required "
                    + "username=\"" + username + "\" password=\"" + password + "\";";
            return builder()
                    .protocol(SecurityProtocol.SASL_PLAINTEXT)
                    .saslMechanism("PLAIN")
                    .saslJaas(jaas)
                    .build();
        }

        public static SecurityOpts saslSsl(String username, String password,
                                           String truststoreLocation, String truststorePassword) {
            String jaas = "org.apache.kafka.common.security.plain.PlainLoginModule required "
                    + "username=\"" + username + "\" password=\"" + password + "\";";
            return builder()
                    .protocol(SecurityProtocol.SASL_SSL)
                    .saslMechanism("PLAIN")
                    .saslJaas(jaas)
                    .sslTruststoreLocation(truststoreLocation)
                    .sslTruststorePassword(truststorePassword)
                    .build();
        }

        public static SecurityOpts saslScramSha256(String username, String password) {
            String jaas = "org.apache.kafka.common.security.scram.ScramLoginModule required "
                    + "username=\"" + username + "\" password=\"" + password + "\";";
            return builder()
                    .protocol(SecurityProtocol.SASL_PLAINTEXT)
                    .saslMechanism("SCRAM-SHA-256")
                    .saslJaas(jaas)
                    .build();
        }

        void apply(Properties p) {
            p.put(CommonClientConfigs.SECURITY_PROTOCOL_CONFIG, protocol.name());
            if (saslMechanism != null) p.put(SaslConfigs.SASL_MECHANISM, saslMechanism);
            if (saslJaas != null) p.put(SaslConfigs.SASL_JAAS_CONFIG, saslJaas);
            if (sslTruststoreLocation != null) {
                p.put(SslConfigs.SSL_TRUSTSTORE_LOCATION_CONFIG, sslTruststoreLocation);
            }
            if (sslTruststorePassword != null) {
                p.put(SslConfigs.SSL_TRUSTSTORE_PASSWORD_CONFIG, sslTruststorePassword);
            }
            if (sslKeystoreLocation != null) {
                p.put(SslConfigs.SSL_KEYSTORE_LOCATION_CONFIG, sslKeystoreLocation);
            }
            if (sslKeystorePassword != null) {
                p.put(SslConfigs.SSL_KEYSTORE_PASSWORD_CONFIG, sslKeystorePassword);
            }
            if (sslKeyPassword != null) {
                p.put(SslConfigs.SSL_KEY_PASSWORD_CONFIG, sslKeyPassword);
            }
            for (Map.Entry<String, String> e : extra.entrySet()) {
                p.put(e.getKey(), e.getValue());
            }
        }

        public static final class Builder {
            private SecurityProtocol protocol = SecurityProtocol.PLAINTEXT;
            private String saslMechanism;
            private String saslJaas;
            private String sslTruststoreLocation;
            private String sslTruststorePassword;
            private String sslKeystoreLocation;
            private String sslKeystorePassword;
            private String sslKeyPassword;
            private Map<String, String> extra;

            public Builder protocol(SecurityProtocol protocol) {
                this.protocol = protocol;
                return this;
            }

            public Builder saslMechanism(String saslMechanism) {
                this.saslMechanism = saslMechanism;
                return this;
            }

            public Builder saslJaas(String saslJaas) {
                this.saslJaas = saslJaas;
                return this;
            }

            public Builder sslTruststoreLocation(String sslTruststoreLocation) {
                this.sslTruststoreLocation = sslTruststoreLocation;
                return this;
            }

            public Builder sslTruststorePassword(String sslTruststorePassword) {
                this.sslTruststorePassword = sslTruststorePassword;
                return this;
            }

            public Builder sslKeystoreLocation(String sslKeystoreLocation) {
                this.sslKeystoreLocation = sslKeystoreLocation;
                return this;
            }

            public Builder sslKeystorePassword(String sslKeystorePassword) {
                this.sslKeystorePassword = sslKeystorePassword;
                return this;
            }

            public Builder sslKeyPassword(String sslKeyPassword) {
                this.sslKeyPassword = sslKeyPassword;
                return this;
            }

            public Builder extra(Map<String, String> extra) {
                this.extra = extra;
                return this;
            }

            public SecurityOpts build() {
                return new SecurityOpts(this);
            }
        }
    }

    /** Parse {@code kafka://host:9092,host2:9092?group=g&topic=t&client.id=c}. */
    public static KafkaOptions fromUri(String uri) {
        Objects.requireNonNull(uri, "uri");
        String s = uri.trim();
        if (s.startsWith("kafka://")) s = s.substring("kafka://".length());
        else if (s.startsWith("kafkas://")) s = s.substring("kafkas://".length());
        String servers;
        String query = null;
        int q = s.indexOf('?');
        if (q >= 0) {
            servers = s.substring(0, q);
            query = s.substring(q + 1);
        } else {
            servers = s;
        }
        Builder b = builder().bootstrapServers(servers);
        if (uri.trim().toLowerCase(Locale.ROOT).startsWith("kafkas://")) {
            b.security(SecurityOpts.builder().protocol(SecurityProtocol.SSL).build());
        }
        if (query != null && !query.isBlank()) {
            Map<String, String> params = new HashMap<>();
            for (String part : query.split("&")) {
                int eq = part.indexOf('=');
                if (eq > 0) {
                    params.put(part.substring(0, eq), part.substring(eq + 1));
                }
            }
            if (params.containsKey("group")) b.groupId(params.get("group"));
            if (params.containsKey("group.id")) b.groupId(params.get("group.id"));
            if (params.containsKey("topic")) b.topic(params.get("topic"));
            if (params.containsKey("client.id")) b.clientId(params.get("client.id"));
            if (params.containsKey("key.column")) b.keyColumn(params.get("key.column"));
        }
        return b.build();
    }
}
