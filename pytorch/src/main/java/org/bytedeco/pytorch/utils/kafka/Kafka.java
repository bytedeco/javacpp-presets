package org.bytedeco.pytorch.utils.kafka;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.Closeable;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Enterprise Kafka façade for DataFrame streaming I/O, topic admin, offline dump
 * replay, and online-training feature pipelines.
 *
 * <p>Built on official {@code org.apache.kafka:kafka-clients}. Defaults follow
 * high-QPS recsys practice: idempotent producer ({@code acks=all}, LZ4), manual
 * consumer commit, cooperative-sticky assignment, {@code read_committed}.
 *
 * <pre>{@code
 * try (Kafka k = Kafka.connect("kafka-1:9092,kafka-2:9092")) {
 *     k.createTopic("rec.feature.log", 64, (short) 3);
 *
 *     df.toKafka(k, KafkaOptions.builder()
 *         .topic("rec.feature.log")
 *         .keyColumn("user_id")
 *         .build());
 *
 *     k.streamDataFrame(
 *         KafkaOptions.consumer("kafka-1:9092", "rec.feature.log", "fe-v1"),
 *         2048,
 *         batch -> {
 *             Tensor x = batch.toTensor("f1", "f2", "f3");
 *             // online train / rank …
 *         });
 *
 *     // offline dump ↔ live
 *     KafkaFile.writeJsonl(df, Path.of("dump.jsonl"));
 *     k.produceFile(Path.of("dump.jsonl"),
 *         KafkaOptions.producer("kafka-1:9092", "rec.feature.log"));
 * }
 * }</pre>
 *
 * @see KafkaProducer
 * @see KafkaConsumer
 * @see KafkaAdmin
 * @see KafkaStream
 * @see KafkaFile
 * @see KafkaFeatureBridge
 */
public final class Kafka implements Closeable {

    public static final String VERSION = "4.3.1";

    private final KafkaOptions options;
    private final KafkaMetrics metrics;
    private KafkaProducer producer;
    private KafkaConsumer consumer;
    private KafkaAdmin admin;
    private boolean closed;

    private Kafka(KafkaOptions options) {
        this.options = options == null ? KafkaOptions.defaults() : options;
        this.metrics = new KafkaMetrics();
    }

    // ── factories ────────────────────────────────────────────────────────────

    public static Kafka connect() {
        return connect("localhost:9092");
    }

    public static Kafka connect(String bootstrapServers) {
        return connect(KafkaOptions.builder().bootstrapServers(bootstrapServers).build());
    }

    public static Kafka connect(KafkaOptions options) {
        return new Kafka(options);
    }

    /** Parse {@code kafka://host:9092?group=g&topic=t}. */
    public static Kafka connectUri(String uri) {
        return connect(KafkaOptions.fromUri(uri));
    }

    public static KafkaStream stream(KafkaOptions options) {
        return KafkaStream.open(options);
    }

    public static KafkaStream stream(String bootstrap, String topic, String groupId) {
        return stream(KafkaOptions.consumer(bootstrap, topic, groupId));
    }

    // ── accessors ────────────────────────────────────────────────────────────

    public KafkaOptions options() {
        return options;
    }

    public KafkaMetrics metrics() {
        return metrics;
    }

    public String bootstrapServers() {
        return options.bootstrapServers();
    }

    // ── lazy clients ─────────────────────────────────────────────────────────

    public synchronized KafkaProducer producer() {
        ensureOpen();
        if (producer == null) {
            producer = KafkaProducer.connect(options);
        }
        return producer;
    }

    public synchronized KafkaProducer producer(KafkaOptions opts) {
        ensureOpen();
        // one-shot producer with different options (caller owns close if different)
        if (opts == null || sameBootstrap(opts)) {
            return producer();
        }
        return KafkaProducer.connect(opts);
    }

    public synchronized KafkaConsumer consumer() {
        ensureOpen();
        if (consumer == null) {
            consumer = KafkaConsumer.connect(options);
            if (options.topicName() != null) {
                consumer.subscribe(options.topicName());
            }
        }
        return consumer;
    }

    public synchronized KafkaConsumer consumer(KafkaOptions opts) {
        ensureOpen();
        if (opts == null || sameBootstrap(opts)) {
            KafkaConsumer c = consumer();
            if (opts != null && opts.topicName() != null
                    && (c.subscription().isEmpty() && c.assignment().isEmpty())) {
                c.subscribe(opts.topicName());
            }
            return c;
        }
        KafkaConsumer c = KafkaConsumer.connect(opts);
        if (opts.topicName() != null) c.subscribe(opts.topicName());
        return c;
    }

    public synchronized KafkaAdmin admin() {
        ensureOpen();
        if (admin == null) {
            admin = KafkaAdmin.connect(options);
        }
        return admin;
    }

    // ── admin shortcuts ──────────────────────────────────────────────────────

    public void createTopic(String name, int partitions, short replicationFactor) {
        admin().createTopic(name, partitions, replicationFactor);
    }

    public void createTopic(KafkaOptions.TopicOpts topic) {
        admin().createTopic(topic);
    }

    public void deleteTopic(String name) {
        admin().deleteTopic(name);
    }

    public boolean topicExists(String name) {
        return admin().topicExists(name);
    }

    public Set<String> listTopics() {
        return admin().listTopics();
    }

    public KafkaAdmin.TopicInfo describeTopic(String name) {
        return admin().describeTopic(name);
    }

    public void alterTopicConfig(String topic, Map<String, String> configs) {
        admin().alterTopicConfig(topic, configs);
    }

    public void createPartitions(String topic, int totalPartitions) {
        admin().createPartitions(topic, totalPartitions);
    }

    // ── DataFrame write ──────────────────────────────────────────────────────

    /**
     * Write DataFrame rows to Kafka. Returns acknowledged record count.
     */
    public int writeDataFrame(DataFrame df, KafkaOptions opts) {
        Objects.requireNonNull(df, "df");
        KafkaOptions o = merge(opts);
        if (o.topicName() == null || o.topicName().isBlank()) {
            throw new KafkaException("topic required for writeDataFrame", null, "writeDataFrame", null);
        }
        // optionally ensure topic exists
        if (o.topic() != null && o.topic().partitions() > 0 && o.ifNotExists()) {
            try {
                admin().createTopic(o.topic(), true);
            } catch (KafkaException ignored) {
                // best-effort; produce may still succeed if topic exists with different config
            }
        }
        KafkaProducer p = producer(o);
        try {
            int n = p.sendDataFrame(df, o);
            // merge producer metrics into façade
            KafkaMetrics.Snapshot s = p.metrics().snapshot();
            metrics.recordProduce((int) s.produced(), s.produceBytes(), s.avgProduceLatencyMs(), true);
            return n;
        } finally {
            if (p != producer) {
                p.close();
            }
        }
    }

    public int writeDataFrame(DataFrame df) {
        return writeDataFrame(df, options);
    }

    public int writeDataFrame(DataFrame df, String topic, String keyColumn) {
        return writeDataFrame(df, options.toBuilder()
                .topic(topic)
                .keyColumn(keyColumn)
                .build());
    }

    // ── DataFrame read ───────────────────────────────────────────────────────

    /**
     * One poll cycle → DataFrame (may be empty).
     */
    public DataFrame readDataFrame(KafkaOptions opts) {
        KafkaOptions o = merge(opts);
        KafkaConsumer c = consumer(o);
        try {
            DataFrame df = c.pollDataFrame(o.pollTimeout(), o);
            metrics.recordConsume(df.rowCount(), 0, true);
            metrics.setLag(KafkaOffsets.lag(c.raw(), null));
            return df;
        } finally {
            if (c != consumer) c.close();
        }
    }

    public DataFrame readDataFrame() {
        return readDataFrame(options);
    }

    /**
     * Accumulate up to {@code maxRecords} then return a DataFrame.
     */
    public DataFrame readDataFrame(KafkaOptions opts, int maxRecords) {
        KafkaOptions o = merge(opts).toBuilder().maxRecords(maxRecords).build();
        KafkaConsumer c = consumer(o);
        try {
            DataFrame df = c.pollDataFrameUntil(maxRecords, o.pollTimeout(), Duration.ofSeconds(5), o);
            metrics.recordConsume(df.rowCount(), 0, true);
            return df;
        } finally {
            if (c != consumer) c.close();
        }
    }

    // ── stream ───────────────────────────────────────────────────────────────

    /**
     * Continuous consume → handler. Returns total rows delivered.
     * Commits after each batch by default (manual commit client).
     */
    public long streamDataFrame(KafkaOptions opts, int batchRows, Consumer<DataFrame> handler) {
        Objects.requireNonNull(handler, "handler");
        KafkaOptions o = merge(opts);
        try (KafkaStream stream = KafkaStream.open(o)) {
            stream.batchRows(batchRows <= 0 ? 2048 : batchRows);
            long n = stream.forEachBatch(handler);
            KafkaMetrics.Snapshot s = stream.metrics().snapshot();
            metrics.recordConsume((int) Math.min(Integer.MAX_VALUE, s.consumed()), s.consumeBytes(), true);
            metrics.setLag(s.lag());
            return n;
        }
    }

    public long streamDataFrame(int batchRows, Consumer<DataFrame> handler) {
        return streamDataFrame(options, batchRows, handler);
    }

    public long streamDataFrame(Consumer<DataFrame> handler) {
        return streamDataFrame(options, 2048, handler);
    }

    // ── file bridge ──────────────────────────────────────────────────────────

    /**
     * Replay an offline dump file onto a live topic.
     *
     * @return records produced
     */
    public int produceFile(Path path, KafkaOptions opts) {
        Objects.requireNonNull(path, "path");
        KafkaOptions o = merge(opts);
        DataFrame df = KafkaFile.read(path);
        return writeDataFrame(df, o);
    }

    public int produceFile(Path path) {
        return produceFile(path, options);
    }

    /**
     * Consume live topic into an offline dump file.
     *
     * @return records written
     */
    public long consumeToFile(KafkaOptions opts, Path path, int maxRecords) {
        Objects.requireNonNull(path, "path");
        KafkaOptions o = merge(opts);
        DataFrame df = readDataFrame(o, maxRecords <= 0 ? o.consumer().maxPollRecords() : maxRecords);
        KafkaFile.write(df, path);
        // commit if we used the shared consumer
        try {
            if (consumer != null) consumer.commitSync();
        } catch (Exception ignored) {
        }
        return df.rowCount();
    }

    public long consumeToFile(Path path, int maxRecords) {
        return consumeToFile(options, path, maxRecords);
    }

    // ── low-level send / poll ────────────────────────────────────────────────

    public void send(String topic, String key, Object value) {
        producer().send(topic, key, value);
    }

    public void send(KafkaRecord record) {
        producer().send(record);
    }

    public List<KafkaRecord> poll(Duration timeout) {
        return consumer().poll(timeout);
    }

    public void flush() {
        if (producer != null) producer.flush();
    }

    public void commit() {
        if (consumer != null) consumer.commitSync();
    }

    public long lag() {
        if (consumer == null) return -1L;
        long lag = KafkaOffsets.lag(consumer.raw(), null);
        metrics.setLag(lag);
        return lag;
    }

    // ── internals ────────────────────────────────────────────────────────────

    private KafkaOptions merge(KafkaOptions opts) {
        if (opts == null) return options;
        // prefer explicit opts; fill bootstrap from façade if missing default-ish
        if (opts.bootstrapServers() == null || opts.bootstrapServers().isBlank()
                || "localhost:9092".equals(opts.bootstrapServers())
                && options.bootstrapServers() != null
                && !"localhost:9092".equals(options.bootstrapServers())) {
            return opts.toBuilder().bootstrapServers(options.bootstrapServers()).build();
        }
        return opts;
    }

    private boolean sameBootstrap(KafkaOptions opts) {
        return opts != null
                && Objects.equals(opts.bootstrapServers(), options.bootstrapServers());
    }

    private void ensureOpen() {
        if (closed) throw new KafkaException("Kafka client closed", null, "ensureOpen", null);
    }

    @Override
    public synchronized void close() {
        closed = true;
        if (producer != null) {
            try {
                producer.close();
            } catch (Exception ignored) {
            }
            producer = null;
        }
        if (consumer != null) {
            try {
                consumer.close();
            } catch (Exception ignored) {
            }
            consumer = null;
        }
        if (admin != null) {
            try {
                admin.close();
            } catch (Exception ignored) {
            }
            admin = null;
        }
    }
}
