package org.bytedeco.pytorch.utils.kafka;

import org.apache.kafka.clients.consumer.OffsetAndMetadata;
import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.header.internals.RecordHeader;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;

/**
 * High-throughput Kafka producer wrapper.
 *
 * <p>Thread-safe (upstream guarantee). Enterprise defaults from {@link KafkaOptions.ProducerOpts}:
 * {@code acks=all}, idempotent, LZ4, linger/batch tuned for high QPS feature logs.
 *
 * <pre>{@code
 * try (KafkaProducer p = KafkaProducer.connect(opts)) {
 *     p.send("rec.feature.log", "u42", Map.of("item_id", 7, "score", 0.91));
 *     p.sendDataFrame(df, opts);   // keyColumn = user_id
 *     p.flush();
 * }
 * }</pre>
 */
public final class KafkaProducer implements Closeable {

    private final Producer<byte[], byte[]> producer;
    private final boolean ownProducer;
    private final KafkaOptions options;
    private final KafkaMetrics metrics;
    private final AtomicLong inflight = new AtomicLong();

    private KafkaProducer(Producer<byte[], byte[]> producer, boolean ownProducer,
                          KafkaOptions options, KafkaMetrics metrics) {
        this.producer = Objects.requireNonNull(producer, "producer");
        this.ownProducer = ownProducer;
        this.options = options == null ? KafkaOptions.defaults() : options;
        this.metrics = metrics == null ? new KafkaMetrics() : metrics;
        if (this.options.producer().transactional()) {
            try {
                this.producer.initTransactions();
            } catch (Exception e) {
                throw new KafkaException("initTransactions failed: " + e.getMessage(), e, "initTransactions", null);
            }
        }
    }

    public static KafkaProducer connect(String bootstrapServers) {
        return connect(KafkaOptions.builder().bootstrapServers(bootstrapServers).build());
    }

    public static KafkaProducer connect(KafkaOptions options) {
        Objects.requireNonNull(options, "options");
        org.apache.kafka.clients.producer.KafkaProducer<byte[], byte[]> p =
                new org.apache.kafka.clients.producer.KafkaProducer<>(options.producerProperties());
        return new KafkaProducer(p, true, options, new KafkaMetrics());
    }

    public static KafkaProducer wrap(Producer<byte[], byte[]> producer, KafkaOptions options) {
        return new KafkaProducer(producer, false, options, new KafkaMetrics());
    }

    public Producer<byte[], byte[]> raw() {
        return producer;
    }

    public KafkaOptions options() {
        return options;
    }

    public KafkaMetrics metrics() {
        return metrics;
    }

    // ── transactions ─────────────────────────────────────────────────────────

    public void beginTransaction() {
        try {
            producer.beginTransaction();
        } catch (Exception e) {
            throw new KafkaException("beginTransaction failed: " + e.getMessage(), e, "beginTransaction", null);
        }
    }

    public void commitTransaction() {
        try {
            producer.commitTransaction();
        } catch (Exception e) {
            throw new KafkaException("commitTransaction failed: " + e.getMessage(), e, "commitTransaction", null);
        }
    }

    public void abortTransaction() {
        try {
            producer.abortTransaction();
        } catch (Exception e) {
            throw new KafkaException("abortTransaction failed: " + e.getMessage(), e, "abortTransaction", null);
        }
    }

    /**
     * EOS: send offsets to transaction using a live consumer's
     * {@link org.apache.kafka.clients.consumer.Consumer#groupMetadata()} (preferred).
     */
    public void sendOffsetsToTransaction(Map<TopicPartition, OffsetAndMetadata> offsets,
                                         KafkaConsumer consumer) {
        Objects.requireNonNull(consumer, "consumer");
        sendOffsetsToTransaction(offsets, consumer.raw().groupMetadata());
    }

    /** EOS with explicit consumer group metadata. */
    public void sendOffsetsToTransaction(Map<TopicPartition, OffsetAndMetadata> offsets,
                                         org.apache.kafka.clients.consumer.ConsumerGroupMetadata groupMetadata) {
        try {
            producer.sendOffsetsToTransaction(offsets, groupMetadata);
        } catch (Exception e) {
            throw new KafkaException("sendOffsetsToTransaction failed: " + e.getMessage(),
                    e, "sendOffsetsToTransaction", null);
        }
    }

    /**
     * EOS convenience when only a group id is known (no live consumer).
     * Prefer {@link #sendOffsetsToTransaction(Map, KafkaConsumer)} in production.
     */
    @SuppressWarnings("removal")
    public void sendOffsetsToTransaction(Map<TopicPartition, OffsetAndMetadata> offsets, String groupId) {
        try {
            // kafka-clients 4.x marks direct construction for removal; still required for
            // offline / test paths without an assigned consumer member.
            org.apache.kafka.clients.consumer.ConsumerGroupMetadata meta =
                    new org.apache.kafka.clients.consumer.ConsumerGroupMetadata(groupId);
            producer.sendOffsetsToTransaction(offsets, meta);
        } catch (Exception e) {
            throw new KafkaException("sendOffsetsToTransaction failed: " + e.getMessage(),
                    e, "sendOffsetsToTransaction", null);
        }
    }

    // ── send ─────────────────────────────────────────────────────────────────

    public RecordMetadata send(String topic, String key, Object value) {
        return send(KafkaRecord.of(topic, key, value), options);
    }

    public RecordMetadata send(KafkaRecord record) {
        return send(record, options);
    }

    public RecordMetadata send(KafkaRecord record, KafkaOptions opts) {
        try {
            return sendAsync(record, opts).get(opts.pollTimeout().toMillis() + opts.producer().deliveryTimeoutMs(),
                    TimeUnit.MILLISECONDS);
        } catch (KafkaException e) {
            throw e;
        } catch (Exception e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new KafkaException("send failed: " + cause.getMessage(), cause, "send",
                    record == null ? null : record.topic());
        }
    }

    public CompletableFuture<RecordMetadata> sendAsync(String topic, String key, Object value) {
        return sendAsync(KafkaRecord.of(topic, key, value), options);
    }

    public CompletableFuture<RecordMetadata> sendAsync(KafkaRecord record) {
        return sendAsync(record, options);
    }

    public CompletableFuture<RecordMetadata> sendAsync(KafkaRecord record, KafkaOptions opts) {
        return sendAsync(record, opts, null);
    }

    public CompletableFuture<RecordMetadata> sendAsync(
            KafkaRecord record,
            KafkaOptions opts,
            BiConsumer<RecordMetadata, Exception> callback) {
        Objects.requireNonNull(record, "record");
        KafkaOptions o = opts == null ? options : opts;
        String topic = record.topic() != null ? record.topic() : o.topicName();
        if (topic == null || topic.isBlank()) {
            throw new KafkaException("topic required", null, "send", null);
        }

        byte[] keyBytes = KafkaSerde.encodeKey(record.key());
        byte[] valueBytes = KafkaSerde.encodeValue(record.value(), o.valueFormat());
        long bytes = (keyBytes == null ? 0 : keyBytes.length) + (valueBytes == null ? 0 : valueBytes.length);

        ProducerRecord<byte[], byte[]> pr;
        if (record.partition() != null) {
            Long ts = record.timestamp();
            pr = new ProducerRecord<>(topic, record.partition(),
                    ts == null ? null : ts, keyBytes, valueBytes);
        } else if (record.timestamp() != null) {
            pr = new ProducerRecord<>(topic, null, record.timestamp(), keyBytes, valueBytes);
        } else {
            pr = new ProducerRecord<>(topic, keyBytes, valueBytes);
        }

        Map<String, String> hdrs = mergeHeaders(o.headers(), record.headers());
        for (Map.Entry<String, String> h : hdrs.entrySet()) {
            if (h.getKey() == null) continue;
            byte[] hv = h.getValue() == null ? null : h.getValue().getBytes(StandardCharsets.UTF_8);
            pr.headers().add(new RecordHeader(h.getKey(), hv));
        }

        CompletableFuture<RecordMetadata> future = new CompletableFuture<>();
        long start = System.nanoTime();
        inflight.incrementAndGet();
        try {
            producer.send(pr, (metadata, exception) -> {
                inflight.decrementAndGet();
                double ms = (System.nanoTime() - start) / 1_000_000.0;
                if (exception != null) {
                    metrics.recordProduce(1, bytes, ms, false);
                    if (callback != null) {
                        try {
                            callback.accept(null, exception);
                        } catch (Exception ignored) {
                        }
                    }
                    future.completeExceptionally(exception);
                } else {
                    metrics.recordProduce(1, bytes, ms, true);
                    if (callback != null) {
                        try {
                            callback.accept(metadata, null);
                        } catch (Exception ignored) {
                        }
                    }
                    future.complete(metadata);
                }
            });
        } catch (Exception e) {
            inflight.decrementAndGet();
            metrics.recordProduce(1, bytes, -1, false);
            future.completeExceptionally(e);
        }
        return future;
    }

    /**
     * Produce every DataFrame row. Key from {@code opts.keyColumn()}; optional
     * partition from {@code opts.partitionColumn()}; optional event timestamp from
     * {@code opts.timestampColumn()}.
     *
     * @return number of rows successfully acknowledged (sync flush)
     */
    public int sendDataFrame(DataFrame df, KafkaOptions opts) {
        Objects.requireNonNull(df, "df");
        KafkaOptions o = opts == null ? options : opts;
        String topic = o.topicName();
        if (topic == null || topic.isBlank()) {
            throw new KafkaException("topic required for sendDataFrame", null, "sendDataFrame", null);
        }
        List<Map<String, Object>> records = df.toRecords();
        return sendRecords(topic, records, o);
    }

    public int sendRecords(String topic, List<Map<String, Object>> rows, KafkaOptions opts) {
        Objects.requireNonNull(topic, "topic");
        if (rows == null || rows.isEmpty()) return 0;
        KafkaOptions o = opts == null ? options : opts;
        String keyCol = o.keyColumn();
        String partCol = o.partitionColumn();
        String tsCol = o.timestampColumn();

        List<CompletableFuture<RecordMetadata>> futures = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) {
            if (row == null) continue;
            String key = null;
            if (keyCol != null && row.containsKey(keyCol) && row.get(keyCol) != null) {
                key = String.valueOf(row.get(keyCol));
            }
            Integer partition = null;
            if (partCol != null && row.get(partCol) instanceof Number n) {
                partition = n.intValue();
            }
            Long ts = null;
            if (tsCol != null && row.get(tsCol) instanceof Number n) {
                ts = n.longValue();
            }
            // strip metadata-only columns from value payload
            Map<String, Object> value = new LinkedHashMap<>(row);
            value.remove("__topic");
            value.remove("__partition");
            value.remove("__offset");
            value.remove("__timestamp");
            value.remove("__key");
            value.remove("__headers");

            KafkaRecord.Builder b = KafkaRecord.builder()
                    .topic(topic)
                    .key(key)
                    .value(value);
            if (partition != null) b.partition(partition);
            if (ts != null) b.timestamp(ts);
            futures.add(sendAsync(b.build(), o));
        }

        int ok = 0;
        for (CompletableFuture<RecordMetadata> f : futures) {
            try {
                f.get(o.producer().deliveryTimeoutMs(), TimeUnit.MILLISECONDS);
                ok++;
            } catch (Exception e) {
                // continue remaining; count failures in metrics already
            }
        }
        flush();
        return ok;
    }

    public void flush() {
        try {
            producer.flush();
        } catch (Exception e) {
            throw new KafkaException("flush failed: " + e.getMessage(), e, "flush", null);
        }
    }

    public long inflight() {
        return inflight.get();
    }

    private static Map<String, String> mergeHeaders(Map<String, String> a, Map<String, String> b) {
        if ((a == null || a.isEmpty()) && (b == null || b.isEmpty())) return Map.of();
        Map<String, String> out = new LinkedHashMap<>();
        if (a != null) out.putAll(a);
        if (b != null) out.putAll(b);
        return out;
    }

    @Override
    public void close() {
        if (ownProducer) {
            try {
                producer.close(Duration.ofSeconds(30));
            } catch (Exception ignored) {
            }
        }
    }
}
