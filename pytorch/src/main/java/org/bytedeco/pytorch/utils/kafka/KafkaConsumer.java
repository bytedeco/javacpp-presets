package org.bytedeco.pytorch.utils.kafka;

import org.apache.kafka.clients.consumer.Consumer;
import org.apache.kafka.clients.consumer.ConsumerRebalanceListener;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.OffsetAndMetadata;
import org.apache.kafka.common.PartitionInfo;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.header.Header;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Batch Kafka consumer wrapper.
 *
 * <p><b>Not thread-safe</b> (mirrors upstream). Enterprise defaults: manual commit,
 * cooperative-sticky assignor, {@code read_committed}, tunable {@code max.poll.records}
 * for FE / GPU train step sizes.
 *
 * <pre>{@code
 * try (KafkaConsumer c = KafkaConsumer.connect(opts)) {
 *     c.subscribe("rec.feature.log");
 *     DataFrame batch = c.pollDataFrame(Duration.ofSeconds(1), opts);
 *     // feature engineering / train step …
 *     c.commitSync();
 * }
 * }</pre>
 */
public final class KafkaConsumer implements Closeable {

    private final Consumer<byte[], byte[]> consumer;
    private final boolean ownConsumer;
    private final KafkaOptions options;
    private final KafkaMetrics metrics;
    private final Set<TopicPartition> paused = new HashSet<>();

    private KafkaConsumer(Consumer<byte[], byte[]> consumer, boolean ownConsumer,
                          KafkaOptions options, KafkaMetrics metrics) {
        this.consumer = Objects.requireNonNull(consumer, "consumer");
        this.ownConsumer = ownConsumer;
        this.options = options == null ? KafkaOptions.defaults() : options;
        this.metrics = metrics == null ? new KafkaMetrics() : metrics;
    }

    public static KafkaConsumer connect(String bootstrapServers, String groupId) {
        return connect(KafkaOptions.builder()
                .bootstrapServers(bootstrapServers)
                .groupId(groupId)
                .build());
    }

    public static KafkaConsumer connect(KafkaOptions options) {
        Objects.requireNonNull(options, "options");
        org.apache.kafka.clients.consumer.KafkaConsumer<byte[], byte[]> c =
                new org.apache.kafka.clients.consumer.KafkaConsumer<>(options.consumerProperties());
        return new KafkaConsumer(c, true, options, new KafkaMetrics());
    }

    public static KafkaConsumer wrap(Consumer<byte[], byte[]> consumer, KafkaOptions options) {
        return new KafkaConsumer(consumer, false, options, new KafkaMetrics());
    }

    public Consumer<byte[], byte[]> raw() {
        return consumer;
    }

    public KafkaOptions options() {
        return options;
    }

    public KafkaMetrics metrics() {
        return metrics;
    }

    // ── subscribe / assign ───────────────────────────────────────────────────

    public void subscribe(String... topics) {
        if (topics == null || topics.length == 0) {
            throw new IllegalArgumentException("topics required");
        }
        subscribe(List.of(topics), null);
    }

    public void subscribe(Collection<String> topics) {
        subscribe(topics, null);
    }

    public void subscribe(Collection<String> topics, ConsumerRebalanceListener listener) {
        Objects.requireNonNull(topics, "topics");
        if (topics.isEmpty()) throw new IllegalArgumentException("topics empty");
        try {
            if (listener == null) {
                consumer.subscribe(topics);
            } else {
                consumer.subscribe(topics, listener);
            }
        } catch (Exception e) {
            throw new KafkaException("subscribe failed: " + e.getMessage(), e, "subscribe", null);
        }
    }

    public void subscribePattern(String regex) {
        subscribePattern(Pattern.compile(regex), null);
    }

    public void subscribePattern(Pattern pattern, ConsumerRebalanceListener listener) {
        Objects.requireNonNull(pattern, "pattern");
        try {
            if (listener == null) {
                consumer.subscribe(pattern);
            } else {
                consumer.subscribe(pattern, listener);
            }
        } catch (Exception e) {
            throw new KafkaException("subscribePattern failed: " + e.getMessage(), e, "subscribe", null);
        }
    }

    public void assign(Collection<TopicPartition> partitions) {
        Objects.requireNonNull(partitions, "partitions");
        try {
            consumer.assign(partitions);
        } catch (Exception e) {
            throw new KafkaException("assign failed: " + e.getMessage(), e, "assign", null);
        }
    }

    public void assign(String topic, int... partitions) {
        Objects.requireNonNull(topic, "topic");
        List<TopicPartition> tps = new ArrayList<>();
        if (partitions == null || partitions.length == 0) {
            List<PartitionInfo> infos = consumer.partitionsFor(topic);
            if (infos != null) {
                for (PartitionInfo pi : infos) {
                    tps.add(new TopicPartition(topic, pi.partition()));
                }
            }
        } else {
            for (int p : partitions) tps.add(new TopicPartition(topic, p));
        }
        assign(tps);
    }

    public void unsubscribe() {
        consumer.unsubscribe();
    }

    public Set<TopicPartition> assignment() {
        return consumer.assignment();
    }

    public Set<String> subscription() {
        return consumer.subscription();
    }

    // ── seek ─────────────────────────────────────────────────────────────────

    public void seek(TopicPartition partition, long offset) {
        consumer.seek(partition, offset);
    }

    public void seekToBeginning(Collection<TopicPartition> partitions) {
        consumer.seekToBeginning(partitions == null ? consumer.assignment() : partitions);
    }

    public void seekToBeginning() {
        seekToBeginning(null);
    }

    public void seekToEnd(Collection<TopicPartition> partitions) {
        consumer.seekToEnd(partitions == null ? consumer.assignment() : partitions);
    }

    public void seekToEnd() {
        seekToEnd(null);
    }

    public long position(TopicPartition partition) {
        return consumer.position(partition);
    }

    public Map<TopicPartition, Long> endOffsets(Collection<TopicPartition> partitions) {
        return consumer.endOffsets(partitions);
    }

    public Map<TopicPartition, Long> beginningOffsets(Collection<TopicPartition> partitions) {
        return consumer.beginningOffsets(partitions);
    }

    public OffsetAndMetadata committed(TopicPartition partition) {
        Map<TopicPartition, OffsetAndMetadata> m = consumer.committed(Set.of(partition));
        return m == null ? null : m.get(partition);
    }

    // ── poll ─────────────────────────────────────────────────────────────────

    public List<KafkaRecord> poll(Duration timeout) {
        return poll(timeout, options);
    }

    public List<KafkaRecord> poll(Duration timeout, KafkaOptions opts) {
        KafkaOptions o = opts == null ? options : opts;
        Duration t = timeout == null ? o.pollTimeout() : timeout;
        try {
            // auto-subscribe topic from options if nothing assigned/subscribed
            ensureSubscribed(o);
            ConsumerRecords<byte[], byte[]> records = consumer.poll(t);
            List<KafkaRecord> out = new ArrayList<>(records.count());
            long bytes = 0L;
            for (ConsumerRecord<byte[], byte[]> r : records) {
                KafkaRecord kr = toKafkaRecord(r, o);
                out.add(kr);
                byte[] v = r.value();
                byte[] k = r.key();
                bytes += (v == null ? 0 : v.length) + (k == null ? 0 : k.length);
            }
            metrics.recordConsume(out.size(), bytes, true);
            return out;
        } catch (KafkaException e) {
            throw e;
        } catch (Exception e) {
            metrics.recordConsume(1, 0, false);
            throw new KafkaException("poll failed: " + e.getMessage(), e, "poll", o.topicName());
        }
    }

    /**
     * Poll once and materialise a {@link DataFrame} (empty frame if no records).
     */
    public DataFrame pollDataFrame(Duration timeout) {
        return pollDataFrame(timeout, options);
    }

    public DataFrame pollDataFrame(Duration timeout, KafkaOptions opts) {
        KafkaOptions o = opts == null ? options : opts;
        List<KafkaRecord> records = poll(timeout, o);
        return recordsToDataFrame(records, o.includeMetadata(), o.valueFormat());
    }

    /**
     * Poll until {@code maxRecords} accumulated or idle timeout between polls.
     */
    public DataFrame pollDataFrameUntil(int maxRecords, Duration pollTimeout, Duration idleTimeout) {
        return pollDataFrameUntil(maxRecords, pollTimeout, idleTimeout, options);
    }

    public DataFrame pollDataFrameUntil(int maxRecords, Duration pollTimeout,
                                        Duration idleTimeout, KafkaOptions opts) {
        KafkaOptions o = opts == null ? options : opts;
        int limit = maxRecords <= 0 ? o.consumer().maxPollRecords() : maxRecords;
        Duration idle = idleTimeout == null ? Duration.ofSeconds(5) : idleTimeout;
        List<KafkaRecord> acc = new ArrayList<>(limit);
        long deadlineIdle = -1L;
        while (acc.size() < limit) {
            List<KafkaRecord> batch = poll(pollTimeout == null ? o.pollTimeout() : pollTimeout, o);
            if (batch.isEmpty()) {
                if (deadlineIdle < 0) deadlineIdle = System.nanoTime() + idle.toNanos();
                if (System.nanoTime() >= deadlineIdle) break;
                continue;
            }
            deadlineIdle = -1L;
            for (KafkaRecord r : batch) {
                acc.add(r);
                if (acc.size() >= limit) break;
            }
        }
        return recordsToDataFrame(acc, o.includeMetadata(), o.valueFormat());
    }

    public static DataFrame recordsToDataFrame(List<KafkaRecord> records,
                                               boolean includeMetadata,
                                               KafkaOptions.ValueFormat format) {
        if (records == null || records.isEmpty()) {
            return DataFrame.create();
        }
        List<Map<String, Object>> rows = new ArrayList<>(records.size());
        for (KafkaRecord r : records) {
            // re-decode with explicit format if value still raw
            Map<String, Object> row = r.toRowMap(includeMetadata);
            rows.add(row);
        }
        return DataFrame.fromRecords(rows);
    }

    // ── commit ───────────────────────────────────────────────────────────────

    public void commitSync() {
        try {
            consumer.commitSync();
            metrics.recordCommit(consumer.assignment().size());
        } catch (Exception e) {
            throw new KafkaException("commitSync failed: " + e.getMessage(), e, "commitSync", null);
        }
    }

    public void commitSync(Map<TopicPartition, OffsetAndMetadata> offsets) {
        try {
            consumer.commitSync(offsets);
            metrics.recordCommit(offsets == null ? 0 : offsets.size());
        } catch (Exception e) {
            throw new KafkaException("commitSync failed: " + e.getMessage(), e, "commitSync", null);
        }
    }

    public void commitAsync() {
        consumer.commitAsync((offsets, exception) -> {
            if (exception == null) {
                metrics.recordCommit(offsets == null ? 0 : offsets.size());
            }
        });
    }

    public void commitAsync(Map<TopicPartition, OffsetAndMetadata> offsets) {
        consumer.commitAsync(offsets, (o, exception) -> {
            if (exception == null) {
                metrics.recordCommit(o == null ? 0 : o.size());
            }
        });
    }

    /**
     * Build next-offset map from the last polled records (offset+1 per partition).
     */
    public static Map<TopicPartition, OffsetAndMetadata> offsetsToCommit(List<KafkaRecord> records) {
        if (records == null || records.isEmpty()) return Map.of();
        Map<TopicPartition, Long> max = new HashMap<>();
        for (KafkaRecord r : records) {
            if (r.topic() == null || r.partition() == null || r.offset() == null) continue;
            TopicPartition tp = new TopicPartition(r.topic(), r.partition());
            max.merge(tp, r.offset(), Math::max);
        }
        Map<TopicPartition, OffsetAndMetadata> out = new HashMap<>();
        for (Map.Entry<TopicPartition, Long> e : max.entrySet()) {
            out.put(e.getKey(), new OffsetAndMetadata(e.getValue() + 1));
        }
        return out;
    }

    // ── pause / resume (backpressure) ────────────────────────────────────────

    public void pause(Collection<TopicPartition> partitions) {
        Collection<TopicPartition> ps = partitions == null ? consumer.assignment() : partitions;
        consumer.pause(ps);
        paused.addAll(ps);
    }

    public void pause() {
        pause(null);
    }

    public void resume(Collection<TopicPartition> partitions) {
        Collection<TopicPartition> ps = partitions == null ? new HashSet<>(paused) : partitions;
        consumer.resume(ps);
        paused.removeAll(ps);
    }

    public void resume() {
        resume(null);
    }

    public Set<TopicPartition> paused() {
        return Collections.unmodifiableSet(new HashSet<>(paused));
    }

    public void wakeup() {
        consumer.wakeup();
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private void ensureSubscribed(KafkaOptions o) {
        if (!consumer.subscription().isEmpty() || !consumer.assignment().isEmpty()) return;
        String topic = o.topicName();
        if (topic != null && !topic.isBlank()) {
            subscribe(topic);
        }
    }

    static KafkaRecord toKafkaRecord(ConsumerRecord<byte[], byte[]> r, KafkaOptions opts) {
        Map<String, String> headers = new LinkedHashMap<>();
        if (r.headers() != null) {
            for (Header h : r.headers()) {
                if (h == null || h.key() == null) continue;
                String hv = h.value() == null ? null : new String(h.value(), StandardCharsets.UTF_8);
                headers.put(h.key(), hv);
            }
        }
        Object value = KafkaSerde.decodeToObject(r.value(),
                opts == null ? KafkaOptions.ValueFormat.JSON : opts.valueFormat());
        return KafkaRecord.builder()
                .topic(r.topic())
                .partition(r.partition())
                .offset(r.offset())
                .timestamp(r.timestamp())
                .key(KafkaSerde.decodeKey(r.key()))
                .value(value)
                .headers(headers)
                .build();
    }

    @Override
    public void close() {
        if (ownConsumer) {
            try {
                // kafka-clients 4.x: Duration overload is deprecated; use CloseOptions.
                consumer.close(org.apache.kafka.clients.consumer.CloseOptions
                        .timeout(Duration.ofSeconds(30)));
            } catch (Exception ignored) {
            }
        }
    }
}
