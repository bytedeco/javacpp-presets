package org.bytedeco.pytorch.utils.kafka;

import org.apache.kafka.clients.consumer.Consumer;
import org.apache.kafka.clients.consumer.OffsetAndMetadata;
import org.apache.kafka.common.TopicPartition;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Offset / lag helpers shared by consumer paths and online-training loops.
 *
 * <p>Lag = sum over assigned partitions of {@code endOffset - position}
 * (or {@code endOffset - committed} when using committed offsets).
 */
public final class KafkaOffsets {

    private KafkaOffsets() {}

    /**
     * Current consumer lag across {@code partitions} (or current assignment if null).
     * Uses live positions (not committed offsets).
     */
    public static long lag(Consumer<byte[], byte[]> consumer, Collection<TopicPartition> partitions) {
        Objects.requireNonNull(consumer, "consumer");
        Set<TopicPartition> tps = resolve(consumer, partitions);
        if (tps.isEmpty()) return 0L;
        Map<TopicPartition, Long> end = consumer.endOffsets(tps);
        long lag = 0L;
        for (TopicPartition tp : tps) {
            long endOff = end.getOrDefault(tp, 0L);
            long pos;
            try {
                pos = consumer.position(tp);
            } catch (Exception e) {
                pos = endOff;
            }
            lag += Math.max(0L, endOff - pos);
        }
        return lag;
    }

    /**
     * Lag vs committed offsets for a consumer group view on the given partitions.
     */
    public static long committedLag(Consumer<byte[], byte[]> consumer,
                                    Collection<TopicPartition> partitions) {
        Objects.requireNonNull(consumer, "consumer");
        Set<TopicPartition> tps = resolve(consumer, partitions);
        if (tps.isEmpty()) return 0L;
        Map<TopicPartition, Long> end = consumer.endOffsets(tps);
        Map<TopicPartition, OffsetAndMetadata> committed = consumer.committed(tps);
        long lag = 0L;
        for (TopicPartition tp : tps) {
            long endOff = end.getOrDefault(tp, 0L);
            OffsetAndMetadata om = committed == null ? null : committed.get(tp);
            long committedOff = om == null ? 0L : om.offset();
            lag += Math.max(0L, endOff - committedOff);
        }
        return lag;
    }

    /** Per-partition lag map (position-based). */
    public static Map<TopicPartition, Long> lagByPartition(
            Consumer<byte[], byte[]> consumer,
            Collection<TopicPartition> partitions) {
        Objects.requireNonNull(consumer, "consumer");
        Set<TopicPartition> tps = resolve(consumer, partitions);
        if (tps.isEmpty()) return Map.of();
        Map<TopicPartition, Long> end = consumer.endOffsets(tps);
        Map<TopicPartition, Long> out = new LinkedHashMap<>();
        for (TopicPartition tp : tps) {
            long endOff = end.getOrDefault(tp, 0L);
            long pos;
            try {
                pos = consumer.position(tp);
            } catch (Exception e) {
                pos = endOff;
            }
            out.put(tp, Math.max(0L, endOff - pos));
        }
        return Collections.unmodifiableMap(out);
    }

    public static Map<TopicPartition, Long> endOffsets(
            Consumer<byte[], byte[]> consumer,
            Collection<TopicPartition> partitions) {
        return consumer.endOffsets(resolve(consumer, partitions));
    }

    public static Map<TopicPartition, Long> beginningOffsets(
            Consumer<byte[], byte[]> consumer,
            Collection<TopicPartition> partitions) {
        return consumer.beginningOffsets(resolve(consumer, partitions));
    }

    public static Map<TopicPartition, Long> positions(
            Consumer<byte[], byte[]> consumer,
            Collection<TopicPartition> partitions) {
        Set<TopicPartition> tps = resolve(consumer, partitions);
        Map<TopicPartition, Long> out = new LinkedHashMap<>();
        for (TopicPartition tp : tps) {
            try {
                out.put(tp, consumer.position(tp));
            } catch (Exception e) {
                out.put(tp, -1L);
            }
        }
        return Collections.unmodifiableMap(out);
    }

    /**
     * Seek every assigned partition to its committed offset (or beginning if none).
     */
    public static void seekToCommitted(Consumer<byte[], byte[]> consumer) {
        Set<TopicPartition> tps = consumer.assignment();
        if (tps.isEmpty()) return;
        Map<TopicPartition, OffsetAndMetadata> committed = consumer.committed(tps);
        for (TopicPartition tp : tps) {
            OffsetAndMetadata om = committed == null ? null : committed.get(tp);
            if (om != null) {
                consumer.seek(tp, om.offset());
            } else {
                consumer.seekToBeginning(Set.of(tp));
            }
        }
    }

    /**
     * Build commit map: next offset = last seen offset + 1 per partition.
     */
    public static Map<TopicPartition, OffsetAndMetadata> nextOffsets(Map<TopicPartition, Long> lastOffsets) {
        if (lastOffsets == null || lastOffsets.isEmpty()) return Map.of();
        Map<TopicPartition, OffsetAndMetadata> out = new HashMap<>();
        for (Map.Entry<TopicPartition, Long> e : lastOffsets.entrySet()) {
            if (e.getKey() == null || e.getValue() == null) continue;
            out.put(e.getKey(), new OffsetAndMetadata(e.getValue() + 1));
        }
        return out;
    }

    private static Set<TopicPartition> resolve(Consumer<byte[], byte[]> consumer,
                                               Collection<TopicPartition> partitions) {
        if (partitions != null && !partitions.isEmpty()) {
            return new HashSet<>(partitions);
        }
        Set<TopicPartition> assigned = consumer.assignment();
        return assigned == null ? Set.of() : new HashSet<>(assigned);
    }
}
