package org.bytedeco.pytorch.utils.kafka;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

/**
 * Lightweight in-process counters for high-QPS Kafka produce / consume paths.
 * No external metrics backend — callers may scrape {@link #snapshot()}.
 */
public final class KafkaMetrics {

    private final LongAdder produced = new LongAdder();
    private final LongAdder produceFailed = new LongAdder();
    private final LongAdder produceBytes = new LongAdder();
    private final LongAdder consumed = new LongAdder();
    private final LongAdder consumeFailed = new LongAdder();
    private final LongAdder consumeBytes = new LongAdder();
    private final LongAdder committed = new LongAdder();
    private final LongAdder batches = new LongAdder();
    private final DoubleAdder produceLatencyMs = new DoubleAdder();
    private final AtomicLong produceLatencyCount = new AtomicLong();
    private final AtomicLong lastLag = new AtomicLong(-1L);
    private final long startedAtMs = System.currentTimeMillis();

    public void recordProduce(int records, long bytes, double latencyMs, boolean ok) {
        if (ok) {
            produced.add(records);
            produceBytes.add(Math.max(0L, bytes));
        } else {
            produceFailed.add(Math.max(1, records));
        }
        if (latencyMs >= 0) {
            produceLatencyMs.add(latencyMs);
            produceLatencyCount.incrementAndGet();
        }
    }

    public void recordConsume(int records, long bytes, boolean ok) {
        if (ok) {
            consumed.add(records);
            consumeBytes.add(Math.max(0L, bytes));
            batches.increment();
        } else {
            consumeFailed.add(Math.max(1, records));
        }
    }

    public void recordCommit(int partitions) {
        committed.add(Math.max(1, partitions));
    }

    public void setLag(long lag) {
        lastLag.set(lag);
    }

    public long produced() {
        return produced.sum();
    }

    public long consumed() {
        return consumed.sum();
    }

    public long produceFailed() {
        return produceFailed.sum();
    }

    public long consumeFailed() {
        return consumeFailed.sum();
    }

    public long produceBytes() {
        return produceBytes.sum();
    }

    public long consumeBytes() {
        return consumeBytes.sum();
    }

    public long batches() {
        return batches.sum();
    }

    public long lag() {
        return lastLag.get();
    }

    public double avgProduceLatencyMs() {
        long n = produceLatencyCount.get();
        if (n <= 0) return 0.0;
        return produceLatencyMs.sum() / n;
    }

    public double produceQps() {
        double sec = Math.max(0.001, (System.currentTimeMillis() - startedAtMs) / 1000.0);
        return produced.sum() / sec;
    }

    public double consumeQps() {
        double sec = Math.max(0.001, (System.currentTimeMillis() - startedAtMs) / 1000.0);
        return consumed.sum() / sec;
    }

    public Snapshot snapshot() {
        return new Snapshot(
                produced.sum(),
                produceFailed.sum(),
                produceBytes.sum(),
                consumed.sum(),
                consumeFailed.sum(),
                consumeBytes.sum(),
                committed.sum(),
                batches.sum(),
                avgProduceLatencyMs(),
                lastLag.get(),
                produceQps(),
                consumeQps(),
                System.currentTimeMillis() - startedAtMs
        );
    }

    public void reset() {
        produced.reset();
        produceFailed.reset();
        produceBytes.reset();
        consumed.reset();
        consumeFailed.reset();
        consumeBytes.reset();
        committed.reset();
        batches.reset();
        produceLatencyMs.reset();
        produceLatencyCount.set(0);
        lastLag.set(-1L);
    }

    @Override
    public String toString() {
        return snapshot().toString();
    }

    /** Immutable point-in-time view. */
    public record Snapshot(
            long produced,
            long produceFailed,
            long produceBytes,
            long consumed,
            long consumeFailed,
            long consumeBytes,
            long committed,
            long batches,
            double avgProduceLatencyMs,
            long lag,
            double produceQps,
            double consumeQps,
            long uptimeMs
    ) {
        @Override
        public String toString() {
            return "KafkaMetrics{produced=" + produced
                    + ", produceFailed=" + produceFailed
                    + ", consumed=" + consumed
                    + ", consumeFailed=" + consumeFailed
                    + ", batches=" + batches
                    + ", lag=" + lag
                    + ", produceQps=" + String.format("%.1f", produceQps)
                    + ", consumeQps=" + String.format("%.1f", consumeQps)
                    + ", avgProduceLatencyMs=" + String.format("%.2f", avgProduceLatencyMs)
                    + ", uptimeMs=" + uptimeMs
                    + '}';
        }
    }
}
