package org.bytedeco.pytorch.utils.kafka;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.Closeable;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Continuous Kafka → DataFrame batch stream for online training / feature join loops.
 *
 * <pre>{@code
 * try (KafkaStream stream = Kafka.stream(opts)) {
 *     stream.forEachBatch(2048, df -> {
 *         DataFrame feats = df.feature().standardScale("age").build();
 *         // train / rank step
 *         stream.commit();
 *     });
 * }
 * }</pre>
 *
 * <p>Batching is by record count (primary) with optional idle timeout to flush a partial batch.
 * When {@code maxBatches} or {@code idleStop} is hit the loop ends cleanly.
 */
public final class KafkaStream implements Closeable, Iterable<DataFrame> {

    private final KafkaConsumer consumer;
    private final boolean ownConsumer;
    private final KafkaOptions options;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final AtomicBoolean stop = new AtomicBoolean(false);

    private int batchRows = 2048;
    private Duration pollTimeout;
    private Duration idleTimeout = Duration.ofSeconds(10);
    private long maxBatches = Long.MAX_VALUE;
    private boolean autoCommit = false;
    private boolean commitOnBatch = true;

    private KafkaStream(KafkaConsumer consumer, boolean ownConsumer, KafkaOptions options) {
        this.consumer = Objects.requireNonNull(consumer, "consumer");
        this.ownConsumer = ownConsumer;
        this.options = options == null ? KafkaOptions.defaults() : options;
        this.pollTimeout = this.options.pollTimeout();
        if (this.options.consumer() != null) {
            this.batchRows = Math.max(1, this.options.consumer().maxPollRecords());
        }
    }

    public static KafkaStream open(KafkaOptions options) {
        Objects.requireNonNull(options, "options");
        KafkaConsumer c = KafkaConsumer.connect(options);
        if (options.topicName() != null) {
            c.subscribe(options.topicName());
        }
        return new KafkaStream(c, true, options);
    }

    public static KafkaStream wrap(KafkaConsumer consumer, KafkaOptions options) {
        return new KafkaStream(consumer, false, options);
    }

    public KafkaStream batchRows(int batchRows) {
        this.batchRows = Math.max(1, batchRows);
        return this;
    }

    public KafkaStream pollTimeout(Duration pollTimeout) {
        this.pollTimeout = pollTimeout == null ? Duration.ofMillis(1000) : pollTimeout;
        return this;
    }

    public KafkaStream idleTimeout(Duration idleTimeout) {
        this.idleTimeout = idleTimeout == null ? Duration.ofSeconds(10) : idleTimeout;
        return this;
    }

    public KafkaStream maxBatches(long maxBatches) {
        this.maxBatches = maxBatches <= 0 ? Long.MAX_VALUE : maxBatches;
        return this;
    }

    /**
     * When true, commit after each successfully delivered batch (default true with manual commit client).
     */
    public KafkaStream commitOnBatch(boolean commitOnBatch) {
        this.commitOnBatch = commitOnBatch;
        return this;
    }

    public KafkaStream autoCommit(boolean autoCommit) {
        this.autoCommit = autoCommit;
        return this;
    }

    public KafkaConsumer consumer() {
        return consumer;
    }

    public KafkaOptions options() {
        return options;
    }

    public KafkaMetrics metrics() {
        return consumer.metrics();
    }

    public void stop() {
        stop.set(true);
        try {
            consumer.wakeup();
        } catch (Exception ignored) {
        }
    }

    public void commit() {
        consumer.commitSync();
    }

    public void pause() {
        consumer.pause();
    }

    public void resume() {
        consumer.resume();
    }

    public long lag() {
        return KafkaOffsets.lag(consumer.raw(), null);
    }

    /**
     * Blocking loop: deliver DataFrame batches to {@code handler} until stop / maxBatches / idle.
     *
     * @return total rows delivered
     */
    public long forEachBatch(Consumer<DataFrame> handler) {
        return forEachBatch(batchRows, handler);
    }

    public long forEachBatch(int rowsPerBatch, Consumer<DataFrame> handler) {
        Objects.requireNonNull(handler, "handler");
        int size = Math.max(1, rowsPerBatch);
        long totalRows = 0L;
        long batches = 0L;
        List<KafkaRecord> acc = new ArrayList<>(size);
        long idleDeadline = -1L;

        while (!stop.get() && !closed.get() && batches < maxBatches) {
            List<KafkaRecord> polled;
            try {
                polled = consumer.poll(pollTimeout, options);
            } catch (KafkaException e) {
                if (stop.get() || closed.get()) break;
                // wakeup during stop
                if (e.getCause() != null
                        && e.getCause().getClass().getName().contains("WakeupException")) {
                    break;
                }
                throw e;
            }
            if (polled.isEmpty()) {
                if (!acc.isEmpty()) {
                    // flush partial on idle
                    if (idleDeadline < 0) idleDeadline = System.nanoTime() + idleTimeout.toNanos();
                    if (System.nanoTime() >= idleDeadline) {
                        totalRows += deliver(acc, handler);
                        batches++;
                        acc = new ArrayList<>(size);
                        idleDeadline = -1L;
                    }
                } else {
                    if (idleDeadline < 0) idleDeadline = System.nanoTime() + idleTimeout.toNanos();
                    if (System.nanoTime() >= idleDeadline) break;
                }
                continue;
            }
            idleDeadline = -1L;
            for (KafkaRecord r : polled) {
                acc.add(r);
                if (acc.size() >= size) {
                    totalRows += deliver(acc, handler);
                    batches++;
                    acc = new ArrayList<>(size);
                    if (batches >= maxBatches || stop.get()) break;
                }
            }
        }
        if (!acc.isEmpty() && !closed.get() && batches < maxBatches) {
            totalRows += deliver(acc, handler);
        }
        consumer.metrics().setLag(lag());
        return totalRows;
    }

    private long deliver(List<KafkaRecord> acc, Consumer<DataFrame> handler) {
        DataFrame df = KafkaConsumer.recordsToDataFrame(acc, options.includeMetadata(), options.valueFormat());
        long n = df.rowCount();
        handler.accept(df);
        if (commitOnBatch && !autoCommit && !options.consumer().enableAutoCommit()) {
            try {
                consumer.commitSync(KafkaConsumer.offsetsToCommit(acc));
            } catch (Exception e) {
                throw new KafkaException("commit after batch failed: " + e.getMessage(), e, "commit", options.topicName());
            }
        }
        return n;
    }

    /**
     * Pull the next batch (may be empty on idle timeout). Returns empty DF when stopped.
     */
    public DataFrame nextBatch() {
        return nextBatch(batchRows);
    }

    public DataFrame nextBatch(int rowsPerBatch) {
        if (stop.get() || closed.get()) return DataFrame.create();
        return consumer.pollDataFrameUntil(rowsPerBatch, pollTimeout, idleTimeout, options);
    }

    @Override
    public Iterator<DataFrame> iterator() {
        return new Iterator<>() {
            private DataFrame next;
            private boolean fetched;
            private long seen;

            private void fetch() {
                if (fetched) return;
                if (stop.get() || closed.get() || seen >= maxBatches) {
                    next = null;
                    fetched = true;
                    return;
                }
                next = nextBatch(batchRows);
                if (next == null || next.rowCount() == 0) {
                    next = null;
                } else {
                    seen++;
                    if (commitOnBatch && !autoCommit && !options.consumer().enableAutoCommit()) {
                        try {
                            consumer.commitSync();
                        } catch (Exception ignored) {
                        }
                    }
                }
                fetched = true;
            }

            @Override
            public boolean hasNext() {
                fetch();
                return next != null;
            }

            @Override
            public DataFrame next() {
                fetch();
                if (next == null) throw new NoSuchElementException();
                DataFrame out = next;
                next = null;
                fetched = false;
                return out;
            }
        };
    }

    /** Sequential stream of batches; close the KafkaStream (not the java stream) to release the consumer. */
    public Stream<DataFrame> batchStream() {
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(iterator(), Spliterator.ORDERED | Spliterator.NONNULL),
                false);
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        stop.set(true);
        if (ownConsumer) {
            try {
                consumer.close();
            } catch (Exception ignored) {
            }
        }
    }
}
