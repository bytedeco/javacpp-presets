/*
 * Nearline / online learning hooks for recommendation models.
 *
 * Industry patterns:
 *   - Google / YouTube: nearline embedding updates from recent interactions
 *   - Alibaba: real-time example stream -> parameter server incremental update
 *   - ByteDance: streaming sample join + async model patch
 *   - Meta: frequent batch refresh + streaming feature freshness
 *
 * This module does NOT train models itself; it defines the contract for:
 *   1. Accepting interaction events
 *   2. Buffering / batching
 *   3. Triggering incremental update callbacks
 *   4. Publishing new artifact versions to {@link ModelRegistry}
 */
package org.bytedeco.pytorch.utils.recommend.modelops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/** Online / nearline learning event loop. */
public final class OnlineLearningHook {

    /** One user interaction usable as a training signal. */
    public static final class InteractionEvent {
        public final String userId;
        public final String itemId;
        public final String eventType; // click, impress, convert, dwell, like, ...
        public final float label;
        public final long timestampMs;
        public final float weight;

        public InteractionEvent(
                String userId, String itemId, String eventType, float label, long timestampMs) {
            this(userId, itemId, eventType, label, timestampMs, 1.0f);
        }

        public InteractionEvent(
                String userId,
                String itemId,
                String eventType,
                float label,
                long timestampMs,
                float weight) {
            this.userId = Objects.requireNonNull(userId, "userId");
            this.itemId = Objects.requireNonNull(itemId, "itemId");
            this.eventType = eventType != null ? eventType : "click";
            this.label = label;
            this.timestampMs = timestampMs;
            this.weight = weight;
        }
    }

    /** Result of one incremental training flush. */
    public static final class UpdateResult {
        public final boolean success;
        public final int exampleCount;
        public final String newArtifactUri;
        public final String message;
        public final long durationMs;

        public UpdateResult(
                boolean success, int exampleCount, String newArtifactUri, String message, long durationMs) {
            this.success = success;
            this.exampleCount = exampleCount;
            this.newArtifactUri = newArtifactUri;
            this.message = message != null ? message : "";
            this.durationMs = durationMs;
        }

        @Override
        public String toString() {
            return "UpdateResult{success=" + success + ", n=" + exampleCount
                    + ", artifact=" + newArtifactUri + ", msg=" + message + "}";
        }
    }

    /**
     * Pluggable incremental trainer — wraps PS update / embedding patch / LoRA, etc.
     */
    public interface IncrementalTrainer {
        UpdateResult train(List<InteractionEvent> batch, ModelVersion baseVersion) throws Exception;
    }

    private final String modelName;
    private final ModelRegistry registry;
    private final IncrementalTrainer trainer;
    private final int flushSize;
    private final long flushIntervalMs;
    private final ConcurrentLinkedQueue<InteractionEvent> buffer = new ConcurrentLinkedQueue<>();
    private final AtomicLong enqueued = new AtomicLong();
    private final AtomicLong flushed = new AtomicLong();
    private final AtomicLong failedFlushes = new AtomicLong();
    private volatile long lastFlushMs = System.currentTimeMillis();
    private volatile String baseVersionId;
    private Consumer<UpdateResult> onSuccess;
    private Consumer<Exception> onError;

    public OnlineLearningHook(
            String modelName,
            ModelRegistry registry,
            IncrementalTrainer trainer,
            String baseVersionId,
            int flushSize,
            long flushIntervalMs) {
        this.modelName = Objects.requireNonNull(modelName);
        this.registry = Objects.requireNonNull(registry);
        this.trainer = Objects.requireNonNull(trainer);
        this.baseVersionId = baseVersionId;
        this.flushSize = Math.max(1, flushSize);
        this.flushIntervalMs = Math.max(1000L, flushIntervalMs);
    }

    public OnlineLearningHook onSuccess(Consumer<UpdateResult> onSuccess) {
        this.onSuccess = onSuccess;
        return this;
    }

    public OnlineLearningHook onError(Consumer<Exception> onError) {
        this.onError = onError;
        return this;
    }

    public void setBaseVersionId(String baseVersionId) {
        this.baseVersionId = baseVersionId;
    }

    /** Ingest one interaction; may trigger flush. */
    public void accept(InteractionEvent event) {
        Objects.requireNonNull(event, "event");
        buffer.add(event);
        enqueued.incrementAndGet();
        if (buffer.size() >= flushSize
                || System.currentTimeMillis() - lastFlushMs >= flushIntervalMs) {
            flush();
        }
    }

    public void accept(String userId, String itemId, String eventType, float label) {
        accept(new InteractionEvent(userId, itemId, eventType, label, System.currentTimeMillis()));
    }

    /**
     * Force flush buffered events through the incremental trainer.
     * On success, registers a new model version at TRAINED stage (caller promotes).
     */
    public synchronized UpdateResult flush() {
        List<InteractionEvent> batch = new ArrayList<>();
        InteractionEvent e;
        while ((e = buffer.poll()) != null) {
            batch.add(e);
        }
        if (batch.isEmpty()) {
            return new UpdateResult(true, 0, null, "empty", 0L);
        }
        if (baseVersionId == null) {
            failedFlushes.incrementAndGet();
            return new UpdateResult(false, batch.size(), null, "no base version", 0L);
        }
        ModelVersion base;
        try {
            base = registry.require(modelName, baseVersionId);
        } catch (RuntimeException ex) {
            // put back
            buffer.addAll(batch);
            failedFlushes.incrementAndGet();
            if (onError != null) onError.accept(ex);
            return new UpdateResult(false, batch.size(), null, ex.getMessage(), 0L);
        }
        long t0 = System.currentTimeMillis();
        try {
            UpdateResult result = trainer.train(batch, base);
            long dur = System.currentTimeMillis() - t0;
            lastFlushMs = System.currentTimeMillis();
            flushed.addAndGet(batch.size());
            if (result.success && result.newArtifactUri != null && !result.newArtifactUri.isEmpty()) {
                String newVid = baseVersionId + ".ol." + System.currentTimeMillis();
                ModelVersion neo = ModelVersion.builder(modelName, newVid)
                        .artifactUri(result.newArtifactUri)
                        .framework(base.framework())
                        .stage(ModelStage.TRAINED)
                        .parentVersionId(baseVersionId)
                        .trainingJobId("online-" + lastFlushMs)
                        .description("online learning patch from " + batch.size() + " events")
                        .offlineMetrics(base.offlineMetrics())
                        .build();
                try {
                    registry.register(neo);
                } catch (RuntimeException regEx) {
                    // version may race; ignore duplicate
                }
                UpdateResult enriched = new UpdateResult(
                        true, batch.size(), result.newArtifactUri,
                        result.message + "; registered=" + newVid, dur);
                if (onSuccess != null) onSuccess.accept(enriched);
                return enriched;
            }
            UpdateResult timed = new UpdateResult(
                    result.success, batch.size(), result.newArtifactUri, result.message, dur);
            if (result.success && onSuccess != null) onSuccess.accept(timed);
            if (!result.success) failedFlushes.incrementAndGet();
            return timed;
        } catch (Exception ex) {
            failedFlushes.incrementAndGet();
            // Re-queue to avoid silent drop (at-least-once; caller should dedup).
            buffer.addAll(batch);
            if (onError != null) onError.accept(ex);
            return new UpdateResult(false, batch.size(), null, ex.getMessage(),
                    System.currentTimeMillis() - t0);
        }
    }

    public int buffered() {
        return buffer.size();
    }

    public long enqueued() {
        return enqueued.get();
    }

    public long flushed() {
        return flushed.get();
    }

    public long failedFlushes() {
        return failedFlushes.get();
    }

    public List<InteractionEvent> snapshotBuffer() {
        return Collections.unmodifiableList(new ArrayList<>(buffer));
    }
}
