/*
 * Pipeline orchestrator — chains recall → coarse → fine → rerank → mix
 * with per-stage timeout, degradation, and metrics hooks.
 *
 * Mirrors production ranking services:
 *   - ByteDance / TikTok rank service
 *   - Alibaba TPP (The Partner Platform) rec graph
 *   - YouTube / Google retrieval-ranking stack
 *   - Meta feed ranking service
 *   - Tencent / Netflix blender
 */
package org.bytedeco.pytorch.utils.recommend.serving.pipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/** End-to-end multi-stage ranking pipeline. */
public final class PipelineOrchestrator {

    private final List<RankStage> stages;
    private final List<Candidate> ultimateFallback;
    private final Consumer<PipelineEvent> eventListener;

    private PipelineOrchestrator(Builder b) {
        this.stages = Collections.unmodifiableList(new ArrayList<>(b.stages));
        this.ultimateFallback = b.ultimateFallback == null
                ? List.of()
                : Collections.unmodifiableList(new ArrayList<>(b.ultimateFallback));
        this.eventListener = b.eventListener;
    }

    public static Builder builder() {
        return new Builder();
    }

    public List<RankStage> stages() {
        return stages;
    }

    /**
     * Run the full cascade for one request.
     */
    public PipelineResult run(RequestContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        long t0 = System.currentTimeMillis();
        List<RankStage.StageResult> stageResults = new ArrayList<>();
        List<Candidate> current = List.of();
        boolean anyDegraded = false;
        boolean anyTimeout = false;

        emit(PipelineEvent.started(ctx.requestId()));

        for (RankStage stage : stages) {
            if (ctx.deadlineExceeded()) {
                RankStage.StageResult timed = RankStage.StageResult.timeout(
                        stage.name(), current, 0L);
                stageResults.add(timed);
                anyTimeout = true;
                anyDegraded = true;
                emit(PipelineEvent.stageFinished(ctx.requestId(), timed));
                // Try stage fallback then continue / break.
                current = stage.fallback(ctx, current, new RuntimeException("deadline")).candidates;
                continue;
            }
            RankStage.StageResult result;
            try {
                result = stage.execute(ctx, current);
            } catch (RuntimeException ex) {
                result = stage.fallback(ctx, current, ex);
            }
            stageResults.add(result);
            emit(PipelineEvent.stageFinished(ctx.requestId(), result));
            if (result.degraded) anyDegraded = true;
            if (result.timedOut) anyTimeout = true;
            current = result.candidates;
        }

        if ((current == null || current.isEmpty()) && !ultimateFallback.isEmpty()) {
            current = new ArrayList<>();
            for (Candidate c : ultimateFallback) {
                current.add(c.copy().tag("fallback", "ultimate"));
            }
            anyDegraded = true;
            emit(PipelineEvent.ultimateFallback(ctx.requestId(), current.size()));
        }
        if (current == null) {
            current = List.of();
        }
        // Final renumber
        List<Candidate> finalList = new ArrayList<>(current.size());
        for (int i = 0; i < current.size(); i++) {
            Candidate c = current.get(i).copy();
            c.rank(i);
            finalList.add(c);
        }
        long latency = System.currentTimeMillis() - t0;
        PipelineResult result = new PipelineResult(
                ctx.requestId(),
                finalList,
                stageResults,
                latency,
                anyDegraded,
                anyTimeout,
                ctx.remainingBudgetMs() == 0 && latency > 0);
        emit(PipelineEvent.finished(ctx.requestId(), result));
        return result;
    }

    private void emit(PipelineEvent event) {
        if (eventListener != null) {
            try {
                eventListener.accept(event);
            } catch (RuntimeException ignored) {
            }
        }
    }

    // ---- result / events ----------------------------------------------------

    public static final class PipelineResult {
        public final String requestId;
        public final List<Candidate> items;
        public final List<RankStage.StageResult> stageResults;
        public final long totalLatencyMs;
        public final boolean degraded;
        public final boolean timedOut;
        public final boolean budgetExhausted;

        public PipelineResult(
                String requestId,
                List<Candidate> items,
                List<RankStage.StageResult> stageResults,
                long totalLatencyMs,
                boolean degraded,
                boolean timedOut,
                boolean budgetExhausted) {
            this.requestId = requestId;
            this.items = Collections.unmodifiableList(new ArrayList<>(items));
            this.stageResults = Collections.unmodifiableList(new ArrayList<>(stageResults));
            this.totalLatencyMs = totalLatencyMs;
            this.degraded = degraded;
            this.timedOut = timedOut;
            this.budgetExhausted = budgetExhausted;
        }

        public Map<String, Long> stageLatencies() {
            Map<String, Long> m = new LinkedHashMap<>();
            for (RankStage.StageResult s : stageResults) {
                m.put(s.stageName, s.latencyMs);
            }
            return m;
        }

        public Map<String, Integer> stageSizes() {
            Map<String, Integer> m = new LinkedHashMap<>();
            for (RankStage.StageResult s : stageResults) {
                m.put(s.stageName, s.size());
            }
            return m;
        }

        @Override
        public String toString() {
            return "PipelineResult{req=" + requestId + ", items=" + items.size()
                    + ", latencyMs=" + totalLatencyMs + ", degraded=" + degraded
                    + ", timedOut=" + timedOut + "}";
        }
    }

    public static final class PipelineEvent {
        public enum Type { STARTED, STAGE_FINISHED, ULTIMATE_FALLBACK, FINISHED }

        public final Type type;
        public final String requestId;
        public final RankStage.StageResult stageResult;
        public final PipelineResult pipelineResult;
        public final int fallbackSize;
        public final long timestampMs;

        private PipelineEvent(
                Type type,
                String requestId,
                RankStage.StageResult stageResult,
                PipelineResult pipelineResult,
                int fallbackSize) {
            this.type = type;
            this.requestId = requestId;
            this.stageResult = stageResult;
            this.pipelineResult = pipelineResult;
            this.fallbackSize = fallbackSize;
            this.timestampMs = System.currentTimeMillis();
        }

        static PipelineEvent started(String requestId) {
            return new PipelineEvent(Type.STARTED, requestId, null, null, 0);
        }

        static PipelineEvent stageFinished(String requestId, RankStage.StageResult sr) {
            return new PipelineEvent(Type.STAGE_FINISHED, requestId, sr, null, 0);
        }

        static PipelineEvent ultimateFallback(String requestId, int size) {
            return new PipelineEvent(Type.ULTIMATE_FALLBACK, requestId, null, null, size);
        }

        static PipelineEvent finished(String requestId, PipelineResult pr) {
            return new PipelineEvent(Type.FINISHED, requestId, null, pr, 0);
        }
    }

    public static final class Builder {
        private final List<RankStage> stages = new ArrayList<>();
        private List<Candidate> ultimateFallback;
        private Consumer<PipelineEvent> eventListener;

        public Builder addStage(RankStage stage) {
            this.stages.add(Objects.requireNonNull(stage));
            return this;
        }

        public Builder stages(List<RankStage> stages) {
            this.stages.clear();
            if (stages != null) {
                this.stages.addAll(stages);
            }
            return this;
        }

        /** Hot-list / ops fallback when entire cascade returns empty. */
        public Builder ultimateFallback(List<Candidate> items) {
            this.ultimateFallback = items;
            return this;
        }

        public Builder eventListener(Consumer<PipelineEvent> listener) {
            this.eventListener = listener;
            return this;
        }

        /**
         * Convenience: standard 5-stage cascade.
         */
        public Builder standardCascade(
                RecallStage recall,
                CoarseRankStage coarse,
                FineRankStage fine,
                RerankStage rerank,
                MixRankStage mix) {
            stages.clear();
            stages.add(recall);
            stages.add(coarse);
            stages.add(fine);
            stages.add(rerank);
            stages.add(mix);
            return this;
        }

        public PipelineOrchestrator build() {
            if (stages.isEmpty()) {
                throw new IllegalStateException("at least one stage required");
            }
            return new PipelineOrchestrator(this);
        }
    }
}
