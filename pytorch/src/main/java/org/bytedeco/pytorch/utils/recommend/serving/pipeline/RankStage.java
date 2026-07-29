/*
 * Ranking stage contract + shared result types.
 */
package org.bytedeco.pytorch.utils.recommend.serving.pipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** One stage in the multi-stage ranking cascade. */
public interface RankStage {

    /** Stage name for logs / metrics (recall, coarse, fine, rerank, mix). */
    String name();

    /**
     * Execute stage.
     *
     * @param ctx     request context (deadline, exp params)
     * @param input   candidates from previous stage (empty for recall)
     * @return stage output (may be empty on degradation)
     */
    StageResult execute(RequestContext ctx, List<Candidate> input);

    /** Default no-op fallback: pass input through truncated to quota. */
    default StageResult fallback(RequestContext ctx, List<Candidate> input, Throwable cause) {
        int quota = ctx.expParamInt(name() + ".fallback_quota", Math.min(50, input.size()));
        List<Candidate> out = new ArrayList<>(input.subList(0, Math.min(quota, input.size())));
        return StageResult.degraded(name(), out, cause == null ? "fallback" : cause.getMessage());
    }

    /** Stage execution outcome. */
    final class StageResult {
        public final String stageName;
        public final List<Candidate> candidates;
        public final long latencyMs;
        public final boolean degraded;
        public final boolean timedOut;
        public final String message;

        private StageResult(
                String stageName,
                List<Candidate> candidates,
                long latencyMs,
                boolean degraded,
                boolean timedOut,
                String message) {
            this.stageName = stageName;
            this.candidates = Collections.unmodifiableList(new ArrayList<>(
                    Objects.requireNonNull(candidates, "candidates")));
            this.latencyMs = latencyMs;
            this.degraded = degraded;
            this.timedOut = timedOut;
            this.message = message != null ? message : "";
        }

        public static StageResult ok(String stage, List<Candidate> candidates, long latencyMs) {
            return new StageResult(stage, candidates, latencyMs, false, false, "ok");
        }

        public static StageResult degraded(String stage, List<Candidate> candidates, String message) {
            return new StageResult(stage, candidates, 0L, true, false, message);
        }

        public static StageResult timeout(String stage, List<Candidate> candidates, long latencyMs) {
            return new StageResult(stage, candidates, latencyMs, true, true, "timeout");
        }

        public int size() {
            return candidates.size();
        }
    }
}
