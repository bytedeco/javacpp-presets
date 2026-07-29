/*
 * Coarse ranking (pre-rank) stage.
 *
 * After multi-channel recall produces thousands of candidates, a cheap model
 * or formula cuts them to a few hundred before the expensive fine-rank model.
 *
 * Industry (Alibaba "粗排", ByteDance pre-rank, YouTube candidate ranking):
 *   - Logistic / small MLP / gradient-boosted trees / distilled student model
 *   - Vector dot-product with user embedding (two-tower inner product)
 *   - Hand-crafted formula blending recall score, quality, freshness
 *   - Strict latency budget (typically 5–20ms)
 */
package org.bytedeco.pytorch.utils.recommend.serving.pipeline;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.ToDoubleBiFunction;

/** Coarse rank / pre-rank stage. */
public final class CoarseRankStage implements RankStage {

    /** Scorer: (context, candidate) -> coarse score. */
    public interface CoarseScorer {
        double score(RequestContext ctx, Candidate candidate);
    }

    private final CoarseScorer scorer;
    private final int defaultQuota;
    private final long defaultTimeoutMs;

    public CoarseRankStage(CoarseScorer scorer) {
        this(scorer, 300, 20L);
    }

    public CoarseRankStage(CoarseScorer scorer, int defaultQuota, long defaultTimeoutMs) {
        this.scorer = Objects.requireNonNull(scorer, "scorer");
        this.defaultQuota = defaultQuota;
        this.defaultTimeoutMs = defaultTimeoutMs;
    }

    @Override
    public String name() {
        return "coarse";
    }

    @Override
    public StageResult execute(RequestContext ctx, List<Candidate> input) {
        long t0 = System.currentTimeMillis();
        if (input == null || input.isEmpty()) {
            return StageResult.ok(name(), List.of(), 0L);
        }
        if (ctx.deadlineExceeded()) {
            return StageResult.timeout(name(), truncate(input, defaultQuota), 0L);
        }
        int quota = ctx.expParamInt("coarse.quota", defaultQuota);
        long timeout = ctx.expParamInt("coarse.timeout_ms", (int) defaultTimeoutMs);
        long hardDeadline = Math.min(ctx.deadlineEpochMs(), t0 + timeout);

        List<Candidate> scored = new ArrayList<>(input.size());
        boolean timedOut = false;
        for (Candidate c : input) {
            if (System.currentTimeMillis() >= hardDeadline) {
                timedOut = true;
                break;
            }
            Candidate copy = c.copy();
            double s;
            try {
                s = scorer.score(ctx, copy);
            } catch (RuntimeException ex) {
                // On scorer error keep recall score.
                s = copy.getScore("recall_score", copy.score());
            }
            copy.score(s);
            copy.putScore("coarse_score", s);
            scored.add(copy);
        }
        // If timed out mid-way, append remaining with recall score so we don't drop them entirely.
        if (timedOut && scored.size() < input.size()) {
            for (int i = scored.size(); i < input.size(); i++) {
                Candidate copy = input.get(i).copy();
                double s = copy.getScore("recall_score", copy.score());
                copy.score(s);
                copy.putScore("coarse_score", s);
                scored.add(copy);
            }
        }
        scored.sort((a, b) -> Double.compare(b.score(), a.score()));
        List<Candidate> out = truncate(scored, quota);
        for (int i = 0; i < out.size(); i++) {
            out.get(i).rank(i);
        }
        long latency = System.currentTimeMillis() - t0;
        if (timedOut) {
            return StageResult.timeout(name(), out, latency);
        }
        return StageResult.ok(name(), out, latency);
    }

    private static List<Candidate> truncate(List<Candidate> list, int quota) {
        if (list.size() <= quota) {
            return new ArrayList<>(list);
        }
        return new ArrayList<>(list.subList(0, quota));
    }

    /** Formula scorer: weighted sum of named sub-scores already on the candidate. */
    public static CoarseScorer weightedFormula(String[] scoreKeys, double[] weights) {
        if (scoreKeys.length != weights.length) {
            throw new IllegalArgumentException("keys/weights length mismatch");
        }
        return (ctx, c) -> {
            double s = 0.0;
            for (int i = 0; i < scoreKeys.length; i++) {
                s += weights[i] * c.getScore(scoreKeys[i], 0.0);
            }
            return s;
        };
    }

    /** Pass-through: use existing score (e.g. recall score) only. */
    public static CoarseScorer passThrough() {
        return (ctx, c) -> c.score();
    }

    /** Adapt a bi-function. */
    public static CoarseScorer of(ToDoubleBiFunction<RequestContext, Candidate> fn) {
        return fn::applyAsDouble;
    }
}
