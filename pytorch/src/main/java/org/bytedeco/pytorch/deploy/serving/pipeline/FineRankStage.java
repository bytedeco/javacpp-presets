/*
 * Fine ranking stage — heavy CTR / CVR / multi-task model scoring.
 *
 * Industry (Alibaba 精排, ByteDance ranker, YouTube ranking, Meta feed ranker):
 *   - DeepFM / DIN / DIEN / DCN / SIM / multi-task ESMM etc.
 *   - Batch inference on GPU / CPU with feature store lookups
 *   - Score fusion: pCTR * pCVR^a * price^b * ...
 *   - Typical input size: 50–500 candidates; latency 20–80ms
 *
 * Scorer is pluggable so existing recommend models (DeepFM, DIN, ...) can be
 * wrapped without this package depending on Module loading details.
 */
package org.bytedeco.pytorch.deploy.serving.pipeline;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Fine rank stage. */
public final class FineRankStage implements RankStage {

    /**
     * Batch scorer interface — production systems score a batch together for
     * GPU efficiency rather than one-by-one.
     */
    public interface FineScorer {
        /**
         * Score all candidates; return array of scores aligned with input order.
         */
        double[] scoreBatch(RequestContext ctx, List<Candidate> candidates) throws Exception;
    }

    /** Optional multi-task fusion after raw model heads. */
    public interface ScoreFusion {
        double fuse(RequestContext ctx, Candidate candidate, double[] headScores);
    }

    private final FineScorer scorer;
    private final ScoreFusion fusion;
    private final int defaultQuota;
    private final long defaultTimeoutMs;

    public FineRankStage(FineScorer scorer) {
        this(scorer, null, 100, 80L);
    }

    public FineRankStage(FineScorer scorer, ScoreFusion fusion, int defaultQuota, long defaultTimeoutMs) {
        this.scorer = Objects.requireNonNull(scorer, "scorer");
        this.fusion = fusion;
        this.defaultQuota = defaultQuota;
        this.defaultTimeoutMs = defaultTimeoutMs;
    }

    @Override
    public String name() {
        return "fine";
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
        int quota = ctx.expParamInt("fine.quota", defaultQuota);
        long timeout = ctx.expParamInt("fine.timeout_ms", (int) defaultTimeoutMs);
        long hardDeadline = Math.min(ctx.deadlineEpochMs(), t0 + timeout);

        List<Candidate> work = new ArrayList<>(input.size());
        for (Candidate c : input) {
            work.add(c.copy());
        }

        double[] raw;
        try {
            raw = scorer.scoreBatch(ctx, work);
        } catch (Exception ex) {
            // Degrade to coarse scores.
            for (Candidate c : work) {
                double s = c.getScore("coarse_score", c.score());
                c.score(s);
                c.putScore("fine_score", s);
            }
            work.sort((a, b) -> Double.compare(b.score(), a.score()));
            List<Candidate> out = truncate(work, quota);
            renumber(out);
            return StageResult.degraded(name(), out, "scorer_error: " + ex.getMessage());
        }
        if (raw == null || raw.length != work.size()) {
            return StageResult.degraded(name(), truncate(work, quota), "score_size_mismatch");
        }
        if (System.currentTimeMillis() >= hardDeadline) {
            // Still apply scores we have; mark timeout.
            applyScores(ctx, work, raw);
            work.sort((a, b) -> Double.compare(b.score(), a.score()));
            List<Candidate> out = truncate(work, quota);
            renumber(out);
            return StageResult.timeout(name(), out, System.currentTimeMillis() - t0);
        }
        applyScores(ctx, work, raw);
        work.sort((a, b) -> Double.compare(b.score(), a.score()));
        List<Candidate> out = truncate(work, quota);
        renumber(out);
        return StageResult.ok(name(), out, System.currentTimeMillis() - t0);
    }

    private void applyScores(RequestContext ctx, List<Candidate> work, double[] raw) {
        for (int i = 0; i < work.size(); i++) {
            Candidate c = work.get(i);
            double s = raw[i];
            c.putScore("fine_raw", s);
            if (fusion != null) {
                s = fusion.fuse(ctx, c, new double[] {s});
            }
            // Optional multiplicative boosts from experiment params.
            double boost = ctx.expParamDouble("fine.score_boost", 1.0);
            s *= boost;
            c.score(s);
            c.putScore("fine_score", s);
        }
    }

    private static void renumber(List<Candidate> out) {
        for (int i = 0; i < out.size(); i++) {
            out.get(i).rank(i);
        }
    }

    private static List<Candidate> truncate(List<Candidate> list, int quota) {
        if (list.size() <= quota) return new ArrayList<>(list);
        return new ArrayList<>(list.subList(0, quota));
    }

    /** Simple pCTR * weight fusion with optional pCVR from candidate scores map. */
    public static ScoreFusion eCpmLike(double cvrExponent, double priceExponent) {
        return (ctx, c, heads) -> {
            double pctr = heads.length > 0 ? heads[0] : c.score();
            double pcvr = c.getScore("pcvr", 1.0);
            double price = c.getScore("price", 1.0);
            return pctr * Math.pow(Math.max(pcvr, 1e-9), cvrExponent)
                    * Math.pow(Math.max(price, 1e-9), priceExponent);
        };
    }

    /** Identity fusion. */
    public static ScoreFusion identity() {
        return (ctx, c, heads) -> heads.length > 0 ? heads[0] : c.score();
    }

    /** Wrap a per-item function as batch scorer (convenient but slower). */
    public static FineScorer fromSingle(SingleScorer single) {
        return (ctx, candidates) -> {
            double[] out = new double[candidates.size()];
            for (int i = 0; i < candidates.size(); i++) {
                out[i] = single.score(ctx, candidates.get(i));
            }
            return out;
        };
    }

    @FunctionalInterface
    public interface SingleScorer {
        double score(RequestContext ctx, Candidate candidate) throws Exception;
    }
}
