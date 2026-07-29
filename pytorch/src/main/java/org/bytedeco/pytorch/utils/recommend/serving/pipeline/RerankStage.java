/*
 * Re-ranking stage — listwise adjustments after pointwise fine rank.
 *
 * Industry (LinkedIn, Pinterest, Alibaba, ByteDance, Netflix):
 *   - Diversity (MMR, DPP, category / author damping)
 *   - Freshness boost for new items
 *   - Business rules (filter, force demote, age-gate)
 *   - Sequential / listwise models (PRM, MIR, edge-wise transformers)
 *   - Bounce / consecutive-same-category penalties
 *
 * This implementation provides deterministic rule-based re-rankers that are
 * standard building blocks; learnable listwise models plug in via {@link ListwiseReranker}.
 */
package org.bytedeco.pytorch.utils.recommend.serving.pipeline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Re-rank stage. */
public final class RerankStage implements RankStage {

    /** Listwise re-ranker transforming an already-scored list. */
    public interface ListwiseReranker {
        List<Candidate> rerank(RequestContext ctx, List<Candidate> input);
    }

    private final List<ListwiseReranker> rerankers;
    private final int defaultQuota;

    public RerankStage(List<ListwiseReranker> rerankers) {
        this(rerankers, 50);
    }

    public RerankStage(List<ListwiseReranker> rerankers, int defaultQuota) {
        this.rerankers = new ArrayList<>(Objects.requireNonNull(rerankers, "rerankers"));
        this.defaultQuota = defaultQuota;
    }

    @Override
    public String name() {
        return "rerank";
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
        int quota = ctx.expParamInt("rerank.quota", defaultQuota);
        List<Candidate> current = new ArrayList<>(input.size());
        for (Candidate c : input) {
            current.add(c.copy());
        }
        try {
            for (ListwiseReranker r : rerankers) {
                if (ctx.deadlineExceeded()) {
                    break;
                }
                // Per-reranker enable flag: rerank.<simpleName>.enabled
                String simple = r.getClass().getSimpleName();
                if ("false".equalsIgnoreCase(ctx.expParam("rerank." + simple + ".enabled", "true"))) {
                    continue;
                }
                current = r.rerank(ctx, current);
            }
        } catch (RuntimeException ex) {
            List<Candidate> out = truncate(current, quota);
            renumber(out);
            return StageResult.degraded(name(), out, "rerank_error: " + ex.getMessage());
        }
        List<Candidate> out = truncate(current, quota);
        renumber(out);
        for (Candidate c : out) {
            c.putScore("rerank_score", c.score());
        }
        return StageResult.ok(name(), out, System.currentTimeMillis() - t0);
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

    // ---- built-in rerankers -------------------------------------------------

    /**
     * Maximal Marginal Relevance (Carbonell & Goldstein style).
     * Greedy pick maximizing λ * relevance - (1-λ) * max_sim_to_selected.
     *
     * <p>Similarity here uses optional "category" tag; same category => sim 1 else 0
     * (simple but effective production heuristic). Override via custom sim if needed.
     */
    public static ListwiseReranker mmr(double lambda) {
        return (ctx, input) -> {
            double lam = ctx.expParamDouble("rerank.mmr.lambda", lambda);
            int n = input.size();
            if (n <= 1) return new ArrayList<>(input);
            List<Candidate> remaining = new ArrayList<>(input);
            List<Candidate> selected = new ArrayList<>();
            // Normalize relevance to [0,1] roughly by rank score max.
            double maxScore = remaining.stream().mapToDouble(Candidate::score).max().orElse(1.0);
            if (maxScore <= 0) maxScore = 1.0;

            while (!remaining.isEmpty()) {
                int bestIdx = 0;
                double bestVal = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < remaining.size(); i++) {
                    Candidate c = remaining.get(i);
                    double rel = c.score() / maxScore;
                    double maxSim = 0.0;
                    for (Candidate s : selected) {
                        maxSim = Math.max(maxSim, categorySim(c, s));
                    }
                    double mmr = lam * rel - (1.0 - lam) * maxSim;
                    if (mmr > bestVal) {
                        bestVal = mmr;
                        bestIdx = i;
                    }
                }
                Candidate pick = remaining.remove(bestIdx);
                pick.score(bestVal);
                selected.add(pick);
            }
            return selected;
        };
    }

    /**
     * Sliding-window category damping: if same category appears more than
     * {@code maxPerWindow} times in the last {@code window} positions, demote.
     */
    public static ListwiseReranker categoryDamping(int window, int maxPerWindow) {
        return (ctx, input) -> {
            int w = ctx.expParamInt("rerank.category.window", window);
            int max = ctx.expParamInt("rerank.category.max_per_window", maxPerWindow);
            List<Candidate> remaining = new ArrayList<>(input);
            List<Candidate> out = new ArrayList<>();
            while (!remaining.isEmpty()) {
                int chosen = -1;
                for (int i = 0; i < remaining.size(); i++) {
                    Candidate c = remaining.get(i);
                    String cat = categoryOf(c);
                    int count = 0;
                    int from = Math.max(0, out.size() - w + 1);
                    for (int j = from; j < out.size(); j++) {
                        if (categoryOf(out.get(j)).equals(cat)) count++;
                    }
                    if (count < max) {
                        chosen = i;
                        break;
                    }
                }
                if (chosen < 0) {
                    // All violate — pick highest score anyway.
                    chosen = 0;
                    for (int i = 1; i < remaining.size(); i++) {
                        if (remaining.get(i).score() > remaining.get(chosen).score()) {
                            chosen = i;
                        }
                    }
                }
                out.add(remaining.remove(chosen));
            }
            return out;
        };
    }

    /**
     * Filter-based re-ranker: drop candidates matching predicate.
     */
    public static ListwiseReranker filter(CandidateFilter filter) {
        return (ctx, input) -> {
            List<Candidate> out = new ArrayList<>();
            for (Candidate c : input) {
                if (filter.keep(ctx, c)) {
                    out.add(c);
                }
            }
            return out;
        };
    }

    /**
     * Author / uploader consecutive penalty: break runs of same author.
     */
    public static ListwiseReranker authorSpread(String authorTagKey) {
        return (ctx, input) -> {
            List<Candidate> remaining = new ArrayList<>(input);
            List<Candidate> out = new ArrayList<>();
            String lastAuthor = null;
            while (!remaining.isEmpty()) {
                int chosen = 0;
                for (int i = 0; i < remaining.size(); i++) {
                    String a = remaining.get(i).tag(authorTagKey);
                    if (a == null) a = "";
                    if (lastAuthor == null || !lastAuthor.equals(a)) {
                        chosen = i;
                        break;
                    }
                }
                Candidate pick = remaining.remove(chosen);
                lastAuthor = pick.tag(authorTagKey);
                if (lastAuthor == null) lastAuthor = "";
                out.add(pick);
            }
            return out;
        };
    }

    /**
     * Deduplicate by item id (should already be unique, but defensive).
     */
    public static ListwiseReranker dedup() {
        return (ctx, input) -> {
            Set<String> seen = new HashSet<>();
            List<Candidate> out = new ArrayList<>();
            for (Candidate c : input) {
                if (seen.add(c.itemId())) {
                    out.add(c);
                }
            }
            return out;
        };
    }

    private static String categoryOf(Candidate c) {
        String cat = c.tag("category");
        return cat == null ? "" : cat;
    }

    private static double categorySim(Candidate a, Candidate b) {
        String ca = categoryOf(a);
        String cb = categoryOf(b);
        if (ca.isEmpty() || cb.isEmpty()) return 0.0;
        return ca.equals(cb) ? 1.0 : 0.0;
    }

    @FunctionalInterface
    public interface CandidateFilter {
        boolean keep(RequestContext ctx, Candidate candidate);
    }
}
