/*
 * Mix / multi-queue merge stage.
 *
 * After organic re-rank, production feeds often merge multiple queues:
 *   - Organic recommendations
 *   - Ads / promoted items (with frequency caps)
 *   - Operations force-insert (campaigns, safety notices)
 *   - Follow-graph / social inserts
 *   - Cold-start exploration slots
 *
 * Industry names: 混排 (Alibaba/ByteDance), mix-rank, blender (Netflix),
 * feed mixer (Meta). Rules encode business constraints that pure model
 * scores cannot express.
 */
package org.bytedeco.pytorch.deploy.serving.pipeline;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Mix-rank / multi-queue blender. */
public final class MixRankStage implements RankStage {

    /** A named queue of candidates with insert policy. */
    public static final class Queue {
        public final String name;
        public final List<Candidate> candidates;
        /** Minimum gap between two inserts from this queue (positions). */
        public final int minGap;
        /** Maximum inserts from this queue into the final list. */
        public final int maxInserts;
        /** Preferred start positions (0-based); empty = greedy by score. */
        public final List<Integer> fixedPositions;

        public Queue(String name, List<Candidate> candidates, int minGap, int maxInserts) {
            this(name, candidates, minGap, maxInserts, List.of());
        }

        public Queue(
                String name,
                List<Candidate> candidates,
                int minGap,
                int maxInserts,
                List<Integer> fixedPositions) {
            this.name = Objects.requireNonNull(name, "name");
            this.candidates = new ArrayList<>(Objects.requireNonNull(candidates, "candidates"));
            this.minGap = Math.max(0, minGap);
            this.maxInserts = Math.max(0, maxInserts);
            this.fixedPositions = fixedPositions == null ? List.of() : new ArrayList<>(fixedPositions);
        }
    }

    private final int defaultPageSize;

    public MixRankStage() {
        this(20);
    }

    public MixRankStage(int defaultPageSize) {
        this.defaultPageSize = defaultPageSize;
    }

    @Override
    public String name() {
        return "mix";
    }

    /**
     * Default execute treats input as the organic queue only.
     * For multi-queue, use {@link #mix(RequestContext, List, List)}.
     */
    @Override
    public StageResult execute(RequestContext ctx, List<Candidate> input) {
        long t0 = System.currentTimeMillis();
        int pageSize = ctx.expParamInt("mix.page_size", defaultPageSize);
        Queue organic = new Queue("organic", input == null ? List.of() : input, 0, pageSize);
        List<Candidate> out = mix(ctx, List.of(organic), pageSize);
        renumber(out);
        return StageResult.ok(name(), out, System.currentTimeMillis() - t0);
    }

    /**
     * Multi-queue mix.
     *
     * <p>Algorithm (simplified production blender):
     * <ol>
     *   <li>Place fixed-position inserts first (ops force-insert).</li>
     *   <li>Fill remaining slots round-robin by queue priority order,
     *       respecting minGap and maxInserts, skipping duplicates.</li>
     *   <li>Organic queue usually has minGap=0 and high maxInserts to fill gaps.</li>
     * </ol>
     *
     * @param ctx      request context
     * @param queues   queues in priority order (index 0 = highest priority for fixed slots)
     * @param pageSize final list length
     */
    public List<Candidate> mix(RequestContext ctx, List<Queue> queues, int pageSize) {
        Objects.requireNonNull(queues, "queues");
        if (pageSize <= 0) {
            return List.of();
        }
        Candidate[] slots = new Candidate[pageSize];
        Set<String> used = new HashSet<>();
        Map<String, Integer> inserted = new LinkedHashMap<>();
        Map<String, Integer> lastPos = new LinkedHashMap<>();
        Map<String, Integer> cursor = new LinkedHashMap<>();
        for (Queue q : queues) {
            inserted.put(q.name, 0);
            lastPos.put(q.name, -q.minGap - 1);
            cursor.put(q.name, 0);
        }

        // 1) Fixed positions
        for (Queue q : queues) {
            for (int pos : q.fixedPositions) {
                if (pos < 0 || pos >= pageSize) continue;
                if (slots[pos] != null) continue;
                Candidate next = nextUnused(q, cursor, used);
                if (next == null) continue;
                if (inserted.get(q.name) >= q.maxInserts) continue;
                Candidate copy = next.copy();
                copy.tag("mix_queue", q.name);
                slots[pos] = copy;
                used.add(copy.itemId());
                inserted.merge(q.name, 1, Integer::sum);
                lastPos.put(q.name, pos);
            }
        }

        // 2) Fill remaining left-to-right; at each slot try queues in order
        for (int pos = 0; pos < pageSize; pos++) {
            if (slots[pos] != null) continue;
            boolean placed = false;
            for (Queue q : queues) {
                if (inserted.get(q.name) >= q.maxInserts) continue;
                int last = lastPos.get(q.name);
                if (pos - last <= q.minGap) continue;
                Candidate next = nextUnused(q, cursor, used);
                if (next == null) continue;
                Candidate copy = next.copy();
                copy.tag("mix_queue", q.name);
                slots[pos] = copy;
                used.add(copy.itemId());
                inserted.merge(q.name, 1, Integer::sum);
                lastPos.put(q.name, pos);
                placed = true;
                break;
            }
            if (!placed) {
                // nothing available
                break;
            }
        }

        List<Candidate> out = new ArrayList<>();
        for (Candidate c : slots) {
            if (c != null) out.add(c);
        }
        return out;
    }

    /**
     * Convenience: organic + ads with every N-th slot for ads.
     *
     * @param adInterval insert an ad every N organic positions (e.g. 5 => positions 4,9,...)
     */
    public List<Candidate> mixOrganicAndAds(
            RequestContext ctx,
            List<Candidate> organic,
            List<Candidate> ads,
            int pageSize,
            int adInterval,
            int maxAds) {
        List<Integer> adPositions = new ArrayList<>();
        if (adInterval > 0) {
            for (int p = adInterval - 1; p < pageSize && adPositions.size() < maxAds; p += adInterval) {
                adPositions.add(p);
            }
        }
        Queue adQ = new Queue("ads", ads, adInterval, maxAds, adPositions);
        Queue orgQ = new Queue("organic", organic, 0, pageSize);
        // Ads first for fixed positions, organic fills the rest.
        return mix(ctx, List.of(adQ, orgQ), pageSize);
    }

    private static Candidate nextUnused(Queue q, Map<String, Integer> cursor, Set<String> used) {
        int i = cursor.getOrDefault(q.name, 0);
        while (i < q.candidates.size()) {
            Candidate c = q.candidates.get(i);
            i++;
            cursor.put(q.name, i);
            if (!used.contains(c.itemId())) {
                return c;
            }
        }
        cursor.put(q.name, i);
        return null;
    }

    private static void renumber(List<Candidate> out) {
        for (int i = 0; i < out.size(); i++) {
            out.get(i).rank(i);
        }
    }
}
