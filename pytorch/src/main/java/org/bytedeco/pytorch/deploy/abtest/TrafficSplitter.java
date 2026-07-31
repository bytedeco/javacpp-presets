/*
 * Traffic split utilities used both by experiment diversion and by gateway
 * canary / blue-green percentage routing.
 *
 * Distinction:
 *   - BucketAssigner: sticky unit-level experiment diversion (salted hash)
 *   - TrafficSplitter: percentage / weight routing for deployment & gateway
 *     (may be sticky or per-request depending on mode)
 *
 * Industry:
 *   - Envoy / Istio / Kubernetes Gateway: weight-based clusters
 *   - Netflix Zuul / Spring Cloud Gateway: predicate + weight
 *   - Alibaba MSE / Tencent TSF: canary rules by header / percentage / user
 *   - Meta / Google internal L7: consistent hash + percentage ramp
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Weighted / percentage traffic splitter.
 */
public final class TrafficSplitter {

    private TrafficSplitter() {}

    /** A named backend / variant with relative weight. */
    public static final class WeightedTarget {
        public final String id;
        public final double weight;

        public WeightedTarget(String id, double weight) {
            if (id == null || id.isEmpty()) {
                throw new IllegalArgumentException("id required");
            }
            if (weight < 0.0) {
                throw new IllegalArgumentException("weight must be >= 0");
            }
            this.id = id;
            this.weight = weight;
        }

        @Override
        public String toString() {
            return id + "=" + weight;
        }
    }

    /**
     * Sticky weighted choice: same key always maps to same target
     * (consistent with canary user stickiness).
     */
    public static String selectSticky(String key, String salt, List<WeightedTarget> targets) {
        Objects.requireNonNull(targets, "targets");
        if (targets.isEmpty()) {
            throw new IllegalArgumentException("targets empty");
        }
        double total = 0.0;
        for (WeightedTarget t : targets) {
            total += t.weight;
        }
        if (total <= 0.0) {
            throw new IllegalArgumentException("total weight must be > 0");
        }
        String material = (salt == null ? "" : salt) + "\0" + (key == null ? "" : key);
        int h = BucketAssigner.murmur3_32(material.getBytes(StandardCharsets.UTF_8), BucketAssigner.DEFAULT_SEED);
        double u = (h & 0xffffffffL) / (double) 0x100000000L;
        double acc = 0.0;
        for (WeightedTarget t : targets) {
            acc += t.weight / total;
            if (u < acc) {
                return t.id;
            }
        }
        return targets.get(targets.size() - 1).id;
    }

    /**
     * Non-sticky (per-call random) weighted choice — for pure load share.
     */
    public static String selectRandom(List<WeightedTarget> targets) {
        Objects.requireNonNull(targets, "targets");
        if (targets.isEmpty()) {
            throw new IllegalArgumentException("targets empty");
        }
        double total = 0.0;
        for (WeightedTarget t : targets) {
            total += t.weight;
        }
        if (total <= 0.0) {
            throw new IllegalArgumentException("total weight must be > 0");
        }
        double u = ThreadLocalRandom.current().nextDouble() * total;
        double acc = 0.0;
        for (WeightedTarget t : targets) {
            acc += t.weight;
            if (u < acc) {
                return t.id;
            }
        }
        return targets.get(targets.size() - 1).id;
    }

    /**
     * Binary percentage split: return {@code treatmentId} with probability
     * {@code percent/100}, else {@code controlId}. Sticky on key.
     */
    public static String selectByPercent(
            String key, String salt, double percent, String controlId, String treatmentId) {
        if (percent <= 0.0) return controlId;
        if (percent >= 100.0) return treatmentId;
        List<WeightedTarget> targets = new ArrayList<>(2);
        targets.add(new WeightedTarget(controlId, 100.0 - percent));
        targets.add(new WeightedTarget(treatmentId, percent));
        return selectSticky(key, salt, targets);
    }

    /**
     * Build equal-weight targets from ids.
     */
    public static List<WeightedTarget> equalWeights(String... ids) {
        List<WeightedTarget> list = new ArrayList<>();
        for (String id : ids) {
            list.add(new WeightedTarget(id, 1.0));
        }
        return Collections.unmodifiableList(list);
    }

    /**
     * Normalize raw weights into percentages that sum to 100.
     */
    public static List<WeightedTarget> normalizeToPercent(List<WeightedTarget> targets) {
        double total = 0.0;
        for (WeightedTarget t : targets) {
            total += t.weight;
        }
        if (total <= 0.0) {
            throw new IllegalArgumentException("total weight must be > 0");
        }
        List<WeightedTarget> out = new ArrayList<>(targets.size());
        for (WeightedTarget t : targets) {
            out.add(new WeightedTarget(t.id, t.weight / total * 100.0));
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Canary ramp schedule helper: given ramp steps (percent at each stage),
     * return the percent for stage index.
     *
     * <p>Common Meta / Google / Alibaba canary steps: 1% -> 5% -> 10% -> 25% -> 50% -> 100%.
     */
    public static double canaryPercent(int stageIndex, double... stages) {
        if (stages == null || stages.length == 0) {
            throw new IllegalArgumentException("stages required");
        }
        if (stageIndex < 0) {
            return stages[0];
        }
        if (stageIndex >= stages.length) {
            return stages[stages.length - 1];
        }
        return stages[stageIndex];
    }

    /** Default industry canary ramp. */
    public static double[] defaultCanaryStages() {
        return new double[] {1.0, 5.0, 10.0, 25.0, 50.0, 100.0};
    }
}
