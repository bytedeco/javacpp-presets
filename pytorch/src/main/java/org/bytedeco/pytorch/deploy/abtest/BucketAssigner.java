/*
 * Deterministic bucket assignment via salted hash.
 *
 * Industry standard (Meta, Google, ByteDance, Alibaba, Tencent):
 *   bucket = hash(salt + unitId) % bucketCount
 *
 * Properties required in production:
 *   1. Deterministic — same (salt, unit) always maps to same bucket.
 *   2. Uniform — buckets roughly equal size (chi-square SRM check).
 *   3. Independent across layers — different salts break correlation.
 *   4. Sticky — unit stays in same variant for experiment lifetime.
 *
 * Hash choice: MurmurHash3 32-bit is widely used (Guava, Alibaba, many
 * open-source AB libs). We implement a pure-Java Murmur3_32 so this module
 * has zero extra dependency.
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;

/**
 * Assigns diversion units into experiment buckets and variants.
 *
 * <p>Two-stage diversion (common at ByteDance / Alibaba):
 * <ol>
 *   <li>Layer / experiment entry: hash decides whether unit enters the
 *       experiment traffic window {@code [0, trafficPercent)}.</li>
 *   <li>Variant split: remaining hash range is partitioned by relative weights.</li>
 * </ol>
 */
public final class BucketAssigner {

    /** Default seed used when experiment salt is empty. */
    public static final int DEFAULT_SEED = 0x9747b28c;

    private BucketAssigner() {}

    /**
     * Assign a unit to a variant of the given experiment, or empty if not in traffic.
     *
     * @param experiment active experiment
     * @param unitId     diversion key (userId / deviceId / ...)
     * @return assignment, or {@code null} if unit is outside experiment traffic
     */
    public static Assignment assign(Experiment experiment, String unitId) {
        return assign(experiment, unitId, System.currentTimeMillis());
    }

    /**
     * Assign with explicit evaluation time (for replay / backfill).
     */
    public static Assignment assign(Experiment experiment, String unitId, long nowEpochMs) {
        Objects.requireNonNull(experiment, "experiment");
        Objects.requireNonNull(unitId, "unitId");
        if (unitId.isEmpty()) {
            throw new IllegalArgumentException("unitId must not be empty");
        }

        if (!experiment.status().acceptsTraffic()) {
            return null;
        }
        if (experiment.startTime() != null
                && nowEpochMs < experiment.startTime().toEpochMilli()) {
            return null;
        }
        if (experiment.endTime() != null
                && nowEpochMs >= experiment.endTime().toEpochMilli()) {
            return null;
        }

        long bucket = bucketOf(experiment.salt(), unitId, experiment.bucketCount());
        // Map bucket into [0, 100) percentage space.
        double pct = (bucket * 100.0) / (double) experiment.bucketCount();
        if (pct >= experiment.trafficPercent()) {
            return null; // not in experiment traffic
        }

        Variant variant = pickVariant(experiment.variants(), experiment.salt(), unitId);
        return new Assignment(
                experiment.id(),
                experiment.layerId(),
                variant.id(),
                variant.isControl(),
                bucket,
                unitId,
                experiment.diversionUnit(),
                nowEpochMs);
    }

    /**
     * Compute bucket index in {@code [0, bucketCount)}.
     */
    public static long bucketOf(String salt, String unitId, long bucketCount) {
        if (bucketCount <= 0) {
            throw new IllegalArgumentException("bucketCount must be > 0");
        }
        String key = (salt == null ? "" : salt) + "\0" + unitId;
        int h = murmur3_32(key.getBytes(StandardCharsets.UTF_8), DEFAULT_SEED);
        // Convert signed int to non-negative long then mod.
        long unsigned = h & 0xffffffffL;
        return unsigned % bucketCount;
    }

    /**
     * Pick variant by re-hashing with a variant-specific salt so entry and
     * variant split are not perfectly correlated (standard practice).
     */
    public static Variant pickVariant(List<Variant> variants, String salt, String unitId) {
        double total = 0.0;
        for (Variant v : variants) {
            total += v.trafficWeight();
        }
        if (total <= 0.0) {
            throw new IllegalArgumentException("variant weight sum must be > 0");
        }
        // Independent hash for variant selection.
        String key = (salt == null ? "" : salt) + ":variant\0" + unitId;
        int h = murmur3_32(key.getBytes(StandardCharsets.UTF_8), DEFAULT_SEED ^ 0x85ebca6b);
        double u = (h & 0xffffffffL) / (double) 0x100000000L; // [0, 1)
        double acc = 0.0;
        for (Variant v : variants) {
            acc += v.trafficWeight() / total;
            if (u < acc) {
                return v;
            }
        }
        return variants.get(variants.size() - 1);
    }

    /**
     * Whether two experiments in the same layer would conflict on traffic
     * windows. Used by {@link LayeredExperimentManager} for mutex checks.
     *
     * <p>Simplified model: experiments in the same layer whose traffic windows
     * would overlap in bucket space are conflicts. Production systems often
     * allocate explicit bucket ranges; here we only check capacity.
     */
    public static boolean layerCapacityExceeded(double usedPercent, double newPercent) {
        return usedPercent + newPercent > 100.0 + 1e-9;
    }

    // ---- MurmurHash3 32-bit (public domain algorithm by Austin Appleby) ----

    /**
     * MurmurHash3_x86_32.
     *
     * @param data input bytes
     * @param seed seed
     * @return 32-bit hash (may be negative as Java int)
     */
    public static int murmur3_32(byte[] data, int seed) {
        final int c1 = 0xcc9e2d51;
        final int c2 = 0x1b873593;
        final int len = data.length;
        int h1 = seed;
        int i = 0;

        while (i + 4 <= len) {
            int k1 = (data[i] & 0xff)
                    | ((data[i + 1] & 0xff) << 8)
                    | ((data[i + 2] & 0xff) << 16)
                    | ((data[i + 3] & 0xff) << 24);
            i += 4;
            k1 *= c1;
            k1 = Integer.rotateLeft(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            h1 = Integer.rotateLeft(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
        }

        int k1 = 0;
        switch (len - i) {
            case 3:
                k1 ^= (data[i + 2] & 0xff) << 16;
                // fall through
            case 2:
                k1 ^= (data[i + 1] & 0xff) << 8;
                // fall through
            case 1:
                k1 ^= (data[i] & 0xff);
                k1 *= c1;
                k1 = Integer.rotateLeft(k1, 15);
                k1 *= c2;
                h1 ^= k1;
                break;
            default:
                break;
        }

        h1 ^= len;
        // fmix32
        h1 ^= h1 >>> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >>> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >>> 16;
        return h1;
    }

    /**
     * Result of a single diversion assignment.
     */
    public static final class Assignment {
        private final String experimentId;
        private final String layerId;
        private final String variantId;
        private final boolean control;
        private final long bucket;
        private final String unitId;
        private final DiversionUnit diversionUnit;
        private final long assignedAtEpochMs;

        public Assignment(
                String experimentId,
                String layerId,
                String variantId,
                boolean control,
                long bucket,
                String unitId,
                DiversionUnit diversionUnit,
                long assignedAtEpochMs) {
            this.experimentId = experimentId;
            this.layerId = layerId;
            this.variantId = variantId;
            this.control = control;
            this.bucket = bucket;
            this.unitId = unitId;
            this.diversionUnit = diversionUnit;
            this.assignedAtEpochMs = assignedAtEpochMs;
        }

        public String experimentId() {
            return experimentId;
        }

        public String layerId() {
            return layerId;
        }

        public String variantId() {
            return variantId;
        }

        public boolean isControl() {
            return control;
        }

        public long bucket() {
            return bucket;
        }

        public String unitId() {
            return unitId;
        }

        public DiversionUnit diversionUnit() {
            return diversionUnit;
        }

        public long assignedAtEpochMs() {
            return assignedAtEpochMs;
        }

        @Override
        public String toString() {
            return "Assignment{exp=" + experimentId + ", variant=" + variantId
                    + ", bucket=" + bucket + ", unit=" + unitId + "}";
        }
    }
}
