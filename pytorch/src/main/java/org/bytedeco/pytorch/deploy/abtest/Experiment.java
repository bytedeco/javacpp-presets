/*
 * Experiment definition: layered orthogonal design as used by Meta XP,
 * ByteDance Libra, Google, Alibaba and Tencent experiment platforms.
 *
 * Key industry constraints encoded here:
 *   1. One experiment occupies one or more mutually exclusive domains (layers).
 *   2. Experiments in the SAME layer share traffic and MUST NOT overlap on the
 *      same unit (domain mutex / traffic domain).
 *   3. Experiments in DIFFERENT layers are orthogonal — each unit is hashed
 *      independently so joint exposure is product of traffic shares.
 *   4. Diversion unit must be stable for sticky assignment.
 *   5. salt/seed changes re-randomize the population (use carefully).
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable experiment specification.
 *
 * <p>Traffic model:
 * <pre>
 *   P(enter experiment) = trafficPercent / 100
 *   P(variant_i | entered) = weight_i / sum(weights)
 * </pre>
 */
public final class Experiment {

    private final String id;
    private final String name;
    private final String layerId;
    private final String owner;
    private final DiversionUnit diversionUnit;
    private final String salt;
    private final double trafficPercent;
    private final List<Variant> variants;
    private final ExperimentStatus status;
    private final Instant startTime;
    private final Instant endTime;
    private final List<String> primaryMetrics;
    private final List<String> guardrailMetrics;
    private final Map<String, String> tags;
    private final String hypothesis;
    private final long bucketCount;
    private final String description;

    private Experiment(Builder b) {
        if (b.id == null || b.id.isEmpty()) {
            throw new IllegalArgumentException("experiment id required");
        }
        if (b.layerId == null || b.layerId.isEmpty()) {
            throw new IllegalArgumentException("layerId required (layered design)");
        }
        if (b.trafficPercent < 0.0 || b.trafficPercent > 100.0) {
            throw new IllegalArgumentException("trafficPercent must be in [0, 100]");
        }
        if (b.variants == null || b.variants.isEmpty()) {
            throw new IllegalArgumentException("at least one variant required");
        }
        long controlCount = b.variants.stream().filter(Variant::isControl).count();
        if (controlCount < 1) {
            throw new IllegalArgumentException("at least one control variant required");
        }
        if (b.bucketCount < 100) {
            throw new IllegalArgumentException("bucketCount should be >= 100 for stable hashing");
        }
        double weightSum = 0.0;
        for (Variant v : b.variants) {
            weightSum += v.trafficWeight();
        }
        if (weightSum <= 0.0) {
            throw new IllegalArgumentException("sum of variant weights must be > 0");
        }

        this.id = b.id;
        this.name = b.name != null ? b.name : b.id;
        this.layerId = b.layerId;
        this.owner = b.owner != null ? b.owner : "";
        this.diversionUnit = b.diversionUnit != null ? b.diversionUnit : DiversionUnit.USER_ID;
        this.salt = b.salt != null ? b.salt : b.id;
        this.trafficPercent = b.trafficPercent;
        this.variants = Collections.unmodifiableList(new ArrayList<>(b.variants));
        this.status = b.status != null ? b.status : ExperimentStatus.DRAFT;
        this.startTime = b.startTime;
        this.endTime = b.endTime;
        this.primaryMetrics = Collections.unmodifiableList(new ArrayList<>(b.primaryMetrics));
        this.guardrailMetrics = Collections.unmodifiableList(new ArrayList<>(b.guardrailMetrics));
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        this.hypothesis = b.hypothesis != null ? b.hypothesis : "";
        this.bucketCount = b.bucketCount;
        this.description = b.description != null ? b.description : "";
    }

    public static Builder builder(String id, String layerId) {
        return new Builder(id, layerId);
    }

    public String id() {
        return id;
    }

    public String name() {
        return name;
    }

    public String layerId() {
        return layerId;
    }

    public String owner() {
        return owner;
    }

    public DiversionUnit diversionUnit() {
        return diversionUnit;
    }

    public String salt() {
        return salt;
    }

    public double trafficPercent() {
        return trafficPercent;
    }

    public List<Variant> variants() {
        return variants;
    }

    public Variant control() {
        for (Variant v : variants) {
            if (v.isControl()) {
                return v;
            }
        }
        throw new IllegalStateException("no control variant");
    }

    public Variant variant(String variantId) {
        for (Variant v : variants) {
            if (v.id().equals(variantId)) {
                return v;
            }
        }
        return null;
    }

    public ExperimentStatus status() {
        return status;
    }

    public Instant startTime() {
        return startTime;
    }

    public Instant endTime() {
        return endTime;
    }

    public List<String> primaryMetrics() {
        return primaryMetrics;
    }

    public List<String> guardrailMetrics() {
        return guardrailMetrics;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public String hypothesis() {
        return hypothesis;
    }

    public long bucketCount() {
        return bucketCount;
    }

    public String description() {
        return description;
    }

    public double totalWeight() {
        double s = 0.0;
        for (Variant v : variants) {
            s += v.trafficWeight();
        }
        return s;
    }

    public boolean isActiveAt(Instant now) {
        if (!status.acceptsTraffic()) {
            return false;
        }
        if (startTime != null && now.isBefore(startTime)) {
            return false;
        }
        if (endTime != null && !now.isBefore(endTime)) {
            return false;
        }
        return true;
    }

    public Experiment withStatus(ExperimentStatus newStatus) {
        return builder(id, layerId)
                .name(name)
                .owner(owner)
                .diversionUnit(diversionUnit)
                .salt(salt)
                .trafficPercent(trafficPercent)
                .variants(variants)
                .status(newStatus)
                .startTime(startTime)
                .endTime(endTime)
                .primaryMetrics(primaryMetrics)
                .guardrailMetrics(guardrailMetrics)
                .tags(tags)
                .hypothesis(hypothesis)
                .bucketCount(bucketCount)
                .description(description)
                .build();
    }

    public Experiment withTrafficPercent(double percent) {
        return builder(id, layerId)
                .name(name)
                .owner(owner)
                .diversionUnit(diversionUnit)
                .salt(salt)
                .trafficPercent(percent)
                .variants(variants)
                .status(status)
                .startTime(startTime)
                .endTime(endTime)
                .primaryMetrics(primaryMetrics)
                .guardrailMetrics(guardrailMetrics)
                .tags(tags)
                .hypothesis(hypothesis)
                .bucketCount(bucketCount)
                .description(description)
                .build();
    }

    @Override
    public String toString() {
        return "Experiment{id='" + id + "', layer='" + layerId + "', status=" + status
                + ", traffic=" + trafficPercent + "%}";
    }

    public static final class Builder {
        private final String id;
        private final String layerId;
        private String name;
        private String owner;
        private DiversionUnit diversionUnit = DiversionUnit.USER_ID;
        private String salt;
        private double trafficPercent = 100.0;
        private final List<Variant> variants = new ArrayList<>();
        private ExperimentStatus status = ExperimentStatus.DRAFT;
        private Instant startTime;
        private Instant endTime;
        private final List<String> primaryMetrics = new ArrayList<>();
        private final List<String> guardrailMetrics = new ArrayList<>();
        private final Map<String, String> tags = new LinkedHashMap<>();
        private String hypothesis;
        private long bucketCount = 1000L;
        private String description;

        private Builder(String id, String layerId) {
            this.id = id;
            this.layerId = layerId;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder owner(String owner) {
            this.owner = owner;
            return this;
        }

        public Builder diversionUnit(DiversionUnit diversionUnit) {
            this.diversionUnit = diversionUnit;
            return this;
        }

        public Builder salt(String salt) {
            this.salt = salt;
            return this;
        }

        public Builder trafficPercent(double trafficPercent) {
            this.trafficPercent = trafficPercent;
            return this;
        }

        public Builder addVariant(Variant variant) {
            this.variants.add(Objects.requireNonNull(variant));
            return this;
        }

        public Builder variants(List<Variant> variants) {
            this.variants.clear();
            if (variants != null) {
                this.variants.addAll(variants);
            }
            return this;
        }

        public Builder status(ExperimentStatus status) {
            this.status = status;
            return this;
        }

        public Builder startTime(Instant startTime) {
            this.startTime = startTime;
            return this;
        }

        public Builder endTime(Instant endTime) {
            this.endTime = endTime;
            return this;
        }

        public Builder primaryMetric(String metric) {
            this.primaryMetrics.add(metric);
            return this;
        }

        public Builder primaryMetrics(List<String> metrics) {
            this.primaryMetrics.clear();
            if (metrics != null) {
                this.primaryMetrics.addAll(metrics);
            }
            return this;
        }

        public Builder guardrailMetric(String metric) {
            this.guardrailMetrics.add(metric);
            return this;
        }

        public Builder guardrailMetrics(List<String> metrics) {
            this.guardrailMetrics.clear();
            if (metrics != null) {
                this.guardrailMetrics.addAll(metrics);
            }
            return this;
        }

        public Builder tag(String key, String value) {
            this.tags.put(key, value);
            return this;
        }

        public Builder tags(Map<String, String> tags) {
            this.tags.clear();
            if (tags != null) {
                this.tags.putAll(tags);
            }
            return this;
        }

        public Builder hypothesis(String hypothesis) {
            this.hypothesis = hypothesis;
            return this;
        }

        public Builder bucketCount(long bucketCount) {
            this.bucketCount = bucketCount;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Experiment build() {
            return new Experiment(this);
        }
    }
}
