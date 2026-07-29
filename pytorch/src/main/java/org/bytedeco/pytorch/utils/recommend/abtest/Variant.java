/*
 * A single treatment / control arm inside an experiment.
 *
 * Mirrors:
 *   - Meta XP: experiment arm + parameter bag
 *   - ByteDance Libra: variant + config payload
 *   - Google: experiment variant with parameter overrides
 *   - Alibaba: bucket group with strategy config
 */
package org.bytedeco.pytorch.utils.recommend.abtest;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable experiment variant (control or treatment).
 *
 * <p>{@code trafficWeight} is relative within the experiment. Absolute
 * exposure percentage is {@code experiment.trafficPercent * weight / sum(weights)}.
 */
public final class Variant {

    private final String id;
    private final String name;
    private final boolean control;
    private final double trafficWeight;
    private final Map<String, String> parameters;
    private final String description;

    private Variant(Builder b) {
        if (b.id == null || b.id.isEmpty()) {
            throw new IllegalArgumentException("variant id required");
        }
        if (b.trafficWeight < 0.0) {
            throw new IllegalArgumentException("trafficWeight must be >= 0");
        }
        this.id = b.id;
        this.name = b.name != null ? b.name : b.id;
        this.control = b.control;
        this.trafficWeight = b.trafficWeight;
        this.parameters = Collections.unmodifiableMap(new LinkedHashMap<>(b.parameters));
        this.description = b.description != null ? b.description : "";
    }

    public static Builder builder(String id) {
        return new Builder(id);
    }

    public static Variant control(String id, double weight) {
        return builder(id).name("control").control(true).trafficWeight(weight).build();
    }

    public static Variant treatment(String id, double weight) {
        return builder(id).control(false).trafficWeight(weight).build();
    }

    public String id() {
        return id;
    }

    public String name() {
        return name;
    }

    public boolean isControl() {
        return control;
    }

    public double trafficWeight() {
        return trafficWeight;
    }

    public Map<String, String> parameters() {
        return parameters;
    }

    public String parameter(String key, String defaultValue) {
        String v = parameters.get(key);
        return v != null ? v : defaultValue;
    }

    public String description() {
        return description;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Variant)) return false;
        Variant variant = (Variant) o;
        return id.equals(variant.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return "Variant{id='" + id + "', control=" + control + ", weight=" + trafficWeight + "}";
    }

    public static final class Builder {
        private final String id;
        private String name;
        private boolean control;
        private double trafficWeight = 1.0;
        private final Map<String, String> parameters = new LinkedHashMap<>();
        private String description;

        private Builder(String id) {
            this.id = id;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder control(boolean control) {
            this.control = control;
            return this;
        }

        public Builder trafficWeight(double trafficWeight) {
            this.trafficWeight = trafficWeight;
            return this;
        }

        public Builder parameter(String key, String value) {
            this.parameters.put(Objects.requireNonNull(key), Objects.requireNonNull(value));
            return this;
        }

        public Builder parameters(Map<String, String> params) {
            if (params != null) {
                this.parameters.putAll(params);
            }
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Variant build() {
            return new Variant(this);
        }
    }
}
