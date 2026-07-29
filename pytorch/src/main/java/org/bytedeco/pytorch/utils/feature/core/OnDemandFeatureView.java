/*
 * On-demand FeatureView — request-time features (Feast OnDemandFeatureView).
 * Computed at serving from request context + other view values (no materialize).
 *
 * Examples: hour_of_day, price_diff = request.price - item.avg_price,
 * cross features, real-time context normalization.
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

/** Request-time feature view with pure compute function. */
public final class OnDemandFeatureView {

    /**
     * (requestContext, sourcesByViewName) → output feature map.
     * sources values are column→Object maps already resolved online/offline.
     */
    @FunctionalInterface
    public interface ComputeFn extends BiFunction<Map<String, Object>, Map<String, Map<String, Object>>, Map<String, Object>> {}

    private final String name;
    private final String project;
    private final List<Field> schema;
    private final List<Field> requestSchema;
    private final List<String> sourceViewNames;
    private final ComputeFn compute;
    private final String description;
    private final Map<String, String> tags;

    private OnDemandFeatureView(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.project = b.project != null && !b.project.isEmpty() ? b.project : Project.DEFAULT;
        this.schema = Collections.unmodifiableList(new ArrayList<>(b.schema));
        this.requestSchema = Collections.unmodifiableList(new ArrayList<>(b.requestSchema));
        this.sourceViewNames = Collections.unmodifiableList(new ArrayList<>(b.sourceViewNames));
        this.compute = b.compute;
        this.description = b.description != null ? b.description : "";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() {
        return name;
    }

    public String project() {
        return project;
    }

    public List<Field> schema() {
        return schema;
    }

    public List<Field> requestSchema() {
        return requestSchema;
    }

    public List<String> sourceViewNames() {
        return sourceViewNames;
    }

    public ComputeFn compute() {
        return compute;
    }

    public String description() {
        return description;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public String qualifiedName() {
        return project + "/" + name;
    }

    /**
     * Execute on-demand compute. Missing compute returns empty map.
     */
    public Map<String, Object> apply(Map<String, Object> requestContext,
                                     Map<String, Map<String, Object>> sources) {
        if (compute == null) return Map.of();
        Map<String, Object> req = requestContext != null ? requestContext : Map.of();
        Map<String, Map<String, Object>> src = sources != null ? sources : Map.of();
        Map<String, Object> out = compute.apply(req, src);
        return out != null ? out : Map.of();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof OnDemandFeatureView)) return false;
        OnDemandFeatureView that = (OnDemandFeatureView) o;
        return name.equals(that.name) && project.equals(that.project);
    }

    @Override
    public int hashCode() {
        return Objects.hash(project, name);
    }

    @Override
    public String toString() {
        return "OnDemandFeatureView{" + qualifiedName() + ", outs=" + schema.size() + "}";
    }

    public static final class Builder {
        private final String name;
        private String project = Project.DEFAULT;
        private final List<Field> schema = new ArrayList<>();
        private final List<Field> requestSchema = new ArrayList<>();
        private final List<String> sourceViewNames = new ArrayList<>();
        private ComputeFn compute;
        private String description;
        private final Map<String, String> tags = new LinkedHashMap<>();

        private Builder(String name) {
            this.name = name;
        }

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder schema(Field... fields) {
            if (fields != null) schema.addAll(Arrays.asList(fields));
            return this;
        }

        public Builder schema(List<Field> fields) {
            if (fields != null) schema.addAll(fields);
            return this;
        }

        public Builder requestSchema(Field... fields) {
            if (fields != null) requestSchema.addAll(Arrays.asList(fields));
            return this;
        }

        public Builder sources(String... viewNames) {
            if (viewNames != null) sourceViewNames.addAll(Arrays.asList(viewNames));
            return this;
        }

        public Builder compute(ComputeFn compute) {
            this.compute = compute;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder tag(String k, String v) {
            if (k != null && v != null) tags.put(k, v);
            return this;
        }

        public OnDemandFeatureView build() {
            return new OnDemandFeatureView(this);
        }
    }
}
