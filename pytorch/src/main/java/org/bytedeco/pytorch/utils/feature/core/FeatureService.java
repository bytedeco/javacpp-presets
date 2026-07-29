/*
 * FeatureService — named bundle of features for one model / use-case
 * (Feast FeatureService / Databricks feature serving group).
 *
 * Example: ranker_v3_features = user_stats.* + item_stats.* + on_demand.context
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Serving group selecting views / feature refs. */
public final class FeatureService {

    private final String name;
    private final String project;
    private final List<String> viewNames;
    private final List<FeatureRef> features;
    private final List<String> onDemandViewNames;
    private final String description;
    private final String owner;
    private final Map<String, String> tags;
    private final long createdAtMs;

    private FeatureService(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        if (this.name.isEmpty()) throw new IllegalArgumentException("feature service name empty");
        this.project = b.project != null && !b.project.isEmpty() ? b.project : Project.DEFAULT;
        this.viewNames = Collections.unmodifiableList(new ArrayList<>(b.viewNames));
        this.features = Collections.unmodifiableList(new ArrayList<>(b.features));
        this.onDemandViewNames = Collections.unmodifiableList(new ArrayList<>(b.onDemandViewNames));
        this.description = b.description != null ? b.description : "";
        this.owner = b.owner != null ? b.owner : "";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        this.createdAtMs = b.createdAtMs > 0 ? b.createdAtMs : System.currentTimeMillis();
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

    /** Batch / stream views included wholesale. */
    public List<String> viewNames() {
        return viewNames;
    }

    /** Optional fine-grained feature pins (overrides full-view when non-empty for that view). */
    public List<FeatureRef> features() {
        return features;
    }

    public List<String> onDemandViewNames() {
        return onDemandViewNames;
    }

    public String description() {
        return description;
    }

    public String owner() {
        return owner;
    }

    public Map<String, String> tags() {
        return tags;
    }

    public long createdAtMs() {
        return createdAtMs;
    }

    public String qualifiedName() {
        return project + "/" + name;
    }

    /** All view names referenced (batch + on-demand + from FeatureRefs). */
    public Set<String> allViewNames() {
        Set<String> out = new LinkedHashSet<>(viewNames);
        out.addAll(onDemandViewNames);
        for (FeatureRef ref : features) {
            if (ref.viewName() != null && !ref.viewName().isEmpty()) {
                out.add(ref.viewName());
            }
        }
        return out;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FeatureService)) return false;
        FeatureService that = (FeatureService) o;
        return name.equals(that.name) && project.equals(that.project);
    }

    @Override
    public int hashCode() {
        return Objects.hash(project, name);
    }

    @Override
    public String toString() {
        return "FeatureService{" + qualifiedName()
                + ", views=" + viewNames.size()
                + ", onDemand=" + onDemandViewNames.size()
                + ", refs=" + features.size()
                + "}";
    }

    public static final class Builder {
        private final String name;
        private String project = Project.DEFAULT;
        private final List<String> viewNames = new ArrayList<>();
        private final List<FeatureRef> features = new ArrayList<>();
        private final List<String> onDemandViewNames = new ArrayList<>();
        private String description;
        private String owner;
        private final Map<String, String> tags = new LinkedHashMap<>();
        private long createdAtMs;

        private Builder(String name) {
            this.name = name;
        }

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder views(String... names) {
            if (names != null) viewNames.addAll(Arrays.asList(names));
            return this;
        }

        public Builder views(List<String> names) {
            if (names != null) viewNames.addAll(names);
            return this;
        }

        public Builder view(String name) {
            if (name != null && !name.isEmpty()) viewNames.add(name);
            return this;
        }

        public Builder features(FeatureRef... refs) {
            if (refs != null) features.addAll(Arrays.asList(refs));
            return this;
        }

        public Builder features(List<FeatureRef> refs) {
            if (refs != null) features.addAll(refs);
            return this;
        }

        public Builder onDemandViews(String... names) {
            if (names != null) onDemandViewNames.addAll(Arrays.asList(names));
            return this;
        }

        public Builder onDemandView(String name) {
            if (name != null && !name.isEmpty()) onDemandViewNames.add(name);
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder owner(String owner) {
            this.owner = owner;
            return this;
        }

        public Builder tag(String k, String v) {
            if (k != null && v != null) tags.put(k, v);
            return this;
        }

        public Builder tags(Map<String, String> more) {
            if (more != null) tags.putAll(more);
            return this;
        }

        public Builder createdAtMs(long createdAtMs) {
            this.createdAtMs = createdAtMs;
            return this;
        }

        public FeatureService build() {
            return new FeatureService(this);
        }
    }
}
