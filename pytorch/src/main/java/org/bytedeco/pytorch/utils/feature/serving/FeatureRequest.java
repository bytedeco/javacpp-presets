/*
 * Online / batch feature retrieval request (Databricks Feature Provider / Feast get_online_features).
 */
package org.bytedeco.pytorch.utils.feature.serving;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Immutable feature retrieval request. */
public final class FeatureRequest {

    private final String project;
    private final String featureService;
    private final List<String> viewNames;
    private final Map<String, Object> entities;
    private final List<Map<String, Object>> entityRows;
    private final Map<String, Object> requestContext;
    private final boolean includeOnDemand;

    private FeatureRequest(Builder b) {
        this.project = b.project != null && !b.project.isEmpty() ? b.project : "default";
        this.featureService = b.featureService != null ? b.featureService : "";
        this.viewNames = Collections.unmodifiableList(new ArrayList<>(b.viewNames));
        this.entities = Collections.unmodifiableMap(new LinkedHashMap<>(b.entities));
        this.entityRows = Collections.unmodifiableList(copyRows(b.entityRows));
        this.requestContext = Collections.unmodifiableMap(new LinkedHashMap<>(b.requestContext));
        this.includeOnDemand = b.includeOnDemand;
    }

    private static List<Map<String, Object>> copyRows(List<Map<String, Object>> rows) {
        List<Map<String, Object>> out = new ArrayList<>();
        if (rows == null) return out;
        for (Map<String, Object> r : rows) {
            out.add(Collections.unmodifiableMap(new LinkedHashMap<>(r)));
        }
        return out;
    }

    public static FeatureRequest of(String featureService) {
        return builder().featureService(featureService).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public String project() {
        return project;
    }

    public String featureService() {
        return featureService;
    }

    public List<String> viewNames() {
        return viewNames;
    }

    /** Single-entity key map (online path). */
    public Map<String, Object> entities() {
        return entities;
    }

    /** Multi-row entity keys (batch / candidate fanout). */
    public List<Map<String, Object>> entityRows() {
        return entityRows;
    }

    public Map<String, Object> requestContext() {
        return requestContext;
    }

    public boolean includeOnDemand() {
        return includeOnDemand;
    }

    /** Effective entity rows: explicit list, or single map wrapped. */
    public List<Map<String, Object>> effectiveEntityRows() {
        if (!entityRows.isEmpty()) return entityRows;
        if (!entities.isEmpty()) return List.of(entities);
        return List.of();
    }

    public Builder toBuilder() {
        return builder()
                .project(project)
                .featureService(featureService)
                .views(viewNames)
                .entities(entities)
                .entityRows(entityRows)
                .requestContext(requestContext)
                .includeOnDemand(includeOnDemand);
    }

    @Override
    public String toString() {
        return "FeatureRequest{svc=" + featureService
                + ", project=" + project
                + ", entities=" + entities.size()
                + ", rows=" + effectiveEntityRows().size()
                + "}";
    }

    public static final class Builder {
        private String project = "default";
        private String featureService;
        private final List<String> viewNames = new ArrayList<>();
        private final Map<String, Object> entities = new LinkedHashMap<>();
        private final List<Map<String, Object>> entityRows = new ArrayList<>();
        private final Map<String, Object> requestContext = new LinkedHashMap<>();
        private boolean includeOnDemand = true;

        public Builder project(String project) {
            this.project = project;
            return this;
        }

        public Builder featureService(String featureService) {
            this.featureService = featureService;
            return this;
        }

        public Builder views(String... names) {
            if (names != null) {
                for (String n : names) {
                    if (n != null && !n.isEmpty()) viewNames.add(n);
                }
            }
            return this;
        }

        public Builder views(List<String> names) {
            if (names != null) viewNames.addAll(names);
            return this;
        }

        public Builder entity(String key, Object value) {
            entities.put(key, value);
            return this;
        }

        public Builder entities(Map<String, Object> more) {
            if (more != null) entities.putAll(more);
            return this;
        }

        public Builder entityRow(Map<String, Object> row) {
            if (row != null) entityRows.add(new LinkedHashMap<>(row));
            return this;
        }

        public Builder entityRows(List<Map<String, Object>> rows) {
            if (rows != null) {
                for (Map<String, Object> r : rows) {
                    entityRows.add(new LinkedHashMap<>(r));
                }
            }
            return this;
        }

        public Builder requestContext(String key, Object value) {
            requestContext.put(key, value);
            return this;
        }

        public Builder requestContext(Map<String, Object> more) {
            if (more != null) requestContext.putAll(more);
            return this;
        }

        public Builder includeOnDemand(boolean includeOnDemand) {
            this.includeOnDemand = includeOnDemand;
            return this;
        }

        public FeatureRequest build() {
            if ((featureService == null || featureService.isEmpty()) && viewNames.isEmpty()) {
                throw new IllegalStateException("FeatureRequest requires featureService or viewNames");
            }
            return new FeatureRequest(this);
        }
    }
}
