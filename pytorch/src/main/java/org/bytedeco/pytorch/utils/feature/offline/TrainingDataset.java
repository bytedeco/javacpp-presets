/*
 * Training dataset produced by historical feature retrieval (PIT join output).
 */
package org.bytedeco.pytorch.utils.feature.offline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Immutable training table: entity keys + event_ts + labels + features. */
public final class TrainingDataset {

    private final List<Map<String, Object>> rows;
    private final List<String> columns;
    private final List<String> entityKeys;
    private final List<String> featureColumns;
    private final String eventTimestampColumn;
    private final String labelColumn;
    private final String schemaVersion;
    private final PointInTimeJoin.JoinStats joinStats;
    private final Map<String, String> meta;

    private TrainingDataset(Builder b) {
        this.rows = Collections.unmodifiableList(new ArrayList<>(b.rows));
        this.columns = Collections.unmodifiableList(new ArrayList<>(b.columns));
        this.entityKeys = Collections.unmodifiableList(new ArrayList<>(b.entityKeys));
        this.featureColumns = Collections.unmodifiableList(new ArrayList<>(b.featureColumns));
        this.eventTimestampColumn = b.eventTimestampColumn != null ? b.eventTimestampColumn : PointInTimeJoin.DEFAULT_EVENT_TS;
        this.labelColumn = b.labelColumn != null ? b.labelColumn : "";
        this.schemaVersion = b.schemaVersion != null ? b.schemaVersion : "v1";
        this.joinStats = b.joinStats;
        this.meta = Collections.unmodifiableMap(new LinkedHashMap<>(b.meta));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static TrainingDataset fromJoinResult(PointInTimeJoin.Result result,
                                                 List<String> entityKeys,
                                                 String labelColumn,
                                                 String schemaVersion) {
        Objects.requireNonNull(result, "result");
        List<String> featureCols = new ArrayList<>();
        for (String c : result.outputColumns) {
            if (entityKeys != null && entityKeys.contains(c)) continue;
            if (c.equals(PointInTimeJoin.DEFAULT_EVENT_TS)) continue;
            if (labelColumn != null && labelColumn.equals(c)) continue;
            featureCols.add(c);
        }
        return builder()
                .rows(result.rows)
                .columns(result.outputColumns)
                .entityKeys(entityKeys != null ? entityKeys : List.of())
                .featureColumns(featureCols)
                .labelColumn(labelColumn)
                .schemaVersion(schemaVersion)
                .joinStats(result.stats)
                .build();
    }

    public List<Map<String, Object>> rows() {
        return rows;
    }

    public int size() {
        return rows.size();
    }

    public List<String> columns() {
        return columns;
    }

    public List<String> entityKeys() {
        return entityKeys;
    }

    public List<String> featureColumns() {
        return featureColumns;
    }

    public String eventTimestampColumn() {
        return eventTimestampColumn;
    }

    public String labelColumn() {
        return labelColumn;
    }

    public String schemaVersion() {
        return schemaVersion;
    }

    public PointInTimeJoin.JoinStats joinStats() {
        return joinStats;
    }

    public Map<String, String> meta() {
        return meta;
    }

    public Map<String, Object> row(int i) {
        return rows.get(i);
    }

    /** Extract double labels; missing → 0. */
    public double[] labels() {
        if (labelColumn == null || labelColumn.isEmpty()) return new double[0];
        double[] y = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            Object v = rows.get(i).get(labelColumn);
            y[i] = v instanceof Number ? ((Number) v).doubleValue() : 0.0;
        }
        return y;
    }

    /** Dense matrix [n, f] for numeric feature columns only. */
    public double[][] denseMatrix() {
        int n = rows.size();
        int f = featureColumns.size();
        double[][] m = new double[n][f];
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = rows.get(i);
            for (int j = 0; j < f; j++) {
                Object v = row.get(featureColumns.get(j));
                m[i][j] = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
            }
        }
        return m;
    }

    @Override
    public String toString() {
        return "TrainingDataset{rows=" + rows.size()
                + ", features=" + featureColumns.size()
                + ", schema=" + schemaVersion
                + (joinStats != null ? ", " + joinStats : "")
                + "}";
    }

    public static final class Builder {
        private final List<Map<String, Object>> rows = new ArrayList<>();
        private final List<String> columns = new ArrayList<>();
        private final List<String> entityKeys = new ArrayList<>();
        private final List<String> featureColumns = new ArrayList<>();
        private String eventTimestampColumn = PointInTimeJoin.DEFAULT_EVENT_TS;
        private String labelColumn;
        private String schemaVersion = "v1";
        private PointInTimeJoin.JoinStats joinStats;
        private final Map<String, String> meta = new LinkedHashMap<>();

        public Builder rows(List<Map<String, Object>> rows) {
            this.rows.clear();
            if (rows != null) this.rows.addAll(rows);
            return this;
        }

        public Builder columns(List<String> columns) {
            this.columns.clear();
            if (columns != null) this.columns.addAll(columns);
            return this;
        }

        public Builder entityKeys(List<String> entityKeys) {
            this.entityKeys.clear();
            if (entityKeys != null) this.entityKeys.addAll(entityKeys);
            return this;
        }

        public Builder featureColumns(List<String> featureColumns) {
            this.featureColumns.clear();
            if (featureColumns != null) this.featureColumns.addAll(featureColumns);
            return this;
        }

        public Builder eventTimestampColumn(String eventTimestampColumn) {
            this.eventTimestampColumn = eventTimestampColumn;
            return this;
        }

        public Builder labelColumn(String labelColumn) {
            this.labelColumn = labelColumn;
            return this;
        }

        public Builder schemaVersion(String schemaVersion) {
            this.schemaVersion = schemaVersion;
            return this;
        }

        public Builder joinStats(PointInTimeJoin.JoinStats joinStats) {
            this.joinStats = joinStats;
            return this;
        }

        public Builder meta(String k, String v) {
            if (k != null && v != null) meta.put(k, v);
            return this;
        }

        public TrainingDataset build() {
            return new TrainingDataset(this);
        }
    }
}
