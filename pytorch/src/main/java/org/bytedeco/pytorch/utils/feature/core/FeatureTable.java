/*
 * Physical / logical data source binding for a FeatureView.
 * Feast: DataSource (FileSource, KafkaSource, …); Feathub: source descriptor.
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Batch or stream source descriptor (path, topic, table). */
public final class FeatureTable {

    public enum SourceType {
        FILE,
        PARQUET,
        CSV,
        JSONL,
        DUCKDB,
        LANCE,
        KAFKA,
        MEMORY,
        CUSTOM
    }

    private final String name;
    private final SourceType sourceType;
    private final String uri;
    private final String timestampColumn;
    private final String createdTimestampColumn;
    private final String datePartitionColumn;
    private final Map<String, String> options;

    private FeatureTable(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.sourceType = b.sourceType != null ? b.sourceType : SourceType.FILE;
        this.uri = b.uri != null ? b.uri : "";
        this.timestampColumn = b.timestampColumn != null ? b.timestampColumn : "event_timestamp";
        this.createdTimestampColumn = b.createdTimestampColumn != null ? b.createdTimestampColumn : "";
        this.datePartitionColumn = b.datePartitionColumn != null ? b.datePartitionColumn : "";
        this.options = Collections.unmodifiableMap(new LinkedHashMap<>(b.options));
    }

    public static FeatureTable file(String name, String uri) {
        return builder(name).sourceType(SourceType.FILE).uri(uri).build();
    }

    public static FeatureTable parquet(String name, String uri) {
        return builder(name).sourceType(SourceType.PARQUET).uri(uri).build();
    }

    public static FeatureTable kafka(String name, String topic) {
        return builder(name).sourceType(SourceType.KAFKA).uri(topic).build();
    }

    public static FeatureTable memory(String name) {
        return builder(name).sourceType(SourceType.MEMORY).uri("memory://" + name).build();
    }

    public static FeatureTable duckdb(String name, String sqlOrPath) {
        return builder(name).sourceType(SourceType.DUCKDB).uri(sqlOrPath).build();
    }

    public static FeatureTable lance(String name, String uri) {
        return builder(name).sourceType(SourceType.LANCE).uri(uri).build();
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() {
        return name;
    }

    public SourceType sourceType() {
        return sourceType;
    }

    public String uri() {
        return uri;
    }

    public String timestampColumn() {
        return timestampColumn;
    }

    public String createdTimestampColumn() {
        return createdTimestampColumn;
    }

    public String datePartitionColumn() {
        return datePartitionColumn;
    }

    public Map<String, String> options() {
        return options;
    }

    public String option(String key, String defaultValue) {
        return options.getOrDefault(key, defaultValue);
    }

    @Override
    public String toString() {
        return "FeatureTable{" + name + "," + sourceType + "," + uri + "}";
    }

    public static final class Builder {
        private final String name;
        private SourceType sourceType = SourceType.FILE;
        private String uri;
        private String timestampColumn = "event_timestamp";
        private String createdTimestampColumn;
        private String datePartitionColumn;
        private final Map<String, String> options = new LinkedHashMap<>();

        private Builder(String name) {
            this.name = name;
        }

        public Builder sourceType(SourceType sourceType) {
            this.sourceType = sourceType;
            return this;
        }

        public Builder uri(String uri) {
            this.uri = uri;
            return this;
        }

        public Builder timestampColumn(String timestampColumn) {
            this.timestampColumn = timestampColumn;
            return this;
        }

        public Builder createdTimestampColumn(String createdTimestampColumn) {
            this.createdTimestampColumn = createdTimestampColumn;
            return this;
        }

        public Builder datePartitionColumn(String datePartitionColumn) {
            this.datePartitionColumn = datePartitionColumn;
            return this;
        }

        public Builder option(String k, String v) {
            if (k != null && v != null) options.put(k, v);
            return this;
        }

        public Builder options(Map<String, String> more) {
            if (more != null) options.putAll(more);
            return this;
        }

        public FeatureTable build() {
            return new FeatureTable(this);
        }
    }
}
