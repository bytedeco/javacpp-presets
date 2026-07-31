/*
 * Feature ingest — DataFrame (post FE pipeline) → FeatureView offline rows.
 *
 * Upstream: raw events / logs as DataFrame
 *   → df.feature().impute(...).standardScale(...).oneHot(...).build()
 *   → FeatureIngest.into(platform).view(...).from(df).run()
 * Downstream: materialize → online serve / historical PIT training export
 *
 * Aligns with Feast materialize-from-dataframe / Databricks Feature Engineering
 * write paths used at Meta, Alibaba, ByteDance feature platforms.
 */
package org.bytedeco.pytorch.feature.pipeline;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.FeatureEngineering;
import org.bytedeco.pytorch.dataframe.feature.pipeline.Pipeline;
import org.bytedeco.pytorch.feature.core.*;
import org.bytedeco.pytorch.feature.FeaturePlatform;
import org.bytedeco.pytorch.feature.bridge.DataFrameBridge;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.lifecycle.FeatureValidator;
import org.bytedeco.pytorch.feature.offline.FileOfflineStore;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * Fluent ingest of DataFrame rows into the feature warehouse (offline store)
 * with optional schema auto-registration and validation.
 *
 * <pre>{@code
 * FeatureIngest.Result r = FeatureIngest.into(fp)
 *     .project("recsys")
 *     .view("user_stats_7d")
 *     .entities("user_id")
 *     .timestampColumn("event_timestamp")
 *     .ttlDays(7)
 *     .autoRegister(true)
 *     .validate(true)
 *     .featureEngineering(fe -> fe.impute("mean", "age").standardScale("age", "score"))
 *     .from(rawDf)
 *     .run();
 * }</pre>
 */
public final class FeatureIngest {

    /** Outcome of one ingest run. */
    public static final class Result {
        public final String project;
        public final String viewName;
        public final FeatureView view;
        public final long rowsIn;
        public final long rowsWritten;
        public final boolean registered;
        public final boolean replaced;
        public final FeatureValidator.Report validation;
        public final long elapsedNanos;
        public final List<String> featureColumns;
        public final String message;

        Result(String project, String viewName, FeatureView view, long rowsIn, long rowsWritten,
               boolean registered, boolean replaced, FeatureValidator.Report validation,
               long elapsedNanos, List<String> featureColumns, String message) {
            this.project = project;
            this.viewName = viewName;
            this.view = view;
            this.rowsIn = rowsIn;
            this.rowsWritten = rowsWritten;
            this.registered = registered;
            this.replaced = replaced;
            this.validation = validation;
            this.elapsedNanos = elapsedNanos;
            this.featureColumns = featureColumns != null ? List.copyOf(featureColumns) : List.of();
            this.message = message != null ? message : "";
        }

        public double elapsedMs() {
            return elapsedNanos / 1_000_000.0;
        }

        public boolean ok() {
            return validation == null || validation.ok;
        }

        @Override
        public String toString() {
            return "IngestResult{view=" + project + "/" + viewName
                    + ", in=" + rowsIn + ", written=" + rowsWritten
                    + ", registered=" + registered
                    + ", ms=" + String.format("%.2f", elapsedMs())
                    + (validation != null ? ", val=" + validation : "")
                    + "}";
        }
    }

    private final FeaturePlatform platform;
    private String project = "default";
    private String viewName;
    private final List<String> entityCols = new ArrayList<>();
    private final List<String> featureCols = new ArrayList<>(); // empty → all non-entity/non-ts
    private final List<String> excludeCols = new ArrayList<>(); // always skipped (e.g. label)
    private String timestampColumn = "event_timestamp";
    private Duration ttl = Duration.ofDays(7);
    private boolean online = true;
    private boolean autoRegister = true;
    private boolean replace = false;
    private boolean validate = true;
    private boolean createMissingTimestamp = true;
    private UnaryOperator<DataFrame> transform;
    private Pipeline fittedPipeline; // optional reusable sklearn-style pipeline
    private String description = "";
    private DataFrame source;

    private FeatureIngest(FeaturePlatform platform) {
        this.platform = Objects.requireNonNull(platform, "platform");
    }

    public static FeatureIngest into(FeaturePlatform platform) {
        return new FeatureIngest(platform);
    }

    public FeatureIngest project(String project) {
        this.project = project != null && !project.isEmpty() ? project : "default";
        return this;
    }

    public FeatureIngest view(String viewName) {
        this.viewName = Objects.requireNonNull(viewName, "viewName");
        return this;
    }

    public FeatureIngest entities(String... cols) {
        if (cols != null) entityCols.addAll(Arrays.asList(cols));
        return this;
    }

    public FeatureIngest entities(List<String> cols) {
        if (cols != null) entityCols.addAll(cols);
        return this;
    }

    /** Explicit feature columns; default = all columns except entities + timestamp + excludes. */
    public FeatureIngest features(String... cols) {
        if (cols != null) featureCols.addAll(Arrays.asList(cols));
        return this;
    }

    /** Columns never ingested as features (e.g. {@code label}, request-only fields). */
    public FeatureIngest exclude(String... cols) {
        if (cols != null) {
            for (String c : cols) {
                if (c != null && !c.isEmpty() && !excludeCols.contains(c)) excludeCols.add(c);
            }
        }
        return this;
    }

    public FeatureIngest timestampColumn(String timestampColumn) {
        this.timestampColumn = timestampColumn != null ? timestampColumn : "event_timestamp";
        return this;
    }

    public FeatureIngest ttl(Duration ttl) {
        this.ttl = ttl != null ? ttl : Duration.ZERO;
        return this;
    }

    public FeatureIngest ttlDays(long days) {
        this.ttl = Duration.ofDays(days);
        return this;
    }

    public FeatureIngest online(boolean online) {
        this.online = online;
        return this;
    }

    public FeatureIngest autoRegister(boolean autoRegister) {
        this.autoRegister = autoRegister;
        return this;
    }

    /** Replace all existing offline rows for the view instead of append. */
    public FeatureIngest replace(boolean replace) {
        this.replace = replace;
        return this;
    }

    public FeatureIngest validate(boolean validate) {
        this.validate = validate;
        return this;
    }

    public FeatureIngest description(String description) {
        this.description = description != null ? description : "";
        return this;
    }

    /**
     * Apply a fitted {@link Pipeline} (from {@code df.feature()...toPipeline()}) before ingest.
     */
    public FeatureIngest pipeline(Pipeline pipeline) {
        this.fittedPipeline = pipeline;
        return this;
    }

    /**
     * Inline DataFrame feature engineering before ingest.
     * Example: {@code fe -> { try { return fe.impute("mean","age").standardScale("age").build(); }
     * catch (Exception e) { throw new RuntimeException(e); } }}
     */
    public FeatureIngest transform(UnaryOperator<DataFrame> transform) {
        this.transform = transform;
        return this;
    }

    /**
     * Convenience: run {@link FeatureEngineering} steps then ingest.
     * The operator receives {@code df.feature()} and must return a DataFrame
     * (typically via {@link FeatureEngineering#build()}).
     */
    public FeatureIngest featureEngineering(FeatureEngOp op) {
        Objects.requireNonNull(op, "op");
        this.transform = df -> {
            try {
                return op.apply(df.feature());
            } catch (Exception e) {
                throw new IllegalStateException("feature engineering failed: " + e.getMessage(), e);
            }
        };
        return this;
    }

    @FunctionalInterface
    public interface FeatureEngOp {
        DataFrame apply(FeatureEngineering fe) throws Exception;
    }

    public FeatureIngest from(DataFrame df) {
        this.source = Objects.requireNonNull(df, "df");
        return this;
    }

    public FeatureIngest fromRows(List<Map<String, Object>> rows) {
        this.source = DataFrameBridge.fromRows(rows);
        return this;
    }

    public Result run() {
        long t0 = System.nanoTime();
        if (viewName == null || viewName.isEmpty()) {
            throw new IllegalStateException("view name required");
        }
        if (source == null) {
            throw new IllegalStateException("source DataFrame required (from(...))");
        }
        if (entityCols.isEmpty()) {
            throw new IllegalStateException("at least one entity column required");
        }

        DataFrame df = source;
        long rowsIn = df.rowCount();

        // 1) optional fitted pipeline
        if (fittedPipeline != null) {
            try {
                df = fittedPipeline.transform(df);
            } catch (Exception e) {
                throw new IllegalStateException("pipeline transform failed: " + e.getMessage(), e);
            }
        }
        // 2) optional ad-hoc transform / FE
        if (transform != null) {
            df = transform.apply(df);
        }

        // 3) ensure timestamp
        if (createMissingTimestamp && !df.hasColumn(timestampColumn)) {
            df.addColumn(timestampColumn, Column.DType.INT64);
            long now = System.currentTimeMillis();
            for (int i = 0; i < df.rowCount(); i++) {
                df.set(i, timestampColumn, now);
            }
        }

        // 4) resolve feature columns
        List<String> feats = resolveFeatureColumns(df);
        // 5) project to entity + ts + features (stable order)
        List<String> keep = new ArrayList<>();
        for (String e : entityCols) {
            if (!df.hasColumn(e)) {
                throw new IllegalArgumentException("entity column missing in DataFrame: " + e);
            }
            keep.add(e);
        }
        if (df.hasColumn(timestampColumn) && !keep.contains(timestampColumn)) {
            keep.add(timestampColumn);
        }
        for (String f : feats) {
            if (!keep.contains(f)) keep.add(f);
        }
        DataFrame projected = DataFrameBridge.selectColumns(df, keep);

        // 6) register view if needed
        FeatureRegistry registry = platform.registry();
        boolean registered = false;
        FeatureView view = registry.getFeatureView(project, viewName).orElse(null);
        if (view == null) {
            if (!autoRegister) {
                throw new IllegalStateException("feature view not registered: " + project + "/" + viewName
                        + " (enable autoRegister or registerFeatureView first)");
            }
            view = buildViewFromDataFrame(projected, feats);
            // ensure entities exist
            for (String ec : entityCols) {
                final String entityName = ec;
                if (registry.getEntity(project, entityName).isEmpty()) {
                    ValueType vt = inferEntityType(projected, entityName);
                    registry.registerEntity(Entity.builder(entityName).project(project).valueType(vt).joinKey(entityName).build());
                }
            }
            registry.registerProject(Project.of(project));
            registry.registerFeatureView(view);
            registered = true;
        }

        // 7) to rows + validate
        List<Map<String, Object>> rows = DataFrameBridge.toRows(projected);
        // normalize timestamp to epoch millis long
        for (Map<String, Object> row : rows) {
            Object ts = row.get(timestampColumn);
            if (ts != null && !(ts instanceof Long)) {
                row.put(timestampColumn, FileOfflineStore.toEpochMillis(ts));
            }
        }

        FeatureValidator.Report report = null;
        if (validate) {
            report = new FeatureValidator().validate(view, rows);
            if (!report.ok) {
                return new Result(project, viewName, view, rowsIn, 0, registered, replace, report,
                        System.nanoTime() - t0, feats, "validation failed");
            }
        }

        // 8) write offline
        if (replace) {
            platform.offline().replace(project, viewName, rows);
        } else {
            platform.offline().put(project, viewName, rows);
        }

        return new Result(project, viewName, view, rowsIn, rows.size(), registered, replace, report,
                System.nanoTime() - t0, feats, "ok");
    }

    private List<String> resolveFeatureColumns(DataFrame df) {
        Set<String> skip = new LinkedHashSet<>(entityCols);
        skip.add(timestampColumn);
        skip.addAll(excludeCols);
        if (!featureCols.isEmpty()) {
            List<String> out = new ArrayList<>();
            for (String c : featureCols) {
                if (skip.contains(c)) continue;
                if (!df.hasColumn(c)) {
                    throw new IllegalArgumentException("feature column missing: " + c);
                }
                out.add(c);
            }
            return out;
        }
        List<String> out = new ArrayList<>();
        for (String c : df.getColumnNames()) {
            if (!skip.contains(c)) out.add(c);
        }
        return out;
    }

    private FeatureView buildViewFromDataFrame(DataFrame df, List<String> feats) {
        List<Entity> entities = new ArrayList<>();
        for (String ec : entityCols) {
            entities.add(platform.registry().getEntity(project, ec)
                    .orElse(Entity.builder(ec).project(project)
                            .valueType(inferEntityType(df, ec))
                            .joinKey(ec)
                            .build()));
        }
        List<Field> fields = new ArrayList<>();
        for (String f : feats) {
            fields.add(Field.builder(f).valueType(inferValueType(df, f)).build());
        }
        return FeatureView.builder(viewName)
                .project(project)
                .entities(entities)
                .schema(fields)
                .source(FeatureTable.memory(viewName))
                .ttl(ttl)
                .online(online)
                .description(description.isEmpty()
                        ? "auto-registered from DataFrame ingest"
                        : description)
                .tag("source", "dataframe_ingest")
                .build();
    }

    static ValueType inferEntityType(DataFrame df, String col) {
        if (!df.hasColumn(col)) return ValueType.INT64;
        Column.DType dt = df.column(col).dtype();
        switch (dt) {
            case INT32:
                return ValueType.INT32;
            case INT64:
                return ValueType.INT64;
            case STRING:
                return ValueType.STRING;
            case FLOAT32:
            case FLOAT64:
                return ValueType.INT64; // cast ids
            default:
                return ValueType.STRING;
        }
    }

    static ValueType inferValueType(DataFrame df, String col) {
        if (!df.hasColumn(col)) return ValueType.UNKNOWN;
        Column.DType dt = df.column(col).dtype();
        switch (dt) {
            case INT32:
                return ValueType.INT32;
            case INT64:
                return ValueType.INT64;
            case FLOAT32:
                return ValueType.FLOAT32;
            case FLOAT64:
                return ValueType.FLOAT64;
            case BOOLEAN:
                return ValueType.BOOL;
            case STRING:
                return ValueType.STRING;
            case VECTOR:
            case EMBEDDING:
                return ValueType.EMBEDDING;
            case LIST:
                // peek first non-null
                Column c = df.column(col);
                for (int i = 0; i < Math.min(c.size(), 16); i++) {
                    Object v = c.get(i);
                    if (v instanceof long[] || v instanceof int[]) return ValueType.INT64_LIST;
                    if (v instanceof float[] || v instanceof double[]) return ValueType.FLOAT32_LIST;
                    if (v instanceof List && !((List<?>) v).isEmpty()) {
                        Object x = ((List<?>) v).get(0);
                        if (x instanceof Number) {
                            return (x instanceof Double || x instanceof Float)
                                    ? ValueType.FLOAT64_LIST : ValueType.INT64_LIST;
                        }
                        return ValueType.STRING_LIST;
                    }
                }
                return ValueType.INT64_LIST;
            default:
                return ValueType.STRING;
        }
    }
}
