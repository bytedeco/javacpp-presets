package org.bytedeco.pytorch.data.dataframe.mongo;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric;

import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Options for DataFrame ↔ Mongo (Atlas Data API) I/O.
 *
 * <pre>{@code
 * MongoOptions opts = MongoOptions.builder()
 *     .database("rag")
 *     .collection("docs")
 *     .idColumn("id")
 *     .vectorPath("embedding")
 *     .dim(384)
 *     .build();
 * df.toMongo(m, opts);
 * DataFrame back = DataFrame.readMongo(m, opts);
 * }</pre>
 */
public final class MongoOptions {

    public enum IfExists {
        /** Delete all documents then insert. */
        REPLACE,
        /** Upsert by id (default). */
        APPEND,
        /** Fail if collection already has documents. */
        FAIL,
        /** Skip write when collection is non-empty. */
        SKIP
    }

    private final String dataSource;
    private final String database;
    private final String collection;
    private final String idField;
    private final String idColumn;
    private final String vectorPath;
    private final String vectorColumn;
    private final String indexName;
    private final int dim;
    private final VectorMetric metric;
    private final IfExists ifExists;
    private final int batchSize;
    private final boolean ensureCollection;
    private final Map<String, Object> filter;
    private final Map<String, Object> projection;
    private final List<String> payloadColumns;
    private final Map<String, Column.DType> dtype;
    private final boolean includeNulls;
    private final Duration timeout;
    private final int limit;
    private final boolean includeVector;

    private MongoOptions(Builder b) {
        this.dataSource = b.dataSource == null ? "Cluster0" : b.dataSource;
        this.database = Objects.requireNonNullElse(b.database, "test");
        this.collection = Objects.requireNonNullElse(b.collection, "vectors");
        this.idField = b.idField == null ? "_id" : b.idField;
        this.idColumn = b.idColumn;
        this.vectorPath = b.vectorPath == null ? "embedding" : b.vectorPath;
        this.vectorColumn = b.vectorColumn;
        this.indexName = b.indexName == null ? "vector_index" : b.indexName;
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.ifExists = b.ifExists == null ? IfExists.APPEND : b.ifExists;
        this.batchSize = Math.max(1, b.batchSize);
        this.ensureCollection = b.ensureCollection;
        this.filter = b.filter == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.filter));
        this.projection = b.projection == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.projection));
        this.payloadColumns = b.payloadColumns == null ? null : List.copyOf(b.payloadColumns);
        this.dtype = b.dtype == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.dtype));
        this.includeNulls = b.includeNulls;
        this.timeout = b.timeout;
        this.limit = b.limit;
        this.includeVector = b.includeVector;
    }

    public static Builder builder() { return new Builder(); }
    public static MongoOptions defaults() { return builder().build(); }

    public static MongoOptions collection(String database, String collection) {
        return builder().database(database).collection(collection).build();
    }

    public String dataSource() { return dataSource; }
    public String database() { return database; }
    public String collection() { return collection; }
    public String idField() { return idField; }
    public String idColumn() { return idColumn; }
    public String vectorPath() { return vectorPath; }
    public String vectorColumn() { return vectorColumn; }
    public String indexName() { return indexName; }
    public int dim() { return dim; }
    public VectorMetric metric() { return metric; }
    public IfExists ifExists() { return ifExists; }
    public int batchSize() { return batchSize; }
    public boolean ensureCollection() { return ensureCollection; }
    public Map<String, Object> filter() { return filter; }
    public Map<String, Object> projection() { return projection; }
    public List<String> payloadColumns() { return payloadColumns; }
    public Map<String, Column.DType> dtype() { return dtype; }
    public boolean includeNulls() { return includeNulls; }
    public Duration timeout() { return timeout; }
    public int limit() { return limit; }
    public boolean includeVector() { return includeVector; }

    public static final class Builder {
        private String dataSource = "Cluster0";
        private String database = "test";
        private String collection = "vectors";
        private String idField = "_id";
        private String idColumn;
        private String vectorPath = "embedding";
        private String vectorColumn;
        private String indexName = "vector_index";
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private IfExists ifExists = IfExists.APPEND;
        private int batchSize = 100;
        private boolean ensureCollection = true;
        private Map<String, Object> filter;
        private Map<String, Object> projection;
        private List<String> payloadColumns;
        private Map<String, Column.DType> dtype;
        private boolean includeNulls = false;
        private Duration timeout;
        private int limit = 100_000;
        private boolean includeVector = true;

        public Builder dataSource(String v) { this.dataSource = v; return this; }
        public Builder database(String v) { this.database = v; return this; }
        public Builder collection(String v) { this.collection = v; return this; }
        public Builder idField(String v) { this.idField = v; return this; }
        public Builder idColumn(String v) { this.idColumn = v; return this; }
        public Builder vectorPath(String v) { this.vectorPath = v; return this; }
        public Builder vectorColumn(String v) { this.vectorColumn = v; return this; }
        public Builder indexName(String v) { this.indexName = v; return this; }
        public Builder dim(int v) { this.dim = v; return this; }
        public Builder metric(VectorMetric v) { this.metric = v; return this; }
        public Builder ifExists(IfExists v) { this.ifExists = v; return this; }
        public Builder batchSize(int v) { this.batchSize = v; return this; }
        public Builder ensureCollection(boolean v) { this.ensureCollection = v; return this; }
        public Builder filter(Map<String, Object> v) { this.filter = v; return this; }
        public Builder projection(Map<String, Object> v) { this.projection = v; return this; }
        public Builder payloadColumns(List<String> v) { this.payloadColumns = v; return this; }
        public Builder payloadColumns(String... v) {
            this.payloadColumns = v == null ? null : List.of(v);
            return this;
        }
        public Builder dtype(Map<String, Column.DType> v) { this.dtype = v; return this; }
        public Builder includeNulls(boolean v) { this.includeNulls = v; return this; }
        public Builder timeout(Duration v) { this.timeout = v; return this; }
        public Builder limit(int v) { this.limit = v; return this; }
        public Builder includeVector(boolean v) { this.includeVector = v; return this; }

        public MongoOptions build() { return new MongoOptions(this); }
    }

    public static MongoOptions fromMap(Map<String, Object> cfg) {
        if (cfg == null || cfg.isEmpty()) return defaults();
        Builder b = builder();
        if (cfg.get("dataSource") != null) b.dataSource(String.valueOf(cfg.get("dataSource")));
        else if (cfg.get("cluster") != null) b.dataSource(String.valueOf(cfg.get("cluster")));
        if (cfg.get("database") != null) b.database(String.valueOf(cfg.get("database")));
        else if (cfg.get("db") != null) b.database(String.valueOf(cfg.get("db")));
        if (cfg.get("collection") != null) b.collection(String.valueOf(cfg.get("collection")));
        if (cfg.get("idField") != null) b.idField(String.valueOf(cfg.get("idField")));
        if (cfg.get("idColumn") != null) b.idColumn(String.valueOf(cfg.get("idColumn")));
        if (cfg.get("vectorPath") != null) b.vectorPath(String.valueOf(cfg.get("vectorPath")));
        else if (cfg.get("vectorField") != null) b.vectorPath(String.valueOf(cfg.get("vectorField")));
        if (cfg.get("vectorColumn") != null) b.vectorColumn(String.valueOf(cfg.get("vectorColumn")));
        if (cfg.get("indexName") != null) b.indexName(String.valueOf(cfg.get("indexName")));
        else if (cfg.get("index") != null) b.indexName(String.valueOf(cfg.get("index")));
        Object dim = cfg.get("dim");
        if (dim instanceof Number n) b.dim(n.intValue());
        else if (dim != null) {
            try { b.dim(Integer.parseInt(String.valueOf(dim).trim())); } catch (Exception ignored) {}
        }
        Object metric = cfg.get("metric");
        if (metric instanceof VectorMetric m) b.metric(m);
        else if (metric != null) {
            String s = String.valueOf(metric).trim().toUpperCase(Locale.ROOT);
            b.metric(switch (s) {
                case "L2", "EUCLID", "EUCLIDEAN" -> VectorMetric.L2;
                case "IP", "DOT", "INNER_PRODUCT" -> VectorMetric.IP;
                default -> VectorMetric.COSINE;
            });
        }
        Object batch = cfg.get("batchSize");
        if (batch instanceof Number n) b.batchSize(n.intValue());
        Object lim = cfg.get("limit");
        if (lim instanceof Number n) b.limit(n.intValue());
        return b.build();
    }
}
