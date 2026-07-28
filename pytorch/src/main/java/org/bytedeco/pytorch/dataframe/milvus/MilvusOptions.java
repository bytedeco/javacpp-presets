package org.bytedeco.pytorch.dataframe.milvus;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;

import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Options for DataFrame ↔ Milvus I/O (row upsert / query / vector search).
 *
 * <pre>{@code
 * MilvusOptions opts = MilvusOptions.builder()
 *     .collection("docs")
 *     .idColumn("id")
 *     .vectorColumn("emb")
 *     .dim(384)
 *     .metric(VectorMetric.COSINE)
 *     .batchSize(200)
 *     .build();
 * df.toMilvus(m, opts);
 * DataFrame back = DataFrame.readMilvus(m, opts);
 * }</pre>
 */
public final class MilvusOptions {

    public enum IfExists {
        /** Drop collection then recreate (destructive). */
        REPLACE,
        /** Upsert into existing collection (default). */
        APPEND,
        /** Fail if collection already exists. */
        FAIL,
        /** Skip write when collection already has data. */
        SKIP
    }

    private final String collection;
    private final String dbName;
    private final String idField;
    private final String idColumn;
    private final String vectorField;
    private final String vectorColumn;
    private final int dim;
    private final VectorMetric metric;
    private final String indexType;
    private final IfExists ifExists;
    private final int batchSize;
    private final boolean loadAfterWrite;
    private final boolean ensureCollection;
    private final String filter;
    private final List<String> outputFields;
    private final List<String> payloadColumns;
    private final Map<String, Column.DType> dtype;
    private final boolean includeNulls;
    private final Duration timeout;
    private final int limit;
    private final boolean includeVector;

    private MilvusOptions(Builder b) {
        this.collection = Objects.requireNonNullElse(b.collection, "vectors");
        this.dbName = b.dbName == null ? "default" : b.dbName;
        this.idField = b.idField == null ? "id" : b.idField;
        this.idColumn = b.idColumn;
        this.vectorField = b.vectorField == null ? "vector" : b.vectorField;
        this.vectorColumn = b.vectorColumn;
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.indexType = b.indexType == null ? "AUTOINDEX" : b.indexType;
        this.ifExists = b.ifExists == null ? IfExists.APPEND : b.ifExists;
        this.batchSize = Math.max(1, b.batchSize);
        this.loadAfterWrite = b.loadAfterWrite;
        this.ensureCollection = b.ensureCollection;
        this.filter = b.filter;
        this.outputFields = b.outputFields == null ? null : List.copyOf(b.outputFields);
        this.payloadColumns = b.payloadColumns == null ? null : List.copyOf(b.payloadColumns);
        this.dtype = b.dtype == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.dtype));
        this.includeNulls = b.includeNulls;
        this.timeout = b.timeout;
        this.limit = b.limit;
        this.includeVector = b.includeVector;
    }

    public static Builder builder() { return new Builder(); }

    public static MilvusOptions defaults() { return builder().build(); }

    public static MilvusOptions collection(String name) {
        return builder().collection(name).build();
    }

    public static MilvusOptions collection(String name, int dim) {
        return builder().collection(name).dim(dim).build();
    }

    public String collection() { return collection; }
    public String dbName() { return dbName; }
    public String idField() { return idField; }
    public String idColumn() { return idColumn; }
    public String vectorField() { return vectorField; }
    public String vectorColumn() { return vectorColumn; }
    public int dim() { return dim; }
    public VectorMetric metric() { return metric; }
    public String indexType() { return indexType; }
    public IfExists ifExists() { return ifExists; }
    public int batchSize() { return batchSize; }
    public boolean loadAfterWrite() { return loadAfterWrite; }
    public boolean ensureCollection() { return ensureCollection; }
    public String filter() { return filter; }
    public List<String> outputFields() { return outputFields; }
    public List<String> payloadColumns() { return payloadColumns; }
    public Map<String, Column.DType> dtype() { return dtype; }
    public boolean includeNulls() { return includeNulls; }
    public Duration timeout() { return timeout; }
    public int limit() { return limit; }
    public boolean includeVector() { return includeVector; }

    public static final class Builder {
        private String collection = "vectors";
        private String dbName = "default";
        private String idField = "id";
        private String idColumn;
        private String vectorField = "vector";
        private String vectorColumn;
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private String indexType = "AUTOINDEX";
        private IfExists ifExists = IfExists.APPEND;
        private int batchSize = 200;
        private boolean loadAfterWrite = true;
        private boolean ensureCollection = true;
        private String filter;
        private List<String> outputFields;
        private List<String> payloadColumns;
        private Map<String, Column.DType> dtype;
        private boolean includeNulls = false;
        private Duration timeout;
        private int limit = 100_000;
        private boolean includeVector = true;

        public Builder collection(String v) { this.collection = v; return this; }
        public Builder dbName(String v) { this.dbName = v; return this; }
        public Builder idField(String v) { this.idField = v; return this; }
        public Builder idColumn(String v) { this.idColumn = v; return this; }
        public Builder vectorField(String v) { this.vectorField = v; return this; }
        public Builder vectorColumn(String v) { this.vectorColumn = v; return this; }
        public Builder dim(int v) { this.dim = v; return this; }
        public Builder metric(VectorMetric v) { this.metric = v; return this; }
        public Builder indexType(String v) { this.indexType = v; return this; }
        public Builder ifExists(IfExists v) { this.ifExists = v; return this; }
        public Builder batchSize(int v) { this.batchSize = v; return this; }
        public Builder loadAfterWrite(boolean v) { this.loadAfterWrite = v; return this; }
        public Builder ensureCollection(boolean v) { this.ensureCollection = v; return this; }
        public Builder filter(String v) { this.filter = v; return this; }
        public Builder outputFields(List<String> v) { this.outputFields = v; return this; }
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

        public MilvusOptions build() { return new MilvusOptions(this); }
    }

    /** Parse free-form map (URI query / VectorStores config). */
    public static MilvusOptions fromMap(Map<String, Object> cfg) {
        if (cfg == null || cfg.isEmpty()) return defaults();
        Builder b = builder();
        if (cfg.get("collection") != null) b.collection(String.valueOf(cfg.get("collection")));
        else if (cfg.get("name") != null) b.collection(String.valueOf(cfg.get("name")));
        if (cfg.get("dbName") != null) b.dbName(String.valueOf(cfg.get("dbName")));
        else if (cfg.get("database") != null) b.dbName(String.valueOf(cfg.get("database")));
        if (cfg.get("idField") != null) b.idField(String.valueOf(cfg.get("idField")));
        if (cfg.get("idColumn") != null) b.idColumn(String.valueOf(cfg.get("idColumn")));
        if (cfg.get("vectorField") != null) b.vectorField(String.valueOf(cfg.get("vectorField")));
        if (cfg.get("vectorColumn") != null) b.vectorColumn(String.valueOf(cfg.get("vectorColumn")));
        Object dim = cfg.get("dim");
        if (dim instanceof Number n) b.dim(n.intValue());
        else if (dim != null) {
            try { b.dim(Integer.parseInt(String.valueOf(dim).trim())); } catch (Exception ignored) {}
        }
        Object metric = cfg.get("metric");
        if (metric == null) metric = cfg.get("distance");
        if (metric instanceof VectorMetric m) b.metric(m);
        else if (metric != null) {
            String s = String.valueOf(metric).trim().toUpperCase(Locale.ROOT);
            b.metric(switch (s) {
                case "L2", "EUCLID", "EUCLIDEAN" -> VectorMetric.L2;
                case "IP", "DOT", "INNER_PRODUCT" -> VectorMetric.IP;
                default -> VectorMetric.COSINE;
            });
        }
        if (cfg.get("indexType") != null) b.indexType(String.valueOf(cfg.get("indexType")));
        Object batch = cfg.get("batchSize");
        if (batch instanceof Number n) b.batchSize(n.intValue());
        Object lim = cfg.get("limit");
        if (lim instanceof Number n) b.limit(n.intValue());
        if (cfg.get("filter") != null) b.filter(String.valueOf(cfg.get("filter")));
        return b.build();
    }
}
