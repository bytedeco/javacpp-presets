package org.bytedeco.pytorch.dataframe.pgvector;

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
 * Options for DataFrame ↔ pgvector (JDBC) I/O.
 *
 * <p>Two layouts:
 * <ul>
 *   <li>{@link PayloadMode#JSONB} — {@code id TEXT, embedding vector(d), payload JSONB}
 *       (vector-store style, default)</li>
 *   <li>{@link PayloadMode#COLUMNS} — each DataFrame column becomes a SQL column
 *       (table-style, vector column optional)</li>
 * </ul>
 *
 * <pre>{@code
 * PgVectorOptions opts = PgVectorOptions.builder()
 *     .table("docs")
 *     .idColumn("id")
 *     .vectorColumn("emb")
 *     .dim(384)
 *     .payloadMode(PgVectorOptions.PayloadMode.COLUMNS)
 *     .build();
 * df.toPgVector(pg, opts);
 * DataFrame back = DataFrame.readPgVector(pg, opts);
 * }</pre>
 */
public final class PgVectorOptions {

    public enum IfExists {
        /** DROP TABLE then recreate. */
        REPLACE,
        /** Upsert / INSERT into existing (default). */
        APPEND,
        /** Fail if table already exists. */
        FAIL,
        /** Skip write when table is non-empty. */
        SKIP
    }

    /**
     * How non-id / non-vector columns are stored.
     */
    public enum PayloadMode {
        /** Single JSONB column holding a map of remaining fields. */
        JSONB,
        /** Expand each DataFrame column into its own SQL column. */
        COLUMNS
    }

    public enum IndexMethod {
        HNSW,
        IVFFLAT,
        NONE
    }

    private final String table;
    private final String schema;
    private final String idColumn;
    private final String idSqlColumn;
    private final String vectorColumn;
    private final String vectorSqlColumn;
    private final String payloadColumn;
    private final int dim;
    private final VectorMetric metric;
    private final IfExists ifExists;
    private final PayloadMode payloadMode;
    private final IndexMethod indexMethod;
    private final int chunksize;
    private final boolean ensureExtension;
    private final boolean ensureTable;
    private final boolean ensureIndex;
    private final String where;
    private final List<String> payloadColumns;
    private final Map<String, Column.DType> dtype;
    private final boolean includeNulls;
    private final Duration timeout;
    private final int limit;
    private final boolean includeVector;
    private final boolean quoteIdentifiers;

    private PgVectorOptions(Builder b) {
        this.table = Objects.requireNonNullElse(b.table, "vectors");
        this.schema = b.schema;
        this.idColumn = b.idColumn;
        this.idSqlColumn = b.idSqlColumn == null ? "id" : b.idSqlColumn;
        this.vectorColumn = b.vectorColumn;
        this.vectorSqlColumn = b.vectorSqlColumn == null ? "embedding" : b.vectorSqlColumn;
        this.payloadColumn = b.payloadColumn == null ? "payload" : b.payloadColumn;
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.ifExists = b.ifExists == null ? IfExists.APPEND : b.ifExists;
        this.payloadMode = b.payloadMode == null ? PayloadMode.JSONB : b.payloadMode;
        this.indexMethod = b.indexMethod == null ? IndexMethod.HNSW : b.indexMethod;
        this.chunksize = Math.max(1, b.chunksize);
        this.ensureExtension = b.ensureExtension;
        this.ensureTable = b.ensureTable;
        this.ensureIndex = b.ensureIndex;
        this.where = b.where;
        this.payloadColumns = b.payloadColumns == null ? null : List.copyOf(b.payloadColumns);
        this.dtype = b.dtype == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.dtype));
        this.includeNulls = b.includeNulls;
        this.timeout = b.timeout;
        this.limit = b.limit;
        this.includeVector = b.includeVector;
        this.quoteIdentifiers = b.quoteIdentifiers;
    }

    public static Builder builder() { return new Builder(); }
    public static PgVectorOptions defaults() { return builder().build(); }

    public static PgVectorOptions table(String name) {
        return builder().table(name).build();
    }

    public static PgVectorOptions table(String name, int dim) {
        return builder().table(name).dim(dim).build();
    }

    public String table() { return table; }
    public String schema() { return schema; }
    public String idColumn() { return idColumn; }
    public String idSqlColumn() { return idSqlColumn; }
    public String vectorColumn() { return vectorColumn; }
    public String vectorSqlColumn() { return vectorSqlColumn; }
    public String payloadColumn() { return payloadColumn; }
    public int dim() { return dim; }
    public VectorMetric metric() { return metric; }
    public IfExists ifExists() { return ifExists; }
    public PayloadMode payloadMode() { return payloadMode; }
    public IndexMethod indexMethod() { return indexMethod; }
    public int chunksize() { return chunksize; }
    public boolean ensureExtension() { return ensureExtension; }
    public boolean ensureTable() { return ensureTable; }
    public boolean ensureIndex() { return ensureIndex; }
    public String where() { return where; }
    public List<String> payloadColumns() { return payloadColumns; }
    public Map<String, Column.DType> dtype() { return dtype; }
    public boolean includeNulls() { return includeNulls; }
    public Duration timeout() { return timeout; }
    public int limit() { return limit; }
    public boolean includeVector() { return includeVector; }
    public boolean quoteIdentifiers() { return quoteIdentifiers; }

    /** Fully-qualified table reference (schema.table or table). */
    public String qualifiedTable() {
        if (schema == null || schema.isBlank()) return table;
        return schema + "." + table;
    }

    public static final class Builder {
        private String table = "vectors";
        private String schema;
        private String idColumn;
        private String idSqlColumn = "id";
        private String vectorColumn;
        private String vectorSqlColumn = "embedding";
        private String payloadColumn = "payload";
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private IfExists ifExists = IfExists.APPEND;
        private PayloadMode payloadMode = PayloadMode.JSONB;
        private IndexMethod indexMethod = IndexMethod.HNSW;
        private int chunksize = 500;
        private boolean ensureExtension = true;
        private boolean ensureTable = true;
        private boolean ensureIndex = true;
        private String where;
        private List<String> payloadColumns;
        private Map<String, Column.DType> dtype;
        private boolean includeNulls = false;
        private Duration timeout;
        private int limit = 100_000;
        private boolean includeVector = true;
        private boolean quoteIdentifiers = true;

        public Builder table(String v) { this.table = v; return this; }
        public Builder schema(String v) { this.schema = v; return this; }
        public Builder idColumn(String v) { this.idColumn = v; return this; }
        public Builder idSqlColumn(String v) { this.idSqlColumn = v; return this; }
        public Builder vectorColumn(String v) { this.vectorColumn = v; return this; }
        public Builder vectorSqlColumn(String v) { this.vectorSqlColumn = v; return this; }
        public Builder payloadColumn(String v) { this.payloadColumn = v; return this; }
        public Builder dim(int v) { this.dim = v; return this; }
        public Builder metric(VectorMetric v) { this.metric = v; return this; }
        public Builder ifExists(IfExists v) { this.ifExists = v; return this; }
        public Builder payloadMode(PayloadMode v) { this.payloadMode = v; return this; }
        public Builder indexMethod(IndexMethod v) { this.indexMethod = v; return this; }
        public Builder chunksize(int v) { this.chunksize = v; return this; }
        public Builder ensureExtension(boolean v) { this.ensureExtension = v; return this; }
        public Builder ensureTable(boolean v) { this.ensureTable = v; return this; }
        public Builder ensureIndex(boolean v) { this.ensureIndex = v; return this; }
        public Builder where(String v) { this.where = v; return this; }
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
        public Builder quoteIdentifiers(boolean v) { this.quoteIdentifiers = v; return this; }

        public PgVectorOptions build() { return new PgVectorOptions(this); }
    }

    public static PgVectorOptions fromMap(Map<String, Object> cfg) {
        if (cfg == null || cfg.isEmpty()) return defaults();
        Builder b = builder();
        if (cfg.get("table") != null) b.table(String.valueOf(cfg.get("table")));
        else if (cfg.get("collection") != null) b.table(String.valueOf(cfg.get("collection")));
        if (cfg.get("schema") != null) b.schema(String.valueOf(cfg.get("schema")));
        if (cfg.get("idColumn") != null) b.idColumn(String.valueOf(cfg.get("idColumn")));
        if (cfg.get("vectorColumn") != null) b.vectorColumn(String.valueOf(cfg.get("vectorColumn")));
        if (cfg.get("vectorField") != null) b.vectorSqlColumn(String.valueOf(cfg.get("vectorField")));
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
        Object mode = cfg.get("payloadMode");
        if (mode != null) {
            String s = String.valueOf(mode).trim().toUpperCase(Locale.ROOT);
            if ("COLUMNS".equals(s) || "COLUMN".equals(s)) b.payloadMode(PayloadMode.COLUMNS);
            else b.payloadMode(PayloadMode.JSONB);
        }
        Object chunk = cfg.get("chunksize");
        if (chunk == null) chunk = cfg.get("batchSize");
        if (chunk instanceof Number n) b.chunksize(n.intValue());
        Object lim = cfg.get("limit");
        if (lim instanceof Number n) b.limit(n.intValue());
        if (cfg.get("where") != null) b.where(String.valueOf(cfg.get("where")));
        return b.build();
    }
}
