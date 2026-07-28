package org.bytedeco.pytorch.dataframe.opensearch;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.vectorstore.PayloadField;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Options for DataFrame ↔ OpenSearch I/O (document bulk / search / knn).
 *
 * <pre>{@code
 * OpenSearchOptions opts = OpenSearchOptions.builder()
 *     .index("docs")
 *     .idColumn("id")
 *     .vectorColumn("emb")
 *     .dim(384)
 *     .refresh(true)
 *     .build();
 * df.toOpenSearch(os, opts);
 * DataFrame back = DataFrame.readOpenSearch(os, opts);
 * }</pre>
 */
public final class OpenSearchOptions {

    public enum IfExists {
        /** Delete index then recreate. */
        REPLACE,
        /** Bulk index into existing (default). */
        APPEND,
        /** Fail if index already exists. */
        FAIL,
        /** Skip write when index exists. */
        SKIP
    }

    private final String index;
    private final String idColumn;
    private final String vectorField;
    private final String vectorColumn;
    private final int dim;
    private final VectorMetric metric;
    private final String engine;
    private final int m;
    private final int efConstruction;
    private final IfExists ifExists;
    private final int bulkBatch;
    private final boolean refreshOnWrite;
    private final boolean ensureIndex;
    private final String filterQuery;
    private final List<PayloadField> payloadFields;
    private final List<String> payloadColumns;
    private final Map<String, Column.DType> dtype;
    private final boolean includeNulls;
    private final Duration timeout;
    private final int limit;
    private final boolean includeVector;
    private final String pipeline;
    private final String routing;

    private OpenSearchOptions(Builder b) {
        this.index = Objects.requireNonNullElse(b.index, "vectors");
        this.idColumn = b.idColumn;
        this.vectorField = b.vectorField == null ? "vector" : b.vectorField;
        this.vectorColumn = b.vectorColumn;
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.engine = b.engine == null ? "faiss" : b.engine;
        this.m = b.m;
        this.efConstruction = b.efConstruction;
        this.ifExists = b.ifExists == null ? IfExists.APPEND : b.ifExists;
        this.bulkBatch = Math.max(1, b.bulkBatch);
        this.refreshOnWrite = b.refreshOnWrite;
        this.ensureIndex = b.ensureIndex;
        this.filterQuery = b.filterQuery;
        this.payloadFields = List.copyOf(b.payloadFields);
        this.payloadColumns = b.payloadColumns == null ? null : List.copyOf(b.payloadColumns);
        this.dtype = b.dtype == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.dtype));
        this.includeNulls = b.includeNulls;
        this.timeout = b.timeout;
        this.limit = b.limit;
        this.includeVector = b.includeVector;
        this.pipeline = b.pipeline;
        this.routing = b.routing;
    }

    public static Builder builder() { return new Builder(); }
    public static OpenSearchOptions defaults() { return builder().build(); }

    public static OpenSearchOptions index(String name) {
        return builder().index(name).build();
    }

    public static OpenSearchOptions index(String name, int dim) {
        return builder().index(name).dim(dim).build();
    }

    public String index() { return index; }
    public String idColumn() { return idColumn; }
    public String vectorField() { return vectorField; }
    public String vectorColumn() { return vectorColumn; }
    public int dim() { return dim; }
    public VectorMetric metric() { return metric; }
    public String engine() { return engine; }
    public int m() { return m; }
    public int efConstruction() { return efConstruction; }
    public IfExists ifExists() { return ifExists; }
    public int bulkBatch() { return bulkBatch; }
    public boolean refreshOnWrite() { return refreshOnWrite; }
    public boolean ensureIndex() { return ensureIndex; }
    public String filterQuery() { return filterQuery; }
    public List<PayloadField> payloadFields() { return payloadFields; }
    public List<String> payloadColumns() { return payloadColumns; }
    public Map<String, Column.DType> dtype() { return dtype; }
    public boolean includeNulls() { return includeNulls; }
    public Duration timeout() { return timeout; }
    public int limit() { return limit; }
    public boolean includeVector() { return includeVector; }
    public String pipeline() { return pipeline; }
    public String routing() { return routing; }

    public static final class Builder {
        private String index = "vectors";
        private String idColumn;
        private String vectorField = "vector";
        private String vectorColumn;
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private String engine = "faiss";
        private int m = 16;
        private int efConstruction = 100;
        private IfExists ifExists = IfExists.APPEND;
        private int bulkBatch = 500;
        private boolean refreshOnWrite = true;
        private boolean ensureIndex = true;
        private String filterQuery;
        private final List<PayloadField> payloadFields = new ArrayList<>();
        private List<String> payloadColumns;
        private Map<String, Column.DType> dtype;
        private boolean includeNulls = false;
        private Duration timeout;
        private int limit = 100_000;
        private boolean includeVector = true;
        private String pipeline;
        private String routing;

        public Builder index(String v) { this.index = v; return this; }
        public Builder idColumn(String v) { this.idColumn = v; return this; }
        public Builder vectorField(String v) { this.vectorField = v; return this; }
        public Builder vectorColumn(String v) { this.vectorColumn = v; return this; }
        public Builder dim(int v) { this.dim = v; return this; }
        public Builder metric(VectorMetric v) { this.metric = v; return this; }
        public Builder engine(String v) { this.engine = v; return this; }
        public Builder m(int v) { this.m = v; return this; }
        public Builder efConstruction(int v) { this.efConstruction = v; return this; }
        public Builder ifExists(IfExists v) { this.ifExists = v; return this; }
        public Builder bulkBatch(int v) { this.bulkBatch = v; return this; }
        public Builder refreshOnWrite(boolean v) { this.refreshOnWrite = v; return this; }
        public Builder ensureIndex(boolean v) { this.ensureIndex = v; return this; }
        public Builder filterQuery(String v) { this.filterQuery = v; return this; }
        public Builder payloadField(PayloadField pf) {
            if (pf != null) this.payloadFields.add(pf);
            return this;
        }
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
        public Builder pipeline(String v) { this.pipeline = v; return this; }
        public Builder routing(String v) { this.routing = v; return this; }

        public OpenSearchOptions build() { return new OpenSearchOptions(this); }
    }

    public static OpenSearchOptions fromMap(Map<String, Object> cfg) {
        if (cfg == null || cfg.isEmpty()) return defaults();
        Builder b = builder();
        if (cfg.get("index") != null) b.index(String.valueOf(cfg.get("index")));
        else if (cfg.get("collection") != null) b.index(String.valueOf(cfg.get("collection")));
        if (cfg.get("idColumn") != null) b.idColumn(String.valueOf(cfg.get("idColumn")));
        if (cfg.get("vectorField") != null) b.vectorField(String.valueOf(cfg.get("vectorField")));
        if (cfg.get("vectorColumn") != null) b.vectorColumn(String.valueOf(cfg.get("vectorColumn")));
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
        if (cfg.get("engine") != null) b.engine(String.valueOf(cfg.get("engine")));
        Object batch = cfg.get("bulkBatch");
        if (batch == null) batch = cfg.get("batchSize");
        if (batch instanceof Number n) b.bulkBatch(n.intValue());
        Object lim = cfg.get("limit");
        if (lim instanceof Number n) b.limit(n.intValue());
        Object refresh = cfg.get("refresh");
        if (refresh instanceof Boolean bo) b.refreshOnWrite(bo);
        return b.build();
    }
}
