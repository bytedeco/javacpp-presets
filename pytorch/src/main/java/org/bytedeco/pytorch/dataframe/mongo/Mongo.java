package org.bytedeco.pytorch.dataframe.mongo;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.dataframe.vectorstore.http.HttpJson;
import org.bytedeco.pytorch.dataframe.vectorstore.mongo.MongoAtlasVectorStore;

import java.io.Closeable;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Full-featured Mongo client for DataFrame I/O — Atlas Data API (HTTPS) via
 * {@link HttpJson}, no {@code mongodb-driver-sync} dependency.
 *
 * <h2>Coverage (MongoCollection parity subset over Data API)</h2>
 * <ul>
 *   <li><b>Connection</b> — connect / apiKey / dataSource / database / close</li>
 *   <li><b>CRUD</b> — insertOne/Many, find/findOne, updateOne/Many, replaceOne,
 *       deleteOne/Many, countDocuments, aggregate</li>
 *   <li><b>Vector</b> — {@code $vectorSearch} via aggregate</li>
 *   <li><b>DataFrame</b> — {@link #writeDataFrame}, {@link #readDataFrame}, {@link #searchDataFrame}</li>
 * </ul>
 *
 * <h2>Official-SDK switch (SPI only)</h2>
 * Built-in is Data API only. For self-hosted Mongo or {@code mongodb-driver-sync},
 * implement {@link MongoBackend} and register it under scheme {@code "mongo"}.
 *
 * <pre>{@code
 * try (Mongo m = Mongo.connect(dataApiUrl, apiKey, "Cluster0", "rag")) {
 *     df.toMongo(m, MongoOptions.builder().collection("docs").idColumn("id")
 *         .vectorColumn("emb").dim(384).build());
 * }
 * }</pre>
 *
 * @see <a href="https://www.mongodb.com/docs/atlas/api/data-api/">Atlas Data API</a>
 */
public class Mongo implements Closeable {

    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);

    private static final Map<String, MongoBackend> BACKENDS = new ConcurrentHashMap<>();
    static {
        reloadBackends();
    }

    private final HttpJson http;
    private final String url;
    private final String dataSource;
    private final String database;
    private final Duration timeout;
    private final boolean ownHttp;

    protected Mongo(HttpJson http, String url, String dataSource, String database,
                    Duration timeout, boolean ownHttp) {
        this.http = Objects.requireNonNull(http, "http");
        this.url = url;
        this.dataSource = dataSource == null ? "Cluster0" : dataSource;
        this.database = database == null ? "test" : database;
        this.timeout = timeout == null ? DEFAULT_TIMEOUT : timeout;
        this.ownHttp = ownHttp;
    }

    // ── SPI ───────────────────────────────────────────────────────────────

    public static void reloadBackends() {
        BACKENDS.clear();
        try {
            for (MongoBackend b : ServiceLoader.load(MongoBackend.class)) {
                registerBackend(b);
            }
        } catch (Throwable ignored) {}
    }

    public static void registerBackend(MongoBackend backend) {
        if (backend == null || backend.name() == null) return;
        BACKENDS.put(backend.name().toLowerCase(Locale.ROOT), backend);
        if (backend.aliases() != null) {
            for (String a : backend.aliases()) {
                if (a != null && !a.isBlank()) {
                    BACKENDS.put(a.toLowerCase(Locale.ROOT), backend);
                }
            }
        }
    }

    public static MongoBackend backend(String name) {
        if (name == null) return null;
        return BACKENDS.get(name.toLowerCase(Locale.ROOT));
    }

    // ── factories ─────────────────────────────────────────────────────────

    public static Mongo connect(String dataApiUrl, String apiKey) {
        return connect(dataApiUrl, apiKey, "Cluster0", "test", DEFAULT_TIMEOUT);
    }

    public static Mongo connect(String dataApiUrl, String apiKey,
                                 String dataSource, String database) {
        return connect(dataApiUrl, apiKey, dataSource, database, DEFAULT_TIMEOUT);
    }

    public static Mongo connect(String dataApiUrl, String apiKey,
                                 String dataSource, String database, Duration timeout) {
        Map<String, Object> cfg = new LinkedHashMap<>();
        cfg.put("url", dataApiUrl);
        if (apiKey != null) cfg.put("apiKey", apiKey);
        if (dataSource != null) cfg.put("dataSource", dataSource);
        if (database != null) cfg.put("database", database);
        if (timeout != null) cfg.put("timeout", timeout);
        return open(cfg);
    }

    public static Mongo connectUri(String uri) {
        Objects.requireNonNull(uri, "uri");
        String s = uri.trim();
        Map<String, Object> cfg = new LinkedHashMap<>();
        if (s.startsWith("mongo://") || s.startsWith("mongodb://") || s.startsWith("atlas://")) {
            // atlas://host/app?apiKey=...&dataSource=...&database=...
            String rest = s.substring(s.indexOf("://") + 3);
            String path = rest;
            String query = null;
            int q = rest.indexOf('?');
            if (q >= 0) {
                path = rest.substring(0, q);
                query = rest.substring(q + 1);
            }
            cfg.put("url", "https://" + path);
            if (query != null) parseQuery(query, cfg);
        } else {
            cfg.put("url", s);
        }
        return open(cfg);
    }

    public static Mongo open(Map<String, Object> config) {
        Map<String, Object> cfg = config == null ? Map.of() : config;
        MongoBackend plugin = BACKENDS.get("mongo");
        if (plugin == null) plugin = BACKENDS.get("mongodb");
        if (plugin == null) plugin = BACKENDS.get("atlas");
        if (plugin != null) return plugin.open(cfg);
        return openBuiltin(cfg);
    }

    public static Mongo openBuiltin(Map<String, Object> cfg) {
        String url = str(cfg, "url", null);
        if (url == null) throw new MongoException("mongo requires url (Data API base)");
        String apiKey = str(cfg, "apiKey", str(cfg, "api_key", null));
        String dataSource = str(cfg, "dataSource", str(cfg, "cluster", "Cluster0"));
        String database = str(cfg, "database", str(cfg, "db", "test"));
        Duration timeout = duration(cfg.get("timeout"));
        HttpJson.Builder hb = HttpJson.builder(url)
            .backend("mongo")
            .timeout(timeout)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");
        if (apiKey != null && !apiKey.isEmpty()) {
            hb.header("api-key", apiKey);
        }
        return new Mongo(hb.build(), url, dataSource, database, timeout, true);
    }

    public static Builder builder(String dataApiUrl) {
        return new Builder(dataApiUrl);
    }

    // ── accessors ─────────────────────────────────────────────────────────

    public HttpJson http() { return http; }
    public String url() { return url; }
    public String dataSource() { return dataSource; }
    public String database() { return database; }
    public Duration timeout() { return timeout; }

    /** Return a view bound to a collection (same transport). */
    public Mongo withCollectionDefaults(String collection) {
        // collection is per-operation; kept for API familiarity
        return this;
    }

    // ── Data API actions ──────────────────────────────────────────────────

    public Object action(String actionName, String collection, Map<String, Object> extra) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("dataSource", dataSource);
        body.put("database", database);
        body.put("collection", collection);
        if (extra != null) body.putAll(extra);
        try {
            return http.post("/action/" + actionName, body);
        } catch (VectorStoreException e) {
            throw new MongoException(e.getMessage(), e, e.status(), actionName);
        }
    }

    public Object insertOne(String collection, Map<String, Object> document) {
        return action("insertOne", collection, HttpJson.mapOf("document", document));
    }

    public Object insertMany(String collection, List<Map<String, Object>> documents) {
        return action("insertMany", collection, HttpJson.mapOf("documents", documents));
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> find(String collection, Map<String, Object> filter,
                                           Map<String, Object> projection, int limit, int skip) {
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("filter", filter == null ? Map.of() : filter);
        if (projection != null) extra.put("projection", projection);
        if (limit > 0) extra.put("limit", limit);
        if (skip > 0) extra.put("skip", skip);
        Object resp = action("find", collection, extra);
        List<Object> docs = HttpJson.asList(HttpJson.dig(resp, "documents"));
        List<Map<String, Object>> out = new ArrayList<>(docs.size());
        for (Object d : docs) out.add(HttpJson.asMap(d));
        return out;
    }

    public Map<String, Object> findOne(String collection, Map<String, Object> filter) {
        List<Map<String, Object>> docs = find(collection, filter, null, 1, 0);
        return docs.isEmpty() ? null : docs.get(0);
    }

    public Object updateOne(String collection, Map<String, Object> filter,
                             Map<String, Object> update, boolean upsert) {
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("filter", filter == null ? Map.of() : filter);
        extra.put("update", update);
        extra.put("upsert", upsert);
        return action("updateOne", collection, extra);
    }

    public Object updateMany(String collection, Map<String, Object> filter,
                              Map<String, Object> update, boolean upsert) {
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("filter", filter == null ? Map.of() : filter);
        extra.put("update", update);
        extra.put("upsert", upsert);
        return action("updateMany", collection, extra);
    }

    public Object replaceOne(String collection, Map<String, Object> filter,
                              Map<String, Object> replacement, boolean upsert) {
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("filter", filter == null ? Map.of() : filter);
        extra.put("replacement", replacement);
        extra.put("upsert", upsert);
        return action("replaceOne", collection, extra);
    }

    public Object deleteOne(String collection, Map<String, Object> filter) {
        return action("deleteOne", collection, HttpJson.mapOf("filter", filter == null ? Map.of() : filter));
    }

    public Object deleteMany(String collection, Map<String, Object> filter) {
        return action("deleteMany", collection, HttpJson.mapOf("filter", filter == null ? Map.of() : filter));
    }

    public long countDocuments(String collection, Map<String, Object> filter) {
        try {
            // Prefer aggregate $count for broader Data API tier support
            List<Map<String, Object>> pipeline = new ArrayList<>();
            if (filter != null && !filter.isEmpty()) {
                pipeline.add(HttpJson.mapOf("$match", filter));
            }
            pipeline.add(Map.of("$count", "n"));
            Object resp = action("aggregate", collection, HttpJson.mapOf("pipeline", pipeline));
            List<Object> docs = HttpJson.asList(HttpJson.dig(resp, "documents"));
            if (!docs.isEmpty()) {
                return HttpJson.asLong(HttpJson.asMap(docs.get(0)).get("n"), 0L);
            }
            return 0L;
        } catch (MongoException e) {
            return -1L;
        }
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> aggregate(String collection, List<Map<String, Object>> pipeline) {
        Object resp = action("aggregate", collection, HttpJson.mapOf("pipeline", pipeline));
        List<Object> docs = HttpJson.asList(HttpJson.dig(resp, "documents"));
        List<Map<String, Object>> out = new ArrayList<>(docs.size());
        for (Object d : docs) out.add(HttpJson.asMap(d));
        return out;
    }

    public void ensureCollection(String collection, int dim, String idField, String vectorPath) {
        Map<String, Object> doc = new LinkedHashMap<>();
        doc.put(idField == null ? "_id" : idField, "__vectorstore_init__");
        doc.put(vectorPath == null ? "embedding" : vectorPath, zeroVector(Math.max(dim, 1)));
        doc.put("_init", true);
        try {
            insertOne(collection, doc);
            deleteOne(collection, HttpJson.mapOf(
                idField == null ? "_id" : idField, "__vectorstore_init__"));
        } catch (MongoException e) {
            // collection may already exist / duplicate key — fine
        }
    }

    public void dropCollection(String collection) {
        deleteMany(collection, Map.of());
    }

    // ── vector search ─────────────────────────────────────────────────────

    public VectorSearchResult vectorSearch(String collection, VectorQuery query,
                                            String vectorPath, String indexName,
                                            String idField, VectorMetric metric) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        int k = query.topK();
        int numCandidates = query.option("num_candidates", Math.max(k * 10, 100));
        String vp = vectorPath == null ? "embedding" : vectorPath;
        String idx = indexName == null ? "vector_index" : indexName;
        String idF = idField == null ? "_id" : idField;
        VectorMetric met = metric == null ? VectorMetric.COSINE : metric;

        Map<String, Object> vectorSearch = new LinkedHashMap<>();
        vectorSearch.put("index", idx);
        vectorSearch.put("path", vp);
        vectorSearch.put("queryVector", HttpJson.toDoubleList(query.vector()));
        vectorSearch.put("numCandidates", numCandidates);
        vectorSearch.put("limit", k);
        if (query.filter() instanceof Map<?, ?> f) {
            vectorSearch.put("filter", f);
        }

        List<Map<String, Object>> pipeline = new ArrayList<>();
        pipeline.add(HttpJson.mapOf("$vectorSearch", vectorSearch));
        pipeline.add(HttpJson.mapOf("$addFields", HttpJson.mapOf(
            "score", HttpJson.mapOf("$meta", "vectorSearchScore")
        )));
        if (!query.includeVector()) {
            pipeline.add(HttpJson.mapOf("$project", HttpJson.mapOf(vp, 0)));
        }

        List<Map<String, Object>> docs = aggregate(collection, pipeline);
        List<VectorHit> hits = new ArrayList<>(docs.size());
        for (Map<String, Object> m : docs) {
            String id = HttpJson.asString(m.get(idF));
            float score = HttpJson.asFloat(m.get("score"), 0f);
            float[] vec = null;
            if (query.includeVector()) {
                vec = HttpJson.asFloatArray(m.get(vp));
            }
            Map<String, Object> payload = new LinkedHashMap<>();
            if (query.includePayload()) {
                for (Map.Entry<String, Object> e : m.entrySet()) {
                    String key = e.getKey();
                    if (idF.equals(key) || vp.equals(key) || "score".equals(key)) continue;
                    payload.put(key, e.getValue());
                }
            }
            Float distance = met == VectorMetric.COSINE ? (1f - score) : -score;
            hits.add(new VectorHit(id, -1L, false, score, distance, vec, payload));
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    public void upsertRecords(String collection, Collection<VectorRecord> records,
                               String idField, String vectorPath) {
        if (records == null || records.isEmpty()) return;
        String idF = idField == null ? "_id" : idField;
        String vp = vectorPath == null ? "embedding" : vectorPath;
        for (VectorRecord r : records) {
            String id = r.resolvedId();
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put(idF, id);
            doc.put(vp, HttpJson.toDoubleList(r.vector()));
            if (r.payload() != null) {
                for (Map.Entry<String, Object> e : r.payload().entrySet()) {
                    if (idF.equals(e.getKey()) || vp.equals(e.getKey())) continue;
                    doc.put(e.getKey(), e.getValue());
                }
            }
            updateOne(collection, HttpJson.mapOf(idF, id),
                HttpJson.mapOf("$set", doc), true);
        }
    }

    // ── DataFrame I/O ─────────────────────────────────────────────────────

    public int writeDataFrame(DataFrame df, MongoOptions options) {
        Objects.requireNonNull(df, "df");
        MongoOptions opt = options == null ? MongoOptions.defaults() : options;
        String collection = opt.collection();

        long existing = countDocuments(collection, Map.of());
        if (opt.ifExists() == MongoOptions.IfExists.REPLACE) {
            dropCollection(collection);
        } else if (opt.ifExists() == MongoOptions.IfExists.FAIL && existing > 0) {
            throw new MongoException("collection non-empty: " + collection, -1, "writeDataFrame");
        } else if (opt.ifExists() == MongoOptions.IfExists.SKIP && existing > 0) {
            return 0;
        }

        if (opt.ensureCollection()) {
            ensureCollection(collection, opt.dim(), opt.idField(), opt.vectorPath());
        }

        String vectorCol = resolveVectorColumn(df, opt);
        List<String> payloadCols = resolvePayloadColumns(df, opt, vectorCol);
        String idCol = resolveIdColumn(df, opt);
        int written = 0;

        for (int r = 0; r < df.rowCount(); r++) {
            Map<String, Object> doc = new LinkedHashMap<>();
            Object idv = idCol != null ? df.get(r, idCol) : r;
            String id = idv == null ? String.valueOf(r) : String.valueOf(idv);
            doc.put(opt.idField(), id);

            if (vectorCol != null) {
                float[] vec = VectorStore.toFloatArray(df.get(r, vectorCol));
                if (vec != null) {
                    doc.put(opt.vectorPath(), HttpJson.toDoubleList(vec));
                } else if (!opt.includeNulls()) {
                    continue;
                }
            }
            for (String pn : payloadCols) {
                Object v = df.get(r, pn);
                if (v == null && !opt.includeNulls()) continue;
                doc.put(pn, cellToJson(v));
            }
            updateOne(collection, HttpJson.mapOf(opt.idField(), id),
                HttpJson.mapOf("$set", doc), true);
            written++;
        }
        return written;
    }

    public DataFrame readDataFrame(MongoOptions options) {
        MongoOptions opt = options == null ? MongoOptions.defaults() : options;
        int limit = opt.limit() <= 0 ? 100_000 : opt.limit();
        List<Map<String, Object>> docs = find(
            opt.collection(),
            opt.filter() == null ? Map.of() : opt.filter(),
            opt.projection(),
            limit,
            0
        );
        return mapsToDataFrame(docs, opt);
    }

    public DataFrame searchDataFrame(float[] query, int topK, MongoOptions options) {
        MongoOptions opt = options == null ? MongoOptions.defaults() : options;
        VectorQuery vq = VectorQuery.of(query, topK);
        return vectorSearch(opt.collection(), vq, opt.vectorPath(), opt.indexName(),
            opt.idField(), opt.metric()).toDataFrame();
    }

    public VectorStore asVectorStore(String collection, int dim, VectorMetric metric) {
        return MongoAtlasVectorStore
            .builder(url)
            .dataSource(dataSource)
            .database(database)
            .collection(collection)
            .dim(dim)
            .metric(metric == null ? VectorMetric.COSINE : metric)
            .timeout(timeout)
            .build();
    }

    @Override
    public void close() {
        if (ownHttp) http.close();
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static List<Double> zeroVector(int d) {
        List<Double> v = new ArrayList<>(d);
        for (int i = 0; i < d; i++) v.add(0.0);
        return v;
    }

    private static DataFrame mapsToDataFrame(List<Map<String, Object>> rows, MongoOptions opt) {
        DataFrame df = DataFrame.create();
        if (rows == null || rows.isEmpty()) {
            df.addColumn(opt.idField(), Column.DType.STRING);
            if (opt.includeVector()) df.addColumn(opt.vectorPath(), Column.DType.VECTOR);
            return df;
        }
        List<String> keys = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            for (String k : row.keySet()) {
                if (!keys.contains(k)) keys.add(k);
            }
        }
        for (String k : keys) {
            if (k.equals(opt.vectorPath())) {
                if (opt.includeVector()) df.addColumn(k, Column.DType.VECTOR);
            } else {
                df.addColumn(k, Column.DType.STRING);
            }
        }
        for (Map<String, Object> row : rows) {
            int r = df.addEmptyRow();
            for (String k : keys) {
                if (k.equals(opt.vectorPath()) && !opt.includeVector()) continue;
                Object v = row.get(k);
                if (k.equals(opt.vectorPath())) {
                    df.set(r, k, HttpJson.asFloatArray(v));
                } else {
                    df.set(r, k, v == null ? null : String.valueOf(v));
                }
            }
        }
        return df;
    }

    private static String resolveIdColumn(DataFrame df, MongoOptions opt) {
        if (opt.idColumn() != null && df.hasColumn(opt.idColumn())) return opt.idColumn();
        if (df.hasColumn("id")) return "id";
        if (df.hasColumn("_id")) return "_id";
        return null;
    }

    private static String resolveVectorColumn(DataFrame df, MongoOptions opt) {
        if (opt.vectorColumn() != null && df.hasColumn(opt.vectorColumn())) return opt.vectorColumn();
        if (df.hasColumn(opt.vectorPath())) return opt.vectorPath();
        if (df.hasColumn("emb")) return "emb";
        if (df.hasColumn("embedding")) return "embedding";
        if (df.hasColumn("vector")) return "vector";
        for (int c = 0; c < df.columnCount(); c++) {
            Column col = df.column(c);
            if (col.dtype() == Column.DType.VECTOR || col.dtype() == Column.DType.EMBEDDING) {
                return col.name();
            }
        }
        return null;
    }

    private static List<String> resolvePayloadColumns(DataFrame df, MongoOptions opt, String vectorCol) {
        if (opt.payloadColumns() != null && !opt.payloadColumns().isEmpty()) {
            return opt.payloadColumns();
        }
        List<String> out = new ArrayList<>();
        String idCol = resolveIdColumn(df, opt);
        for (int c = 0; c < df.columnCount(); c++) {
            String n = df.column(c).name();
            if (n.equals(vectorCol)) continue;
            if (idCol != null && n.equals(idCol)) continue;
            if (n.equals(opt.vectorPath()) || n.equals(opt.idField())) continue;
            out.add(n);
        }
        return out;
    }

    private static Object cellToJson(Object v) {
        if (v == null) return null;
        if (v instanceof float[] f) return HttpJson.toDoubleList(f);
        if (v instanceof double[] d) {
            List<Double> list = new ArrayList<>(d.length);
            for (double x : d) list.add(x);
            return list;
        }
        if (v instanceof Number || v instanceof Boolean || v instanceof String) return v;
        return String.valueOf(v);
    }

    private static String str(Map<String, Object> cfg, String key, String def) {
        Object v = cfg.get(key);
        if (v == null) return def;
        String s = String.valueOf(v);
        return s.isEmpty() ? def : s;
    }

    private static Duration duration(Object v) {
        if (v instanceof Duration d) return d;
        if (v instanceof Number n) return Duration.ofMillis(n.longValue());
        if (v instanceof String s) {
            try { return Duration.ofMillis(Long.parseLong(s.trim())); } catch (Exception ignored) {}
        }
        return DEFAULT_TIMEOUT;
    }

    private static void parseQuery(String query, Map<String, Object> cfg) {
        for (String pair : query.split("&")) {
            if (pair.isEmpty()) continue;
            int eq = pair.indexOf('=');
            String k = eq < 0 ? pair : pair.substring(0, eq);
            String v = eq < 0 ? "" : pair.substring(eq + 1);
            try {
                k = java.net.URLDecoder.decode(k, java.nio.charset.StandardCharsets.UTF_8);
                v = java.net.URLDecoder.decode(v, java.nio.charset.StandardCharsets.UTF_8);
            } catch (Exception ignored) {}
            cfg.put(k, v);
        }
    }

    public static final class Builder {
        private final String url;
        private String apiKey;
        private String dataSource = "Cluster0";
        private String database = "test";
        private Duration timeout = DEFAULT_TIMEOUT;

        Builder(String url) { this.url = url; }

        public Builder apiKey(String k) { this.apiKey = k; return this; }
        public Builder dataSource(String ds) { this.dataSource = ds; return this; }
        public Builder database(String db) { this.database = db; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }

        public Mongo build() {
            Map<String, Object> cfg = new LinkedHashMap<>();
            cfg.put("url", url);
            if (apiKey != null) cfg.put("apiKey", apiKey);
            if (dataSource != null) cfg.put("dataSource", dataSource);
            if (database != null) cfg.put("database", database);
            if (timeout != null) cfg.put("timeout", timeout);
            return open(cfg);
        }
    }
}
