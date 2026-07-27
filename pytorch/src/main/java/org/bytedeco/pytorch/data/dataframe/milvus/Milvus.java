package org.bytedeco.pytorch.data.dataframe.milvus;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.data.dataframe.vectorstore.http.HttpJson;

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
 * Full-featured Milvus client for DataFrame I/O — pure REST v2 via {@link HttpJson},
 * no {@code milvus-sdk-java} dependency.
 *
 * <h2>Coverage (pymilvus / milvus-sdk REST parity subset)</h2>
 * <ul>
 *   <li><b>Connection</b> — connect / connectUri / token / dbName / close</li>
 *   <li><b>Collections</b> — has / create / drop / load / list / describe</li>
 *   <li><b>Entities</b> — insert / upsert / delete / query / get / get_count</li>
 *   <li><b>Search</b> — search / searchBatch (multi-vector)</li>
 *   <li><b>DataFrame</b> — {@link #writeDataFrame}, {@link #readDataFrame}, {@link #searchDataFrame}</li>
 * </ul>
 *
 * <h2>Official-SDK switch (SPI only)</h2>
 * Implement {@link MilvusBackend} and register via {@code META-INF/services} or
 * {@link #registerBackend}. A backend named {@code "milvus"} overrides this built-in.
 *
 * <pre>{@code
 * try (Milvus m = Milvus.connect("http://localhost:9091", "root:Milvus")) {
 *     m.createCollection("docs", 384, VectorMetric.COSINE);
 *     df.toMilvus(m, MilvusOptions.builder().collection("docs").idColumn("id")
 *         .vectorColumn("emb").dim(384).build());
 *     DataFrame back = DataFrame.readMilvus(m, MilvusOptions.collection("docs", 384));
 * }
 * }</pre>
 *
 * @see <a href="https://milvus.io/api-reference/restful/v2.4.x/About.md">Milvus RESTful v2</a>
 */
public class Milvus implements Closeable {

    public static final int DEFAULT_REST_PORT = 9091;
    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);

    private static final Map<String, MilvusBackend> BACKENDS = new ConcurrentHashMap<>();
    static {
        reloadBackends();
    }

    private final HttpJson http;
    private final String url;
    private final String dbName;
    private final Duration timeout;
    private final boolean ownHttp;

    protected Milvus(HttpJson http, String url, String dbName, Duration timeout, boolean ownHttp) {
        this.http = Objects.requireNonNull(http, "http");
        this.url = url;
        this.dbName = dbName == null ? "default" : dbName;
        this.timeout = timeout == null ? DEFAULT_TIMEOUT : timeout;
        this.ownHttp = ownHttp;
    }

    // ── SPI ───────────────────────────────────────────────────────────────

    public static void reloadBackends() {
        BACKENDS.clear();
        try {
            for (MilvusBackend b : ServiceLoader.load(MilvusBackend.class)) {
                registerBackend(b);
            }
        } catch (Throwable ignored) {}
    }

    public static void registerBackend(MilvusBackend backend) {
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

    public static MilvusBackend backend(String name) {
        if (name == null) return null;
        return BACKENDS.get(name.toLowerCase(Locale.ROOT));
    }

    // ── factories ─────────────────────────────────────────────────────────

    public static Milvus connect() {
        return connect("http://localhost:" + DEFAULT_REST_PORT);
    }

    public static Milvus connect(String url) {
        return connect(url, null, "default", DEFAULT_TIMEOUT);
    }

    public static Milvus connect(String url, String token) {
        return connect(url, token, "default", DEFAULT_TIMEOUT);
    }

    public static Milvus connect(String url, String token, String dbName) {
        return connect(url, token, dbName, DEFAULT_TIMEOUT);
    }

    public static Milvus connect(String url, String token, String dbName, Duration timeout) {
        Map<String, Object> cfg = new LinkedHashMap<>();
        cfg.put("url", url);
        if (token != null) cfg.put("token", token);
        if (dbName != null) cfg.put("dbName", dbName);
        if (timeout != null) cfg.put("timeout", timeout);
        return open(cfg);
    }

    /** Parse {@code milvus://host:port[/db]?token=...} or plain HTTP URL. */
    public static Milvus connectUri(String uri) {
        Objects.requireNonNull(uri, "uri");
        String s = uri.trim();
        Map<String, Object> cfg = new LinkedHashMap<>();
        if (s.startsWith("milvus://") || s.startsWith("zilliz://")) {
            String rest = s.substring(s.indexOf("://") + 3);
            String path = rest;
            String query = null;
            int q = rest.indexOf('?');
            if (q >= 0) {
                path = rest.substring(0, q);
                query = rest.substring(q + 1);
            }
            String hostPort = path;
            String db = null;
            int slash = path.indexOf('/');
            if (slash >= 0) {
                hostPort = path.substring(0, slash);
                db = path.substring(slash + 1);
                if (db.isEmpty()) db = null;
            }
            cfg.put("url", "http://" + hostPort);
            if (db != null) cfg.put("dbName", db);
            if (query != null) parseQuery(query, cfg);
        } else {
            cfg.put("url", s);
        }
        return open(cfg);
    }

    /**
     * Open via SPI if a backend is registered for scheme {@code milvus}/{@code zilliz},
     * otherwise built-in REST client.
     */
    public static Milvus open(Map<String, Object> config) {
        Map<String, Object> cfg = config == null ? Map.of() : config;
        MilvusBackend plugin = BACKENDS.get("milvus");
        if (plugin == null) plugin = BACKENDS.get("zilliz");
        if (plugin != null) {
            return plugin.open(cfg);
        }
        return openBuiltin(cfg);
    }

    public static Milvus openBuiltin(Map<String, Object> cfg) {
        String url = str(cfg, "url", "http://localhost:" + DEFAULT_REST_PORT);
        String token = str(cfg, "token", str(cfg, "apiKey", str(cfg, "api_key", null)));
        String dbName = str(cfg, "dbName", str(cfg, "database", "default"));
        Duration timeout = duration(cfg.get("timeout"));
        HttpJson.Builder hb = HttpJson.builder(url).backend("milvus").timeout(timeout);
        if (token != null && !token.isEmpty()) {
            hb.header("Authorization", "Bearer " + token);
        }
        return new Milvus(hb.build(), url, dbName, timeout, true);
    }

    public static Builder builder(String url) {
        return new Builder(url);
    }

    // ── accessors ─────────────────────────────────────────────────────────

    public HttpJson http() { return http; }
    public String url() { return url; }
    public String dbName() { return dbName; }
    public Duration timeout() { return timeout; }

    // ── collections ───────────────────────────────────────────────────────

    public boolean hasCollection(String collection) {
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        Object resp = post("/v2/vectordb/collections/has", body, "collections/has");
        Object data = HttpJson.dig(resp, "data");
        if (data instanceof Map<?, ?> m && m.containsKey("has")) {
            return Boolean.TRUE.equals(m.get("has"));
        }
        if (data instanceof Boolean bo) return bo;
        return Boolean.TRUE.equals(HttpJson.dig(resp, "data", "has"));
    }

    public void createCollection(String collection, int dim, VectorMetric metric) {
        createCollection(collection, dim, metric, "id", "vector", "AUTOINDEX");
    }

    public void createCollection(String collection, int dim, VectorMetric metric,
                                  String idField, String vectorField, String indexType) {
        if (dim <= 0) throw new MilvusException("dim required to create collection", -1, "collections/create");
        List<Map<String, Object>> fields = new ArrayList<>();
        fields.add(HttpJson.mapOf(
            "fieldName", idField == null ? "id" : idField,
            "dataType", "Int64",
            "isPrimary", true,
            "autoID", false
        ));
        fields.add(HttpJson.mapOf(
            "fieldName", vectorField == null ? "vector" : vectorField,
            "dataType", "FloatVector",
            "elementCount", dim
        ));
        Map<String, Object> schema = HttpJson.mapOf(
            "autoID", false,
            "enableDynamicField", true,
            "fields", fields
        );
        Map<String, Object> indexParams = HttpJson.mapOf(
            "indexName", (vectorField == null ? "vector" : vectorField) + "_idx",
            "fieldName", vectorField == null ? "vector" : vectorField,
            "metricType", (metric == null ? VectorMetric.L2 : metric).milvus(),
            "indexType", indexType == null ? "AUTOINDEX" : indexType,
            "params", HttpJson.mapOf()
        );
        Map<String, Object> create = baseBody();
        create.put("collectionName", collection);
        create.put("schema", schema);
        create.put("indexParams", List.of(indexParams));
        post("/v2/vectordb/collections/create", create, "collections/create");
        loadCollection(collection);
    }

    public void ensureCollection(String collection, int dim, VectorMetric metric) {
        ensureCollection(collection, dim, metric, "id", "vector", "AUTOINDEX");
    }

    public void ensureCollection(String collection, int dim, VectorMetric metric,
                                  String idField, String vectorField, String indexType) {
        if (hasCollection(collection)) {
            try { loadCollection(collection); } catch (MilvusException ignored) {}
            return;
        }
        createCollection(collection, dim, metric, idField, vectorField, indexType);
    }

    public void dropCollection(String collection) {
        try {
            Map<String, Object> body = baseBody();
            body.put("collectionName", collection);
            post("/v2/vectordb/collections/drop", body, "collections/drop");
        } catch (MilvusException e) {
            String msg = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT);
            if (e.status() != 404 && !msg.contains("not found") && !msg.contains("doesn't exist")) {
                throw e;
            }
        }
    }

    public void loadCollection(String collection) {
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        post("/v2/vectordb/collections/load", body, "collections/load");
    }

    public void releaseCollection(String collection) {
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        post("/v2/vectordb/collections/release", body, "collections/release");
    }

    @SuppressWarnings("unchecked")
    public List<String> listCollections() {
        Object resp = post("/v2/vectordb/collections/list", baseBody(), "collections/list");
        Object data = HttpJson.dig(resp, "data");
        List<String> out = new ArrayList<>();
        if (data instanceof List<?> list) {
            for (Object o : list) {
                if (o instanceof String s) out.add(s);
                else if (o instanceof Map<?, ?> m) {
                    Object n = m.get("name");
                    if (n == null) n = m.get("collectionName");
                    if (n != null) out.add(String.valueOf(n));
                }
            }
        } else if (data instanceof Map<?, ?> m) {
            Object names = m.get("collections");
            if (names instanceof List<?> list) {
                for (Object o : list) out.add(String.valueOf(o));
            }
        }
        return out;
    }

    public Object describeCollection(String collection) {
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        return post("/v2/vectordb/collections/describe", body, "collections/describe");
    }

    // ── entities ──────────────────────────────────────────────────────────

    public long count(String collection) {
        try {
            Map<String, Object> body = baseBody();
            body.put("collectionName", collection);
            Object resp = post("/v2/vectordb/entities/get_count", body, "entities/get_count");
            Object n = HttpJson.dig(resp, "data", "count");
            if (n == null) n = HttpJson.dig(resp, "data");
            return HttpJson.asLong(n, -1L);
        } catch (MilvusException e) {
            return -1L;
        }
    }

    public void insert(String collection, List<Map<String, Object>> rows) {
        if (rows == null || rows.isEmpty()) return;
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        body.put("data", rows);
        post("/v2/vectordb/entities/insert", body, "entities/insert");
    }

    public void upsert(String collection, List<Map<String, Object>> rows) {
        if (rows == null || rows.isEmpty()) return;
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        body.put("data", rows);
        post("/v2/vectordb/entities/upsert", body, "entities/upsert");
    }

    public void upsertRecords(String collection, Collection<VectorRecord> records,
                               String idField, String vectorField, int batchSize) {
        if (records == null || records.isEmpty()) return;
        String idF = idField == null ? "id" : idField;
        String vecF = vectorField == null ? "vector" : vectorField;
        int chunk = Math.max(1, batchSize);
        List<VectorRecord> list = records instanceof List
            ? (List<VectorRecord>) records
            : new ArrayList<>(records);
        for (int i = 0; i < list.size(); i += chunk) {
            List<VectorRecord> slice = list.subList(i, Math.min(i + chunk, list.size()));
            List<Map<String, Object>> rows = new ArrayList<>(slice.size());
            for (VectorRecord r : slice) {
                Map<String, Object> row = new LinkedHashMap<>();
                long id = r.hasNumericId() ? r.numericId() : hashId(r.resolvedId());
                row.put(idF, id);
                row.put(vecF, HttpJson.toDoubleList(r.vector()));
                if (r.payload() != null) {
                    for (Map.Entry<String, Object> e : r.payload().entrySet()) {
                        if (idF.equals(e.getKey()) || vecF.equals(e.getKey())) continue;
                        row.put(e.getKey(), e.getValue());
                    }
                    if (r.id() != null) row.putIfAbsent("_str_id", r.id());
                }
                rows.add(row);
            }
            upsert(collection, rows);
        }
    }

    public void delete(String collection, String filter) {
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        body.put("filter", filter == null ? "" : filter);
        post("/v2/vectordb/entities/delete", body, "entities/delete");
    }

    public void deleteByIds(String collection, Collection<String> ids, String idField) {
        if (ids == null || ids.isEmpty()) return;
        String idF = idField == null ? "id" : idField;
        StringBuilder expr = new StringBuilder(idF).append(" in [");
        boolean first = true;
        for (String id : ids) {
            if (id == null) continue;
            if (!first) expr.append(',');
            first = false;
            try {
                expr.append(Long.parseLong(id));
            } catch (NumberFormatException e) {
                expr.append(hashId(id));
            }
        }
        expr.append(']');
        if (first) return;
        delete(collection, expr.toString());
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> query(String collection, String filter,
                                            List<String> outputFields, int limit, long offset) {
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        body.put("filter", filter == null ? "" : filter);
        body.put("limit", Math.max(1, limit));
        if (offset > 0) body.put("offset", offset);
        body.put("outputFields", outputFields == null || outputFields.isEmpty()
            ? List.of("*") : outputFields);
        Object resp = post("/v2/vectordb/entities/query", body, "entities/query");
        List<Object> rows = HttpJson.asList(HttpJson.dig(resp, "data"));
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Object row : rows) out.add(HttpJson.asMap(row));
        return out;
    }

    public List<Map<String, Object>> get(String collection, Collection<String> ids, String idField) {
        if (ids == null || ids.isEmpty()) return List.of();
        String idF = idField == null ? "id" : idField;
        StringBuilder expr = new StringBuilder(idF).append(" in [");
        boolean first = true;
        for (String id : ids) {
            if (id == null) continue;
            if (!first) expr.append(',');
            first = false;
            try {
                expr.append(Long.parseLong(id));
            } catch (NumberFormatException e) {
                expr.append(hashId(id));
            }
        }
        expr.append(']');
        if (first) return List.of();
        return query(collection, expr.toString(), List.of("*"), ids.size(), 0);
    }

    // ── search ────────────────────────────────────────────────────────────

    public VectorSearchResult search(String collection, VectorQuery query,
                                      String vectorField, VectorMetric metric) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        String vecF = vectorField == null ? "vector" : vectorField;
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        body.put("data", List.of(HttpJson.toDoubleList(query.vector())));
        body.put("annsField", query.vectorName() != null ? query.vectorName() : vecF);
        body.put("limit", query.topK());
        if (query.includePayload()) {
            body.put("outputFields", List.of("*"));
        } else {
            List<String> outputs = new ArrayList<>();
            outputs.add("id");
            if (query.includeVector()) outputs.add(vecF);
            body.put("outputFields", outputs);
        }
        Integer nprobe = query.option("nprobe", null);
        Integer ef = query.option("ef", null);
        Map<String, Object> params = new LinkedHashMap<>();
        if (nprobe != null) params.put("nprobe", nprobe);
        if (ef != null) params.put("ef", ef);
        body.put("searchParams", HttpJson.mapOf(
            "metricType", (metric == null ? VectorMetric.L2 : metric).milvus(),
            "params", params
        ));
        if (query.filter() instanceof String s && !s.isBlank()) {
            body.put("filter", s);
        }
        Object resp = post("/v2/vectordb/entities/search", body, "entities/search");
        Object data = HttpJson.dig(resp, "data");
        List<Object> outer = HttpJson.asList(data);
        List<Object> inner;
        if (!outer.isEmpty() && outer.get(0) instanceof List<?>) {
            inner = HttpJson.asList(outer.get(0));
        } else {
            inner = outer;
        }
        List<VectorHit> hits = parseHits(inner, "id", vecF, query.includeVector(), query.includePayload());
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    public List<VectorSearchResult> searchBatch(String collection, List<VectorQuery> queries,
                                                 String vectorField, VectorMetric metric) {
        if (queries == null || queries.isEmpty()) return List.of();
        boolean uniform = true;
        int topK = queries.get(0).topK();
        for (VectorQuery q : queries) {
            if (q.topK() != topK) { uniform = false; break; }
        }
        if (!uniform) {
            List<VectorSearchResult> out = new ArrayList<>(queries.size());
            for (VectorQuery q : queries) out.add(search(collection, q, vectorField, metric));
            return out;
        }
        String vecF = vectorField == null ? "vector" : vectorField;
        List<List<Double>> data = new ArrayList<>(queries.size());
        for (VectorQuery q : queries) data.add(HttpJson.toDoubleList(q.vector()));
        Map<String, Object> body = baseBody();
        body.put("collectionName", collection);
        body.put("data", data);
        body.put("annsField", vecF);
        body.put("limit", topK);
        body.put("outputFields", List.of("*"));
        body.put("searchParams", HttpJson.mapOf(
            "metricType", (metric == null ? VectorMetric.L2 : metric).milvus(),
            "params", Map.of()
        ));
        Object resp = post("/v2/vectordb/entities/search", body, "entities/search");
        List<Object> outer = HttpJson.asList(HttpJson.dig(resp, "data"));
        List<VectorSearchResult> out = new ArrayList<>(queries.size());
        boolean nested = !outer.isEmpty() && outer.get(0) instanceof List<?>;
        if (!nested) {
            out.add(new VectorSearchResult(parseHits(outer, "id", vecF, false, true)));
            while (out.size() < queries.size()) out.add(VectorSearchResult.empty());
            return out;
        }
        for (Object block : outer) {
            out.add(new VectorSearchResult(parseHits(HttpJson.asList(block), "id", vecF, false, true)));
        }
        while (out.size() < queries.size()) out.add(VectorSearchResult.empty());
        return out;
    }

    // ── DataFrame I/O ─────────────────────────────────────────────────────

    /**
     * Write DataFrame rows as Milvus entities (upsert).
     *
     * @return number of rows written
     */
    public int writeDataFrame(DataFrame df, MilvusOptions options) {
        Objects.requireNonNull(df, "df");
        MilvusOptions opt = options == null ? MilvusOptions.defaults() : options;
        String collection = opt.collection();

        if (opt.ifExists() == MilvusOptions.IfExists.REPLACE) {
            dropCollection(collection);
        } else if (opt.ifExists() == MilvusOptions.IfExists.FAIL && hasCollection(collection)) {
            throw new MilvusException("collection exists: " + collection, -1, "writeDataFrame");
        } else if (opt.ifExists() == MilvusOptions.IfExists.SKIP && hasCollection(collection)
                && count(collection) > 0) {
            return 0;
        }

        String vectorCol = resolveVectorColumn(df, opt);
        int dim = opt.dim();
        if (dim <= 0 && vectorCol != null) {
            dim = inferDim(df, vectorCol);
        }
        if (opt.ensureCollection()) {
            ensureCollection(collection, dim, opt.metric(),
                opt.idField(), opt.vectorField(), opt.indexType());
        }

        List<String> payloadCols = resolvePayloadColumns(df, opt, vectorCol);
        String idCol = resolveIdColumn(df, opt);
        int written = 0;
        int batch = opt.batchSize();
        List<Map<String, Object>> rows = new ArrayList<>(batch);

        for (int r = 0; r < df.rowCount(); r++) {
            Map<String, Object> row = new LinkedHashMap<>();
            Object idv = idCol != null ? df.get(r, idCol) : r;
            long id;
            if (idv instanceof Number n) id = n.longValue();
            else if (idv != null) {
                try { id = Long.parseLong(String.valueOf(idv)); }
                catch (NumberFormatException e) { id = hashId(String.valueOf(idv)); }
            } else {
                id = r;
            }
            row.put(opt.idField(), id);
            if (idv != null && !(idv instanceof Number)) {
                row.put("_str_id", String.valueOf(idv));
            }

            if (vectorCol != null) {
                float[] vec = VectorStore.toFloatArray(df.get(r, vectorCol));
                if (vec != null) {
                    row.put(opt.vectorField(), HttpJson.toDoubleList(vec));
                } else if (!opt.includeNulls()) {
                    continue;
                }
            }

            for (String pn : payloadCols) {
                Object v = df.get(r, pn);
                if (v == null && !opt.includeNulls()) continue;
                row.put(pn, cellToJson(v));
            }
            rows.add(row);
            written++;
            if (rows.size() >= batch) {
                upsert(collection, rows);
                rows.clear();
            }
        }
        if (!rows.isEmpty()) upsert(collection, rows);
        if (opt.loadAfterWrite()) {
            try { loadCollection(collection); } catch (MilvusException ignored) {}
        }
        return written;
    }

    public DataFrame readDataFrame(MilvusOptions options) {
        MilvusOptions opt = options == null ? MilvusOptions.defaults() : options;
        int limit = opt.limit() <= 0 ? 100_000 : opt.limit();
        int pageSize = Math.min(opt.batchSize(), 256);
        List<Map<String, Object>> all = new ArrayList<>();
        long offset = 0;
        while (all.size() < limit) {
            int page = Math.min(pageSize, limit - all.size());
            List<Map<String, Object>> chunk = query(
                opt.collection(),
                opt.filter() == null ? "" : opt.filter(),
                opt.outputFields(),
                page,
                offset
            );
            if (chunk.isEmpty()) break;
            all.addAll(chunk);
            offset += chunk.size();
            if (chunk.size() < page) break;
        }
        return mapsToDataFrame(all, opt);
    }

    public DataFrame searchDataFrame(float[] query, int topK, MilvusOptions options) {
        MilvusOptions opt = options == null ? MilvusOptions.defaults() : options;
        VectorQuery vq = VectorQuery.of(query, topK);
        VectorSearchResult result = search(opt.collection(), vq, opt.vectorField(), opt.metric());
        return result.toDataFrame();
    }

    /** Open a {@link VectorStore} view bound to a collection on this client. */
    public VectorStore asVectorStore(String collection, int dim, VectorMetric metric) {
        return org.bytedeco.pytorch.data.dataframe.vectorstore.milvus.MilvusVectorStore
            .builder(url)
            .collection(collection)
            .dbName(dbName)
            .dim(dim)
            .metric(metric == null ? VectorMetric.COSINE : metric)
            .timeout(timeout)
            .build();
    }

    @Override
    public void close() {
        if (ownHttp) http.close();
    }

    // ── raw ───────────────────────────────────────────────────────────────

    public Object post(String path, Object body) {
        return post(path, body, path);
    }

    public Object post(String path, Object body, String op) {
        try {
            return http.post(path, body);
        } catch (VectorStoreException e) {
            throw new MilvusException(e.getMessage(), e, e.status(), op);
        }
    }

    public Object get(String path) {
        try {
            return http.get(path);
        } catch (VectorStoreException e) {
            throw new MilvusException(e.getMessage(), e, e.status(), path);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private Map<String, Object> baseBody() {
        Map<String, Object> m = new LinkedHashMap<>();
        if (dbName != null && !dbName.isEmpty() && !"default".equals(dbName)) {
            m.put("dbName", dbName);
        }
        return m;
    }

    private static List<VectorHit> parseHits(List<Object> inner, String idField, String vectorField,
                                              boolean includeVector, boolean includePayload) {
        List<VectorHit> hits = new ArrayList<>(inner.size());
        for (Object row : inner) {
            Map<String, Object> m = HttpJson.asMap(row);
            Object idObj = m.get(idField);
            if (idObj == null) idObj = m.get("id");
            String id = HttpJson.asString(idObj);
            float distance = HttpJson.asFloat(m.get("distance"), HttpJson.asFloat(m.get("score"), 0f));
            float[] vec = null;
            if (includeVector) vec = HttpJson.asFloatArray(m.get(vectorField));
            Map<String, Object> payload = new LinkedHashMap<>();
            if (includePayload) {
                for (Map.Entry<String, Object> e : m.entrySet()) {
                    String k = e.getKey();
                    if (idField.equals(k) || vectorField.equals(k)
                        || "distance".equals(k) || "score".equals(k) || "id".equals(k)) continue;
                    payload.put(k, e.getValue());
                }
            }
            long numId = HttpJson.asLong(idObj, -1L);
            hits.add(new VectorHit(id, numId, numId >= 0, distance, distance, vec, payload));
        }
        return hits;
    }

    private static DataFrame mapsToDataFrame(List<Map<String, Object>> rows, MilvusOptions opt) {
        DataFrame df = DataFrame.create();
        if (rows == null || rows.isEmpty()) {
            df.addColumn(opt.idField(), Column.DType.STRING);
            if (opt.includeVector()) df.addColumn(opt.vectorField(), Column.DType.VECTOR);
            return df;
        }
        List<String> keys = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            for (String k : row.keySet()) {
                if (!keys.contains(k)) keys.add(k);
            }
        }
        for (String k : keys) {
            if (k.equals(opt.vectorField())) {
                if (opt.includeVector()) df.addColumn(k, Column.DType.VECTOR);
            } else {
                df.addColumn(k, Column.DType.STRING);
            }
        }
        for (Map<String, Object> row : rows) {
            int r = df.addEmptyRow();
            for (String k : keys) {
                if (k.equals(opt.vectorField()) && !opt.includeVector()) continue;
                Object v = row.get(k);
                if (k.equals(opt.vectorField())) {
                    df.set(r, k, HttpJson.asFloatArray(v));
                } else {
                    df.set(r, k, v == null ? null : String.valueOf(v));
                }
            }
        }
        return df;
    }

    private static String resolveIdColumn(DataFrame df, MilvusOptions opt) {
        if (opt.idColumn() != null && df.hasColumn(opt.idColumn())) return opt.idColumn();
        if (df.hasColumn("id")) return "id";
        return null;
    }

    private static String resolveVectorColumn(DataFrame df, MilvusOptions opt) {
        if (opt.vectorColumn() != null && df.hasColumn(opt.vectorColumn())) return opt.vectorColumn();
        if (df.hasColumn(opt.vectorField())) return opt.vectorField();
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

    private static List<String> resolvePayloadColumns(DataFrame df, MilvusOptions opt, String vectorCol) {
        if (opt.payloadColumns() != null && !opt.payloadColumns().isEmpty()) {
            return opt.payloadColumns();
        }
        List<String> out = new ArrayList<>();
        String idCol = resolveIdColumn(df, opt);
        for (int c = 0; c < df.columnCount(); c++) {
            String n = df.column(c).name();
            if (n.equals(vectorCol)) continue;
            if (idCol != null && n.equals(idCol)) continue;
            if (n.equals(opt.vectorField()) || n.equals(opt.idField())) continue;
            out.add(n);
        }
        return out;
    }

    private static int inferDim(DataFrame df, String vectorCol) {
        Column col = df.column(vectorCol);
        for (int i = 0; i < Math.min(col.size(), 16); i++) {
            float[] v = VectorStore.toFloatArray(col.get(i));
            if (v != null && v.length > 0) return v.length;
        }
        return 0;
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

    public static long hashId(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001b3L;
        }
        return h == Long.MIN_VALUE ? 0L : Math.abs(h);
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
        private String token;
        private String apiKey;
        private String dbName = "default";
        private Duration timeout = DEFAULT_TIMEOUT;

        Builder(String url) { this.url = url; }

        public Builder token(String t) { this.token = t; return this; }
        public Builder apiKey(String k) { this.apiKey = k; return this; }
        public Builder dbName(String d) { this.dbName = d; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }

        public Milvus build() {
            Map<String, Object> cfg = new LinkedHashMap<>();
            cfg.put("url", url);
            if (token != null) cfg.put("token", token);
            if (apiKey != null) cfg.put("apiKey", apiKey);
            if (dbName != null) cfg.put("dbName", dbName);
            if (timeout != null) cfg.put("timeout", timeout);
            return open(cfg);
        }
    }
}
