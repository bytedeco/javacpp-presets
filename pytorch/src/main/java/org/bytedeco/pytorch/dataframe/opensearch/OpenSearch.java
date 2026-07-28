package org.bytedeco.pytorch.dataframe.opensearch;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.vectorstore.PayloadField;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.dataframe.vectorstore.http.HttpJson;
import org.bytedeco.pytorch.dataframe.vectorstore.opensearch.OpenSearchVectorStore;
import org.bytedeco.pytorch.utils.json.Json;

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
 * Full-featured OpenSearch client for DataFrame I/O — pure REST via {@link HttpJson},
 * no {@code opensearch-java} / {@code elasticsearch-java} dependency.
 *
 * <h2>Coverage (opensearch-java HighLevel parity subset)</h2>
 * <ul>
 *   <li><b>Connection</b> — connect / basicAuth / apiKey / close</li>
 *   <li><b>Indices</b> — exists / create / delete / refresh</li>
 *   <li><b>Documents</b> — index / bulk / get / mget / delete / count</li>
 *   <li><b>Search</b> — search / knn / script_score fallback / search_after scroll</li>
 *   <li><b>DataFrame</b> — {@link #writeDataFrame}, {@link #readDataFrame}, {@link #searchDataFrame}</li>
 * </ul>
 *
 * <h2>Official-SDK switch (SPI only)</h2>
 * Implement {@link OpenSearchBackend} and register via {@code META-INF/services} or
 * {@link #registerBackend}. A backend named {@code "opensearch"} overrides this built-in.
 *
 * <pre>{@code
 * try (OpenSearch os = OpenSearch.connect("http://localhost:9200", "admin", "admin")) {
 *     os.createKnnIndex("docs", 384, VectorMetric.COSINE);
 *     df.toOpenSearch(os, OpenSearchOptions.builder().index("docs").idColumn("id")
 *         .vectorColumn("emb").dim(384).build());
 * }
 * }</pre>
 */
public class OpenSearch implements Closeable {

    public static final int DEFAULT_PORT = 9200;
    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);

    private static final Map<String, OpenSearchBackend> BACKENDS = new ConcurrentHashMap<>();
    static {
        reloadBackends();
    }

    private final HttpJson http;
    private final String url;
    private final Duration timeout;
    private final boolean ownHttp;

    protected OpenSearch(HttpJson http, String url, Duration timeout, boolean ownHttp) {
        this.http = Objects.requireNonNull(http, "http");
        this.url = url;
        this.timeout = timeout == null ? DEFAULT_TIMEOUT : timeout;
        this.ownHttp = ownHttp;
    }

    // ── SPI ───────────────────────────────────────────────────────────────

    public static void reloadBackends() {
        BACKENDS.clear();
        try {
            for (OpenSearchBackend b : ServiceLoader.load(OpenSearchBackend.class)) {
                registerBackend(b);
            }
        } catch (Throwable ignored) {}
    }

    public static void registerBackend(OpenSearchBackend backend) {
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

    public static OpenSearchBackend backend(String name) {
        if (name == null) return null;
        return BACKENDS.get(name.toLowerCase(Locale.ROOT));
    }

    // ── factories ─────────────────────────────────────────────────────────

    public static OpenSearch connect() {
        return connect("http://localhost:" + DEFAULT_PORT);
    }

    public static OpenSearch connect(String url) {
        return connect(url, null, null, null, DEFAULT_TIMEOUT);
    }

    public static OpenSearch connect(String url, String user, String password) {
        return connect(url, user, password, null, DEFAULT_TIMEOUT);
    }

    public static OpenSearch connect(String url, String user, String password,
                                      String apiKey, Duration timeout) {
        Map<String, Object> cfg = new LinkedHashMap<>();
        cfg.put("url", url);
        if (user != null) cfg.put("username", user);
        if (password != null) cfg.put("password", password);
        if (apiKey != null) cfg.put("apiKey", apiKey);
        if (timeout != null) cfg.put("timeout", timeout);
        return open(cfg);
    }

    public static OpenSearch connectUri(String uri) {
        Objects.requireNonNull(uri, "uri");
        String s = uri.trim();
        Map<String, Object> cfg = new LinkedHashMap<>();
        if (s.startsWith("opensearch://") || s.startsWith("elasticsearch://") || s.startsWith("es://")) {
            String rest = s.substring(s.indexOf("://") + 3);
            String path = rest;
            String query = null;
            int q = rest.indexOf('?');
            if (q >= 0) {
                path = rest.substring(0, q);
                query = rest.substring(q + 1);
            }
            // optional user:pass@host:port
            String hostPort = path;
            int at = path.lastIndexOf('@');
            if (at >= 0) {
                String auth = path.substring(0, at);
                hostPort = path.substring(at + 1);
                int colon = auth.indexOf(':');
                if (colon >= 0) {
                    cfg.put("username", auth.substring(0, colon));
                    cfg.put("password", auth.substring(colon + 1));
                } else {
                    cfg.put("username", auth);
                }
            }
            int slash = hostPort.indexOf('/');
            if (slash >= 0) {
                cfg.put("index", hostPort.substring(slash + 1));
                hostPort = hostPort.substring(0, slash);
            }
            cfg.put("url", "http://" + hostPort);
            if (query != null) parseQuery(query, cfg);
        } else {
            cfg.put("url", s);
        }
        return open(cfg);
    }

    public static OpenSearch open(Map<String, Object> config) {
        Map<String, Object> cfg = config == null ? Map.of() : config;
        OpenSearchBackend plugin = BACKENDS.get("opensearch");
        if (plugin == null) plugin = BACKENDS.get("elasticsearch");
        if (plugin == null) plugin = BACKENDS.get("es");
        if (plugin != null) return plugin.open(cfg);
        return openBuiltin(cfg);
    }

    public static OpenSearch openBuiltin(Map<String, Object> cfg) {
        String url = str(cfg, "url", "http://localhost:" + DEFAULT_PORT);
        Duration timeout = duration(cfg.get("timeout"));
        HttpJson.Builder hb = HttpJson.builder(url).backend("opensearch").timeout(timeout);
        String user = str(cfg, "username", str(cfg, "user", null));
        if (user != null) hb.basic(user, str(cfg, "password", ""));
        String apiKey = str(cfg, "apiKey", str(cfg, "api_key", null));
        if (apiKey != null && !apiKey.isEmpty()) {
            hb.header("Authorization", "ApiKey " + apiKey);
        }
        return new OpenSearch(hb.build(), url, timeout, true);
    }

    public static Builder builder(String url) {
        return new Builder(url);
    }

    // ── accessors ─────────────────────────────────────────────────────────

    public HttpJson http() { return http; }
    public String url() { return url; }
    public Duration timeout() { return timeout; }

    // ── indices ───────────────────────────────────────────────────────────

    public boolean indexExists(String index) {
        try {
            http.get("/" + enc(index));
            return true;
        } catch (VectorStoreException e) {
            if (e.status() == 404) return false;
            throw wrap(e, "indices/exists");
        }
    }

    public void createIndex(String index, Map<String, Object> body) {
        try {
            http.put("/" + enc(index), body == null ? Map.of() : body);
        } catch (VectorStoreException e) {
            throw wrap(e, "indices/create");
        }
    }

    public void createKnnIndex(String index, int dim, VectorMetric metric) {
        createKnnIndex(index, dim, metric, "vector", "faiss", 16, 100, List.of());
    }

    public void createKnnIndex(String index, int dim, VectorMetric metric,
                                String vectorField, String engine,
                                int m, int efConstruction, List<PayloadField> payloadFields) {
        if (dim <= 0) throw new OpenSearchException("dim required to create knn index", -1, "indices/create");
        String vf = vectorField == null ? "vector" : vectorField;
        String eng = engine == null ? "faiss" : engine;
        VectorMetric met = metric == null ? VectorMetric.L2 : metric;

        Map<String, Object> method = new LinkedHashMap<>();
        method.put("name", "hnsw");
        method.put("space_type", met.openSearch());
        method.put("engine", eng);
        Map<String, Object> params = new LinkedHashMap<>();
        if (m > 0) params.put("m", m);
        if (efConstruction > 0) params.put("ef_construction", efConstruction);
        method.put("parameters", params);

        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put(vf, HttpJson.mapOf(
            "type", "knn_vector",
            "dimension", dim,
            "method", method
        ));
        if (payloadFields != null) {
            for (PayloadField pf : payloadFields) {
                if (pf == null || vf.equals(pf.name())) continue;
                properties.put(pf.name(), pf.openSearchProperty());
            }
        }

        Map<String, Object> body = HttpJson.mapOf(
            "settings", HttpJson.mapOf("index", HttpJson.mapOf("knn", true)),
            "mappings", HttpJson.mapOf("dynamic", true, "properties", properties)
        );
        createIndex(index, body);
    }

    public void ensureKnnIndex(String index, int dim, VectorMetric metric,
                                String vectorField, String engine,
                                int m, int efConstruction, List<PayloadField> payloadFields) {
        if (indexExists(index)) return;
        createKnnIndex(index, dim, metric, vectorField, engine, m, efConstruction, payloadFields);
    }

    public void deleteIndex(String index) {
        try {
            http.delete("/" + enc(index));
        } catch (VectorStoreException e) {
            if (e.status() != 404) throw wrap(e, "indices/delete");
        }
    }

    public void refresh(String index) {
        try {
            http.post("/" + enc(index) + "/_refresh", Map.of());
        } catch (VectorStoreException ignored) {}
    }

    // ── documents ─────────────────────────────────────────────────────────

    public long count(String index) {
        try {
            Object resp = http.post("/" + enc(index) + "/_count", Map.of());
            return HttpJson.asLong(HttpJson.dig(resp, "count"), -1L);
        } catch (VectorStoreException e) {
            return -1L;
        }
    }

    public void index(String index, String id, Map<String, Object> doc) {
        try {
            String path = "/" + enc(index) + "/_doc"
                + (id == null ? "" : "/" + enc(id));
            if (id == null) http.post(path, doc);
            else http.put(path, doc);
        } catch (VectorStoreException e) {
            throw wrap(e, "index");
        }
    }

    public void bulkIndex(String index, List<Map<String, Object>> docs, String idField) {
        if (docs == null || docs.isEmpty()) return;
        StringBuilder ndjson = new StringBuilder(docs.size() * 256);
        for (Map<String, Object> doc : docs) {
            Object id = idField == null ? null : doc.get(idField);
            Map<String, Object> actionMeta = new LinkedHashMap<>();
            actionMeta.put("_index", index);
            if (id != null) actionMeta.put("_id", String.valueOf(id));
            Map<String, Object> action = HttpJson.mapOf("index", actionMeta);
            ndjson.append(Json.encode(action)).append('\n');
            Map<String, Object> body = new LinkedHashMap<>(doc);
            if (idField != null) body.remove(idField); // _id is meta, not necessarily in source
            // keep id field in source too if present under different key — leave as-is for payload
            ndjson.append(Json.encode(doc)).append('\n');
        }
        Object resp;
        try {
            resp = http.postNdjson("/_bulk", ndjson.toString());
        } catch (VectorStoreException e) {
            throw wrap(e, "bulk");
        }
        checkBulkErrors(resp, "index");
    }

    public void bulkIndexRecords(String index, Collection<VectorRecord> records,
                                  String vectorField, int bulkBatch) {
        if (records == null || records.isEmpty()) return;
        String vf = vectorField == null ? "vector" : vectorField;
        int batch = Math.max(1, bulkBatch);
        List<VectorRecord> list = records instanceof List
            ? (List<VectorRecord>) records
            : new ArrayList<>(records);
        for (int i = 0; i < list.size(); i += batch) {
            List<VectorRecord> slice = list.subList(i, Math.min(i + batch, list.size()));
            StringBuilder ndjson = new StringBuilder(slice.size() * 256);
            for (VectorRecord r : slice) {
                String id = r.resolvedId();
                Map<String, Object> action = HttpJson.mapOf(
                    "index", HttpJson.mapOf("_index", index, "_id", id)
                );
                ndjson.append(Json.encode(action)).append('\n');
                Map<String, Object> doc = new LinkedHashMap<>();
                doc.put(vf, HttpJson.toDoubleList(r.vector()));
                if (r.payload() != null) {
                    for (Map.Entry<String, Object> e : r.payload().entrySet()) {
                        if (vf.equals(e.getKey())) continue;
                        doc.put(e.getKey(), e.getValue());
                    }
                }
                ndjson.append(Json.encode(doc)).append('\n');
            }
            Object resp;
            try {
                resp = http.postNdjson("/_bulk", ndjson.toString());
            } catch (VectorStoreException e) {
                throw wrap(e, "bulk");
            }
            checkBulkErrors(resp, "index");
        }
    }

    public void deleteByIds(String index, Collection<String> ids, int bulkBatch) {
        if (ids == null || ids.isEmpty()) return;
        int batch = Math.max(1, bulkBatch);
        List<String> list = ids instanceof List ? (List<String>) ids : new ArrayList<>(ids);
        for (int i = 0; i < list.size(); i += batch) {
            List<String> slice = list.subList(i, Math.min(i + batch, list.size()));
            StringBuilder ndjson = new StringBuilder(slice.size() * 64);
            for (String id : slice) {
                Map<String, Object> action = HttpJson.mapOf(
                    "delete", HttpJson.mapOf("_index", index, "_id", id)
                );
                ndjson.append(Json.encode(action)).append('\n');
            }
            try {
                Object resp = http.postNdjson("/_bulk", ndjson.toString());
                checkBulkErrors(resp, "delete");
            } catch (VectorStoreException e) {
                throw wrap(e, "bulk/delete");
            }
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> get(String index, String id) {
        try {
            Object resp = http.get("/" + enc(index) + "/_doc/" + enc(id));
            Map<String, Object> m = HttpJson.asMap(resp);
            if (!Boolean.TRUE.equals(m.get("found"))) return null;
            Map<String, Object> source = HttpJson.asMap(m.get("_source"));
            source.putIfAbsent("_id", m.get("_id"));
            return source;
        } catch (VectorStoreException e) {
            if (e.status() == 404) return null;
            throw wrap(e, "get");
        }
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> mget(String index, Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return List.of();
        List<Map<String, Object>> docs = new ArrayList<>();
        for (String id : ids) {
            if (id != null) docs.add(HttpJson.mapOf("_id", id));
        }
        Object resp;
        try {
            resp = http.post("/" + enc(index) + "/_mget", HttpJson.mapOf("docs", docs));
        } catch (VectorStoreException e) {
            throw wrap(e, "mget");
        }
        List<Object> raw = HttpJson.asList(HttpJson.dig(resp, "docs"));
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object row : raw) {
            Map<String, Object> m = HttpJson.asMap(row);
            if (!Boolean.TRUE.equals(m.get("found"))) continue;
            Map<String, Object> source = HttpJson.asMap(m.get("_source"));
            source.put("_id", m.get("_id"));
            out.add(source);
        }
        return out;
    }

    // ── search ────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public Object search(String index, Map<String, Object> body) {
        try {
            return http.post("/" + enc(index) + "/_search", body);
        } catch (VectorStoreException e) {
            throw wrap(e, "search");
        }
    }

    public VectorSearchResult knnSearch(String index, VectorQuery query,
                                         String vectorField, VectorMetric metric) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        String vf = vectorField == null ? "vector" : vectorField;
        int k = query.topK();
        VectorMetric met = metric == null ? VectorMetric.L2 : metric;

        Map<String, Object> knnClause = HttpJson.mapOf(
            "vector", HttpJson.toDoubleList(query.vector()),
            "k", k
        );
        Map<String, Object> knn = HttpJson.mapOf(vf, knnClause);
        Map<String, Object> knnQuery = HttpJson.mapOf("knn", knn);

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("size", k);
        if (query.filter() instanceof Map<?, ?> filterMap) {
            body.put("query", HttpJson.mapOf(
                "bool", HttpJson.mapOf(
                    "must", knnQuery,
                    "filter", filterMap
                )
            ));
        } else {
            body.put("query", knnQuery);
        }
        if (!query.includeVector()) {
            body.put("_source", HttpJson.mapOf("excludes", List.of(vf)));
        }

        Object resp;
        try {
            resp = http.post("/" + enc(index) + "/_search", body);
        } catch (VectorStoreException e) {
            resp = scriptScoreSearch(index, query, k, vf, met);
        }

        List<Object> hitsRaw = HttpJson.asList(HttpJson.dig(resp, "hits", "hits"));
        List<VectorHit> hits = new ArrayList<>(hitsRaw.size());
        for (Object row : hitsRaw) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get("_id"));
            float score = HttpJson.asFloat(m.get("_score"), 0f);
            Map<String, Object> source = HttpJson.asMap(m.get("_source"));
            float[] vec = null;
            if (query.includeVector()) {
                vec = HttpJson.asFloatArray(source.get(vf));
            }
            Map<String, Object> payload = new LinkedHashMap<>();
            if (query.includePayload()) {
                for (Map.Entry<String, Object> e : source.entrySet()) {
                    if (vf.equals(e.getKey())) continue;
                    payload.put(e.getKey(), e.getValue());
                }
            }
            Float distance = met == VectorMetric.L2
                ? (score > 0 ? 1f / score : null)
                : (1f - score);
            hits.add(new VectorHit(id, -1L, false, score, distance, vec, payload));
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        Object tookObj = HttpJson.dig(resp, "took");
        if (tookObj instanceof Number n) took = n.longValue();
        return new VectorSearchResult(hits, took);
    }

    private Object scriptScoreSearch(String index, VectorQuery query, int k,
                                      String vectorField, VectorMetric metric) {
        String source;
        Map<String, Object> params = HttpJson.mapOf(
            "query_vector", HttpJson.toDoubleList(query.vector())
        );
        switch (metric) {
            case COSINE -> source = "cosineSimilarity(params.query_vector, '" + vectorField + "') + 1.0";
            case IP -> source = "dotProduct(params.query_vector, '" + vectorField + "')";
            default -> source = "1 / (1 + l2norm(params.query_vector, '" + vectorField + "'))";
        }
        Map<String, Object> body = HttpJson.mapOf(
            "size", k,
            "query", HttpJson.mapOf(
                "script_score", HttpJson.mapOf(
                    "query", HttpJson.mapOf("match_all", Map.of()),
                    "script", HttpJson.mapOf("source", source, "params", params)
                )
            )
        );
        try {
            return http.post("/" + enc(index) + "/_search", body);
        } catch (VectorStoreException e) {
            throw wrap(e, "script_score");
        }
    }

    /**
     * Page through documents with {@code search_after} on {@code _id}.
     *
     * @return hits as maps including {@code _id}; empty list means end
     */
    @SuppressWarnings("unchecked")
    public ScrollPage scroll(String index, int limit, Object cursor) {
        int lim = Math.max(1, limit);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("size", lim);
        body.put("query", HttpJson.mapOf("match_all", Map.of()));
        body.put("sort", List.of(HttpJson.mapOf("_id", "asc")));
        if (cursor instanceof String s && !s.isBlank()) {
            body.put("search_after", List.of(s));
        } else if (cursor instanceof List<?> l && !l.isEmpty()) {
            body.put("search_after", l);
        }
        Object resp;
        try {
            resp = http.post("/" + enc(index) + "/_search", body);
        } catch (VectorStoreException e) {
            throw wrap(e, "scroll");
        }
        List<Object> hitsRaw = HttpJson.asList(HttpJson.dig(resp, "hits", "hits"));
        List<Map<String, Object>> page = new ArrayList<>(hitsRaw.size());
        Object nextCur = null;
        for (Object row : hitsRaw) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get("_id"));
            Map<String, Object> source = HttpJson.asMap(m.get("_source"));
            Map<String, Object> doc = new LinkedHashMap<>(source);
            doc.put("_id", id);
            page.add(doc);
            Object sort = m.get("sort");
            if (sort instanceof List<?> sl && !sl.isEmpty()) {
                nextCur = sl;
            } else {
                nextCur = id;
            }
        }
        if (page.size() < lim) nextCur = null;
        return new ScrollPage(page, nextCur);
    }

    public record ScrollPage(List<Map<String, Object>> hits, Object nextCursor) {
        public boolean isEmpty() { return hits == null || hits.isEmpty(); }
    }

    // ── DataFrame I/O ─────────────────────────────────────────────────────

    public int writeDataFrame(DataFrame df, OpenSearchOptions options) {
        Objects.requireNonNull(df, "df");
        OpenSearchOptions opt = options == null ? OpenSearchOptions.defaults() : options;
        String index = opt.index();

        if (opt.ifExists() == OpenSearchOptions.IfExists.REPLACE) {
            deleteIndex(index);
        } else if (opt.ifExists() == OpenSearchOptions.IfExists.FAIL && indexExists(index)) {
            throw new OpenSearchException("index exists: " + index, -1, "writeDataFrame");
        } else if (opt.ifExists() == OpenSearchOptions.IfExists.SKIP && indexExists(index)
                && count(index) > 0) {
            return 0;
        }

        String vectorCol = resolveVectorColumn(df, opt);
        int dim = opt.dim();
        if (dim <= 0 && vectorCol != null) dim = inferDim(df, vectorCol);

        if (opt.ensureIndex() && !indexExists(index)) {
            if (dim > 0) {
                ensureKnnIndex(index, dim, opt.metric(), opt.vectorField(),
                    opt.engine(), opt.m(), opt.efConstruction(), opt.payloadFields());
            } else {
                createIndex(index, HttpJson.mapOf(
                    "mappings", HttpJson.mapOf("dynamic", true)
                ));
            }
        }

        List<String> payloadCols = resolvePayloadColumns(df, opt, vectorCol);
        String idCol = resolveIdColumn(df, opt);
        int written = 0;
        int batch = opt.bulkBatch();
        List<Map<String, Object>> docs = new ArrayList<>(batch);

        for (int r = 0; r < df.rowCount(); r++) {
            Map<String, Object> doc = new LinkedHashMap<>();
            Object idv = idCol != null ? df.get(r, idCol) : r;
            String id = idv == null ? String.valueOf(r) : String.valueOf(idv);
            doc.put("_id", id);

            if (vectorCol != null) {
                float[] vec = VectorStore.toFloatArray(df.get(r, vectorCol));
                if (vec != null) {
                    doc.put(opt.vectorField(), HttpJson.toDoubleList(vec));
                } else if (!opt.includeNulls()) {
                    continue;
                }
            }
            for (String pn : payloadCols) {
                Object v = df.get(r, pn);
                if (v == null && !opt.includeNulls()) continue;
                doc.put(pn, cellToJson(v));
            }
            docs.add(doc);
            written++;
            if (docs.size() >= batch) {
                flushDocs(index, docs, opt);
                docs.clear();
            }
        }
        if (!docs.isEmpty()) flushDocs(index, docs, opt);
        if (opt.refreshOnWrite()) refresh(index);
        return written;
    }

    private void flushDocs(String index, List<Map<String, Object>> docs, OpenSearchOptions opt) {
        StringBuilder ndjson = new StringBuilder(docs.size() * 256);
        for (Map<String, Object> doc : docs) {
            Object id = doc.get("_id");
            Map<String, Object> action = HttpJson.mapOf(
                "index", HttpJson.mapOf("_index", index, "_id", String.valueOf(id))
            );
            ndjson.append(Json.encode(action)).append('\n');
            Map<String, Object> source = new LinkedHashMap<>(doc);
            source.remove("_id");
            ndjson.append(Json.encode(source)).append('\n');
        }
        Object resp;
        try {
            resp = http.postNdjson("/_bulk", ndjson.toString());
        } catch (VectorStoreException e) {
            throw wrap(e, "bulk");
        }
        checkBulkErrors(resp, "index");
    }

    public DataFrame readDataFrame(OpenSearchOptions options) {
        OpenSearchOptions opt = options == null ? OpenSearchOptions.defaults() : options;
        int limit = opt.limit() <= 0 ? 100_000 : opt.limit();
        int pageSize = Math.min(opt.bulkBatch(), 256);
        List<Map<String, Object>> all = new ArrayList<>();
        Object cursor = null;
        while (all.size() < limit) {
            int page = Math.min(pageSize, limit - all.size());
            ScrollPage sp = scroll(opt.index(), page, cursor);
            if (sp.isEmpty()) break;
            all.addAll(sp.hits());
            cursor = sp.nextCursor();
            if (cursor == null || sp.hits().size() < page) break;
        }
        return mapsToDataFrame(all, opt);
    }

    public DataFrame searchDataFrame(float[] query, int topK, OpenSearchOptions options) {
        OpenSearchOptions opt = options == null ? OpenSearchOptions.defaults() : options;
        VectorQuery vq = VectorQuery.of(query, topK);
        return knnSearch(opt.index(), vq, opt.vectorField(), opt.metric()).toDataFrame();
    }

    public VectorStore asVectorStore(String index, int dim, VectorMetric metric) {
        return OpenSearchVectorStore
            .builder(url)
            .index(index)
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

    private void checkBulkErrors(Object resp, String op) {
        if (!(resp instanceof Map<?, ?>)) return;
        Object errors = ((Map<?, ?>) resp).get("errors");
        if (!Boolean.TRUE.equals(errors)) return;
        List<Object> items = HttpJson.asList(((Map<?, ?>) resp).get("items"));
        for (Object item : items) {
            Map<String, Object> m = HttpJson.asMap(item);
            Object sub = m.get(op);
            if (sub == null && !m.isEmpty()) sub = m.values().iterator().next();
            Map<String, Object> sm = HttpJson.asMap(sub);
            Object err = sm.get("error");
            int status = HttpJson.asInt(sm.get("status"), 0);
            if (err != null && status >= 400 && status != 404) {
                throw new OpenSearchException(
                    "opensearch _bulk " + op + " error: " + err, status, "bulk");
            }
        }
    }

    private static DataFrame mapsToDataFrame(List<Map<String, Object>> rows, OpenSearchOptions opt) {
        DataFrame df = DataFrame.create();
        if (rows == null || rows.isEmpty()) {
            df.addColumn("_id", Column.DType.STRING);
            if (opt.includeVector()) df.addColumn(opt.vectorField(), Column.DType.VECTOR);
            return df;
        }
        List<String> keys = new ArrayList<>();
        keys.add("_id");
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

    private static String resolveIdColumn(DataFrame df, OpenSearchOptions opt) {
        if (opt.idColumn() != null && df.hasColumn(opt.idColumn())) return opt.idColumn();
        if (df.hasColumn("id")) return "id";
        if (df.hasColumn("_id")) return "_id";
        return null;
    }

    private static String resolveVectorColumn(DataFrame df, OpenSearchOptions opt) {
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

    private static List<String> resolvePayloadColumns(DataFrame df, OpenSearchOptions opt, String vectorCol) {
        if (opt.payloadColumns() != null && !opt.payloadColumns().isEmpty()) {
            return opt.payloadColumns();
        }
        List<String> out = new ArrayList<>();
        String idCol = resolveIdColumn(df, opt);
        for (int c = 0; c < df.columnCount(); c++) {
            String n = df.column(c).name();
            if (n.equals(vectorCol)) continue;
            if (idCol != null && n.equals(idCol)) continue;
            if (n.equals(opt.vectorField())) continue;
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

    private static OpenSearchException wrap(VectorStoreException e, String op) {
        return new OpenSearchException(e.getMessage(), e, e.status(), op);
    }

    private static String enc(String s) {
        return java.net.URLEncoder.encode(s, java.nio.charset.StandardCharsets.UTF_8).replace("+", "%20");
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
        private String username;
        private String password;
        private String apiKey;
        private Duration timeout = DEFAULT_TIMEOUT;

        Builder(String url) { this.url = url; }

        public Builder basicAuth(String user, String pass) {
            this.username = user;
            this.password = pass;
            return this;
        }
        public Builder apiKey(String k) { this.apiKey = k; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }

        public OpenSearch build() {
            Map<String, Object> cfg = new LinkedHashMap<>();
            cfg.put("url", url);
            if (username != null) cfg.put("username", username);
            if (password != null) cfg.put("password", password);
            if (apiKey != null) cfg.put("apiKey", apiKey);
            if (timeout != null) cfg.put("timeout", timeout);
            return open(cfg);
        }
    }
}
