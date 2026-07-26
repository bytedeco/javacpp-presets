package org.bytedeco.pytorch.data.dataframe.vectorstore.opensearch;

import org.bytedeco.pytorch.data.dataframe.vectorstore.PayloadField;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.data.dataframe.vectorstore.http.HttpJson;
import org.bytedeco.pytorch.utils.json.Json;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * OpenSearch (and Elasticsearch 8+ k-NN) adapter over the REST API —
 * no {@code opensearch-java} / {@code elasticsearch-java} client dependency.
 *
 * <p>Bulk path uses the native {@code _bulk} NDJSON API
 * ({@code Content-Type: application/x-ndjson}). Fetch uses {@code mget},
 * scroll uses {@code search_after} on {@code _id}.
 *
 * <pre>{@code
 * try (VectorStore vs = OpenSearchVectorStore.builder("https://localhost:9200")
 *         .index("clips").dim(768).metric(VectorMetric.L2)
 *         .payloadField(PayloadField.text("title"))
 *         .payloadField(PayloadField.tag("category"))
 *         .basicAuth("admin", "admin").build()) {
 *     vs.ensureCollection();
 *     vs.upsert(largeBatch);          // _bulk
 *     vs.search(query, 10);
 * }
 * }</pre>
 */
public final class OpenSearchVectorStore implements VectorStore {

    private final HttpJson http;
    private final String index;
    private final int dim;
    private final VectorMetric metric;
    private final String vectorField;
    private final String engine;
    private final int m;
    private final int efConstruction;
    private final List<PayloadField> payloadFields;
    private final int bulkBatch;
    private final boolean refreshOnWrite;

    private OpenSearchVectorStore(Builder b) {
        this.index = Objects.requireNonNull(b.index, "index");
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.L2 : b.metric;
        this.vectorField = b.vectorField == null ? "vector" : b.vectorField;
        this.engine = b.engine == null ? "nmslib" : b.engine;
        this.m = b.m;
        this.efConstruction = b.efConstruction;
        this.payloadFields = List.copyOf(b.payloadFields);
        this.bulkBatch = Math.max(1, b.bulkBatch);
        this.refreshOnWrite = b.refreshOnWrite;

        HttpJson.Builder hb = HttpJson.builder(b.url)
            .backend("opensearch")
            .timeout(b.timeout);
        if (b.username != null) {
            hb.basic(b.username, b.password == null ? "" : b.password);
        }
        if (b.apiKey != null && !b.apiKey.isEmpty()) {
            hb.header("Authorization", "ApiKey " + b.apiKey);
        }
        this.http = hb.build();
    }

    public static Builder builder(String url) { return new Builder(url); }

    @Override public String backend() { return "opensearch"; }
    @Override public String name() { return index; }
    @Override public int dim() { return dim; }
    @Override public VectorMetric metric() { return metric; }

    @Override
    public void ensureCollection() {
        try {
            http.get("/" + enc(index));
            return;
        } catch (VectorStoreException e) {
            if (e.status() != 404) throw e;
        }
        if (dim <= 0) {
            throw new VectorStoreException("dim required to create OpenSearch knn index", -1, backend());
        }
        Map<String, Object> method = new LinkedHashMap<>();
        method.put("name", "hnsw");
        method.put("space_type", metric.openSearch());
        method.put("engine", engine);
        Map<String, Object> params = new LinkedHashMap<>();
        if (m > 0) params.put("m", m);
        if (efConstruction > 0) params.put("ef_construction", efConstruction);
        method.put("parameters", params);

        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put(vectorField, HttpJson.mapOf(
            "type", "knn_vector",
            "dimension", dim,
            "method", method
        ));
        for (PayloadField pf : payloadFields) {
            if (vectorField.equals(pf.name())) continue;
            properties.put(pf.name(), pf.openSearchProperty());
        }

        Map<String, Object> body = HttpJson.mapOf(
            "settings", HttpJson.mapOf("index", HttpJson.mapOf("knn", true)),
            "mappings", HttpJson.mapOf("dynamic", true, "properties", properties)
        );
        http.put("/" + enc(index), body);
    }

    @Override
    public void dropCollection() {
        try {
            http.delete("/" + enc(index));
        } catch (VectorStoreException e) {
            if (e.status() != 404) throw e;
        }
    }

    @Override
    public long count() {
        try {
            Object resp = http.post("/" + enc(index) + "/_count", Map.of());
            return HttpJson.asLong(HttpJson.dig(resp, "count"), -1L);
        } catch (VectorStoreException e) {
            return -1L;
        }
    }

    @Override
    public void upsert(Collection<VectorRecord> records) {
        if (records == null || records.isEmpty()) return;
        List<VectorRecord> list = records instanceof List
            ? (List<VectorRecord>) records
            : new ArrayList<>(records);
        for (int i = 0; i < list.size(); i += bulkBatch) {
            List<VectorRecord> slice = list.subList(i, Math.min(i + bulkBatch, list.size()));
            bulkIndex(slice);
        }
        if (refreshOnWrite) refresh();
    }

    private void bulkIndex(List<VectorRecord> slice) {
        StringBuilder ndjson = new StringBuilder(slice.size() * 256);
        for (VectorRecord r : slice) {
            String id = r.resolvedId();
            Map<String, Object> action = HttpJson.mapOf(
                "index", HttpJson.mapOf("_index", index, "_id", id)
            );
            ndjson.append(Json.encode(action)).append('\n');
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put(vectorField, HttpJson.toDoubleList(r.vector()));
            if (r.payload() != null) {
                for (Map.Entry<String, Object> e : r.payload().entrySet()) {
                    if (vectorField.equals(e.getKey())) continue;
                    doc.put(e.getKey(), e.getValue());
                }
            }
            ndjson.append(Json.encode(doc)).append('\n');
        }
        Object resp = http.postNdjson("/_bulk", ndjson.toString());
        checkBulkErrors(resp, "index");
    }

    @Override
    public void delete(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return;
        List<String> list = ids instanceof List ? (List<String>) ids : new ArrayList<>(ids);
        for (int i = 0; i < list.size(); i += bulkBatch) {
            List<String> slice = list.subList(i, Math.min(i + bulkBatch, list.size()));
            StringBuilder ndjson = new StringBuilder(slice.size() * 64);
            for (String id : slice) {
                Map<String, Object> action = HttpJson.mapOf(
                    "delete", HttpJson.mapOf("_index", index, "_id", id)
                );
                ndjson.append(Json.encode(action)).append('\n');
            }
            Object resp = http.postNdjson("/_bulk", ndjson.toString());
            // ignore not_found items
            checkBulkErrors(resp, "delete");
        }
        if (refreshOnWrite) refresh();
    }

    private void checkBulkErrors(Object resp, String op) {
        if (!(resp instanceof Map<?, ?>)) return;
        Object errors = ((Map<?, ?>) resp).get("errors");
        if (!Boolean.TRUE.equals(errors)) return;
        // surface first error item
        List<Object> items = HttpJson.asList(((Map<?, ?>) resp).get("items"));
        for (Object item : items) {
            Map<String, Object> m = HttpJson.asMap(item);
            Object sub = m.get(op);
            if (sub == null) {
                // try any key
                if (!m.isEmpty()) sub = m.values().iterator().next();
            }
            Map<String, Object> sm = HttpJson.asMap(sub);
            Object err = sm.get("error");
            int status = HttpJson.asInt(sm.get("status"), 0);
            if (err != null && status >= 400 && status != 404) {
                throw new VectorStoreException(
                    "opensearch _bulk " + op + " error: " + err, status, backend());
            }
        }
    }

    private void refresh() {
        try {
            http.post("/" + enc(index) + "/_refresh", Map.of());
        } catch (VectorStoreException ignored) {}
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<VectorRecord> fetch(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return List.of();
        List<Map<String, Object>> docs = new ArrayList<>();
        for (String id : ids) {
            if (id != null) docs.add(HttpJson.mapOf("_id", id));
        }
        if (docs.isEmpty()) return List.of();
        Object resp = http.post("/" + enc(index) + "/_mget", HttpJson.mapOf("docs", docs));
        List<Object> raw = HttpJson.asList(HttpJson.dig(resp, "docs"));
        List<VectorRecord> out = new ArrayList<>();
        for (Object row : raw) {
            Map<String, Object> m = HttpJson.asMap(row);
            if (!Boolean.TRUE.equals(m.get("found"))) continue;
            String id = HttpJson.asString(m.get("_id"));
            Map<String, Object> source = HttpJson.asMap(m.get("_source"));
            float[] vec = HttpJson.asFloatArray(source.get(vectorField));
            if (vec == null) continue;
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : source.entrySet()) {
                if (vectorField.equals(e.getKey())) continue;
                payload.put(e.getKey(), e.getValue());
            }
            out.add(VectorRecord.of(id, vec, payload));
        }
        return out;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ScrollPage scroll(int limit, Object cursor) {
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
        Object resp = http.post("/" + enc(index) + "/_search", body);
        List<Object> hitsRaw = HttpJson.asList(HttpJson.dig(resp, "hits", "hits"));
        List<VectorRecord> page = new ArrayList<>(hitsRaw.size());
        Object nextCur = null;
        for (Object row : hitsRaw) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get("_id"));
            Map<String, Object> source = HttpJson.asMap(m.get("_source"));
            float[] vec = HttpJson.asFloatArray(source.get(vectorField));
            if (vec == null) vec = new float[Math.max(dim, 0)];
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : source.entrySet()) {
                if (vectorField.equals(e.getKey())) continue;
                payload.put(e.getKey(), e.getValue());
            }
            page.add(VectorRecord.of(id, vec, payload));
            Object sort = m.get("sort");
            if (sort instanceof List<?> sl && !sl.isEmpty()) {
                nextCur = sl; // pass whole sort values back
            } else {
                nextCur = id;
            }
        }
        if (page.size() < lim) nextCur = null;
        return new ScrollPage(page, nextCur);
    }

    @Override
    @SuppressWarnings("unchecked")
    public VectorSearchResult search(VectorQuery query) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        int k = query.topK();

        Map<String, Object> knnClause = HttpJson.mapOf(
            "vector", HttpJson.toDoubleList(query.vector()),
            "k", k
        );
        Map<String, Object> knn = HttpJson.mapOf(vectorField, knnClause);
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
            body.put("_source", HttpJson.mapOf("excludes", List.of(vectorField)));
        }

        Object resp;
        try {
            resp = http.post("/" + enc(index) + "/_search", body);
        } catch (VectorStoreException e) {
            resp = scriptScoreSearch(query, k);
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
                vec = HttpJson.asFloatArray(source.get(vectorField));
            }
            Map<String, Object> payload = new LinkedHashMap<>();
            if (query.includePayload()) {
                for (Map.Entry<String, Object> e : source.entrySet()) {
                    if (vectorField.equals(e.getKey())) continue;
                    payload.put(e.getKey(), e.getValue());
                }
            }
            Float distance = metric == VectorMetric.L2
                ? (score > 0 ? 1f / score : null)
                : (1f - score);
            hits.add(new VectorHit(id, -1L, false, score, distance, vec, payload));
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        Object tookObj = HttpJson.dig(resp, "took");
        if (tookObj instanceof Number n) took = n.longValue();
        return new VectorSearchResult(hits, took);
    }

    private Object scriptScoreSearch(VectorQuery query, int k) {
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
        return http.post("/" + enc(index) + "/_search", body);
    }

    @Override
    public void close() {
        http.close();
    }

    private static String enc(String s) {
        return java.net.URLEncoder.encode(s, java.nio.charset.StandardCharsets.UTF_8).replace("+", "%20");
    }

    public static final class Builder {
        private final String url;
        private String index = "vectors";
        private int dim;
        private VectorMetric metric = VectorMetric.L2;
        private String vectorField = "vector";
        private String engine = "nmslib";
        private int m = 16;
        private int efConstruction = 100;
        private String username;
        private String password;
        private String apiKey;
        private Duration timeout = Duration.ofSeconds(30);
        private final List<PayloadField> payloadFields = new ArrayList<>();
        private int bulkBatch = 500;
        private boolean refreshOnWrite = true;

        Builder(String url) { this.url = url; }

        public Builder index(String i) { this.index = i; return this; }
        public Builder dim(int d) { this.dim = d; return this; }
        public Builder metric(VectorMetric m) { this.metric = m; return this; }
        public Builder vectorField(String f) { this.vectorField = f; return this; }
        public Builder engine(String e) { this.engine = e; return this; }
        public Builder m(int m) { this.m = m; return this; }
        public Builder efConstruction(int ef) { this.efConstruction = ef; return this; }
        public Builder basicAuth(String user, String pass) {
            this.username = user; this.password = pass; return this;
        }
        public Builder apiKey(String k) { this.apiKey = k; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }
        public Builder bulkBatch(int n) { this.bulkBatch = n; return this; }
        public Builder refreshOnWrite(boolean v) { this.refreshOnWrite = v; return this; }
        public Builder payloadField(PayloadField f) {
            if (f != null) payloadFields.add(f);
            return this;
        }
        public Builder payloadFields(Collection<PayloadField> fields) {
            if (fields != null) payloadFields.addAll(fields);
            return this;
        }

        public OpenSearchVectorStore build() {
            return new OpenSearchVectorStore(this);
        }
    }
}
