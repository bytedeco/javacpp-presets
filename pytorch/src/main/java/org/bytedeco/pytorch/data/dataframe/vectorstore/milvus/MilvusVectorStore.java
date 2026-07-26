package org.bytedeco.pytorch.data.dataframe.vectorstore.milvus;

import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.data.dataframe.vectorstore.http.HttpJson;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Milvus adapter over the <b>Milvus RESTful API v2</b> (Milvus ≥ 2.3 / Zilliz Cloud).
 * No {@code milvus-sdk-java} dependency.
 *
 * <p>Default endpoint: {@code http://localhost:19530} (native gRPC port is different;
 * REST usually sits on {@code 9091} for standalone or the cloud HTTPS endpoint).
 * Set {@code url} to the REST base that serves {@code /v2/vectordb/...}.
 *
 * <pre>{@code
 * try (VectorStore vs = MilvusVectorStore.builder("http://localhost:9091")
 *         .collection("clips").dim(768).metric(VectorMetric.L2)
 *         .token("root:Milvus").build()) {
 *     vs.ensureCollection();
 *     vs.upsert(records);
 *     vs.search(query, 10);
 * }
 * }</pre>
 *
 * @see <a href="https://milvus.io/api-reference/restful/v2.4.x/About.md">Milvus RESTful v2</a>
 */
public final class MilvusVectorStore implements VectorStore {

    private final HttpJson http;
    private final String collection;
    private final String dbName;
    private final int dim;
    private final VectorMetric metric;
    private final String idField;
    private final String vectorField;
    private final String indexType;

    private MilvusVectorStore(Builder b) {
        this.collection = Objects.requireNonNull(b.collection, "collection");
        this.dbName = b.dbName == null ? "default" : b.dbName;
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.L2 : b.metric;
        this.idField = b.idField == null ? "id" : b.idField;
        this.vectorField = b.vectorField == null ? "vector" : b.vectorField;
        this.indexType = b.indexType == null ? "AUTOINDEX" : b.indexType;

        HttpJson.Builder hb = HttpJson.builder(b.url)
            .backend("milvus")
            .timeout(b.timeout);
        if (b.token != null && !b.token.isEmpty()) {
            // Zilliz / Milvus REST: Authorization: Bearer <token>
            // token may be "user:pass" or an API key
            hb.header("Authorization", "Bearer " + b.token);
        }
        if (b.apiKey != null && !b.apiKey.isEmpty()) {
            hb.header("Authorization", "Bearer " + b.apiKey);
        }
        this.http = hb.build();
    }

    public static Builder builder(String url) { return new Builder(url); }

    @Override public String backend() { return "milvus"; }
    @Override public String name() { return collection; }
    @Override public int dim() { return dim; }
    @Override public VectorMetric metric() { return metric; }

    @Override
    public void ensureCollection() {
        // has collection?
        Map<String, Object> hasBody = baseBody();
        hasBody.put("collectionName", collection);
        Object hasResp = http.post("/v2/vectordb/collections/has", hasBody);
        boolean exists = Boolean.TRUE.equals(HttpJson.dig(hasResp, "data", "has"))
            || Boolean.TRUE.equals(HttpJson.dig(hasResp, "data"));
        // some versions return data: true directly
        Object data = HttpJson.dig(hasResp, "data");
        if (data instanceof Map<?, ?> m && m.containsKey("has")) {
            exists = Boolean.TRUE.equals(m.get("has"));
        } else if (data instanceof Boolean bo) {
            exists = bo;
        }
        if (exists) {
            // load into memory
            try {
                http.post("/v2/vectordb/collections/load", baseBodyWithCollection());
            } catch (VectorStoreException ignored) {}
            return;
        }
        if (dim <= 0) {
            throw new VectorStoreException("dim required to create Milvus collection", -1, backend());
        }

        // create schema
        List<Map<String, Object>> fields = new ArrayList<>();
        fields.add(HttpJson.mapOf(
            "fieldName", idField,
            "dataType", "Int64",
            "isPrimary", true,
            "autoID", false
        ));
        fields.add(HttpJson.mapOf(
            "fieldName", vectorField,
            "dataType", "FloatVector",
            "elementCount", dim
        ));
        // optional dynamic JSON field for payload
        Map<String, Object> schema = HttpJson.mapOf(
            "autoID", false,
            "enableDynamicField", true,
            "fields", fields
        );
        Map<String, Object> indexParams = HttpJson.mapOf(
            "indexName", vectorField + "_idx",
            "fieldName", vectorField,
            "metricType", metric.milvus(),
            "indexType", indexType,
            "params", HttpJson.mapOf()
        );
        Map<String, Object> create = baseBody();
        create.put("collectionName", collection);
        create.put("schema", schema);
        create.put("indexParams", List.of(indexParams));
        http.post("/v2/vectordb/collections/create", create);

        // load
        http.post("/v2/vectordb/collections/load", baseBodyWithCollection());
    }

    @Override
    public void dropCollection() {
        try {
            http.post("/v2/vectordb/collections/drop", baseBodyWithCollection());
        } catch (VectorStoreException e) {
            if (e.status() != 404 && !String.valueOf(e.getMessage()).toLowerCase().contains("not found")
                && !String.valueOf(e.getMessage()).toLowerCase().contains("doesn't exist")) {
                throw e;
            }
        }
    }

    @Override
    public long count() {
        try {
            Object resp = http.post("/v2/vectordb/entities/get_count", baseBodyWithCollection());
            Object n = HttpJson.dig(resp, "data", "count");
            if (n == null) n = HttpJson.dig(resp, "data");
            return HttpJson.asLong(n, -1L);
        } catch (VectorStoreException e) {
            return -1L;
        }
    }

    @Override
    public void upsert(Collection<VectorRecord> records) {
        if (records == null || records.isEmpty()) return;
        // Milvus REST upsert: /v2/vectordb/entities/upsert
        // data: list of row maps
        final int chunk = 200;
        List<VectorRecord> list = new ArrayList<>(records);
        for (int i = 0; i < list.size(); i += chunk) {
            List<VectorRecord> slice = list.subList(i, Math.min(i + chunk, list.size()));
            List<Map<String, Object>> rows = new ArrayList<>(slice.size());
            for (VectorRecord r : slice) {
                Map<String, Object> row = new LinkedHashMap<>();
                long id = r.hasNumericId() ? r.numericId() : hashId(r.resolvedId());
                row.put(idField, id);
                row.put(vectorField, HttpJson.toDoubleList(r.vector()));
                if (r.payload() != null) {
                    for (Map.Entry<String, Object> e : r.payload().entrySet()) {
                        if (idField.equals(e.getKey()) || vectorField.equals(e.getKey())) continue;
                        row.put(e.getKey(), e.getValue());
                    }
                    // keep original string id if present
                    if (r.id() != null) row.putIfAbsent("_str_id", r.id());
                }
                rows.add(row);
            }
            Map<String, Object> body = baseBodyWithCollection();
            body.put("data", rows);
            http.post("/v2/vectordb/entities/upsert", body);
        }
    }

    @Override
    public void delete(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return;
        // expr: id in [1,2,3]
        StringBuilder expr = new StringBuilder(idField).append(" in [");
        boolean first = true;
        for (String id : ids) {
            if (!first) expr.append(',');
            first = false;
            try {
                expr.append(Long.parseLong(id));
            } catch (NumberFormatException e) {
                expr.append(hashId(id));
            }
        }
        expr.append(']');
        Map<String, Object> body = baseBodyWithCollection();
        body.put("filter", expr.toString());
        http.post("/v2/vectordb/entities/delete", body);
    }

    @Override
    @SuppressWarnings("unchecked")
    public VectorSearchResult search(VectorQuery query) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        Map<String, Object> body = baseBodyWithCollection();
        body.put("data", List.of(HttpJson.toDoubleList(query.vector())));
        body.put("annsField", query.vectorName() != null ? query.vectorName() : vectorField);
        body.put("limit", query.topK());
        List<String> outputs = new ArrayList<>();
        outputs.add(idField);
        if (query.includeVector()) outputs.add(vectorField);
        // dynamic fields returned when outputFields = ["*"]
        if (query.includePayload()) {
            body.put("outputFields", List.of("*"));
        } else {
            body.put("outputFields", outputs);
        }
        Integer nprobe = query.option("nprobe", null);
        Map<String, Object> searchParams = new LinkedHashMap<>();
        searchParams.put("metricType", metric.milvus());
        Map<String, Object> params = new LinkedHashMap<>();
        if (nprobe != null) params.put("nprobe", nprobe);
        Integer ef = query.option("ef", null);
        if (ef != null) params.put("ef", ef);
        searchParams.put("params", params);
        body.put("searchParams", searchParams);
        if (query.filter() instanceof String s && !s.isBlank()) {
            body.put("filter", s);
        }

        Object resp = http.post("/v2/vectordb/entities/search", body);
        // data: [ [ {id, distance, ...}, ... ] ]  (one list per query vector)
        Object data = HttpJson.dig(resp, "data");
        List<Object> outer = HttpJson.asList(data);
        List<Object> inner;
        if (!outer.isEmpty() && outer.get(0) instanceof List<?>) {
            inner = HttpJson.asList(outer.get(0));
        } else {
            inner = outer;
        }
        List<VectorHit> hits = new ArrayList<>(inner.size());
        for (Object row : inner) {
            Map<String, Object> m = HttpJson.asMap(row);
            Object idObj = m.get(idField);
            if (idObj == null) idObj = m.get("id");
            String id = HttpJson.asString(idObj);
            float distance = HttpJson.asFloat(m.get("distance"), HttpJson.asFloat(m.get("score"), 0f));
            float score = distance;
            float[] vec = null;
            if (query.includeVector()) {
                vec = HttpJson.asFloatArray(m.get(vectorField));
            }
            Map<String, Object> payload = new LinkedHashMap<>();
            if (query.includePayload()) {
                for (Map.Entry<String, Object> e : m.entrySet()) {
                    String k = e.getKey();
                    if (idField.equals(k) || vectorField.equals(k)
                        || "distance".equals(k) || "score".equals(k) || "id".equals(k)) continue;
                    payload.put(k, e.getValue());
                }
            }
            long numId = HttpJson.asLong(idObj, -1L);
            hits.add(new VectorHit(id, numId, numId >= 0, score, distance, vec, payload));
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<VectorRecord> fetch(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return List.of();
        // /v2/vectordb/entities/get  with filter id in [...]
        StringBuilder expr = new StringBuilder(idField).append(" in [");
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
        Map<String, Object> body = baseBodyWithCollection();
        body.put("filter", expr.toString());
        body.put("outputFields", List.of("*"));
        Object resp = http.post("/v2/vectordb/entities/query", body);
        List<Object> rows = HttpJson.asList(HttpJson.dig(resp, "data"));
        List<VectorRecord> out = new ArrayList<>(rows.size());
        for (Object row : rows) {
            Map<String, Object> m = HttpJson.asMap(row);
            Object idObj = m.get(idField);
            if (idObj == null) idObj = m.get("id");
            String id = HttpJson.asString(idObj);
            float[] vec = HttpJson.asFloatArray(m.get(vectorField));
            if (vec == null) continue;
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : m.entrySet()) {
                String k = e.getKey();
                if (idField.equals(k) || vectorField.equals(k) || "id".equals(k)) continue;
                payload.put(k, e.getValue());
            }
            if (idObj instanceof Number n) {
                out.add(VectorRecord.of(n.longValue(), vec, payload));
            } else {
                out.add(VectorRecord.of(id, vec, payload));
            }
        }
        return out;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ScrollPage scroll(int limit, Object cursor) {
        int lim = Math.max(1, limit);
        long offset = 0L;
        if (cursor instanceof Number n) offset = Math.max(0L, n.longValue());
        else if (cursor instanceof String s) {
            try { offset = Long.parseLong(s); } catch (NumberFormatException ignored) {}
        }
        Map<String, Object> body = baseBodyWithCollection();
        body.put("limit", lim);
        body.put("offset", offset);
        body.put("outputFields", List.of("*"));
        // empty filter = all
        body.put("filter", "");
        Object resp;
        try {
            resp = http.post("/v2/vectordb/entities/query", body);
        } catch (VectorStoreException e) {
            return ScrollPage.empty();
        }
        List<Object> rows = HttpJson.asList(HttpJson.dig(resp, "data"));
        List<VectorRecord> page = new ArrayList<>(rows.size());
        for (Object row : rows) {
            Map<String, Object> m = HttpJson.asMap(row);
            Object idObj = m.get(idField);
            if (idObj == null) idObj = m.get("id");
            String id = HttpJson.asString(idObj);
            float[] vec = HttpJson.asFloatArray(m.get(vectorField));
            if (vec == null) vec = new float[Math.max(dim, 0)];
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : m.entrySet()) {
                String k = e.getKey();
                if (idField.equals(k) || vectorField.equals(k) || "id".equals(k)) continue;
                payload.put(k, e.getValue());
            }
            page.add(VectorRecord.of(id, vec, payload));
        }
        Object next = page.size() < lim ? null : Long.valueOf(offset + page.size());
        return new ScrollPage(page, next);
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<VectorSearchResult> searchBatch(List<VectorQuery> queries) {
        if (queries == null || queries.isEmpty()) return List.of();
        // Milvus REST accepts data: [[v1],[v2],...] multi-vector search
        // Use first query's topK / options as shared; per-query topK may differ → fall back
        boolean uniform = true;
        int topK = queries.get(0).topK();
        for (VectorQuery q : queries) {
            if (q.topK() != topK) { uniform = false; break; }
        }
        if (!uniform) {
            return VectorStore.super.searchBatch(queries);
        }
        List<List<Double>> data = new ArrayList<>(queries.size());
        for (VectorQuery q : queries) data.add(HttpJson.toDoubleList(q.vector()));
        Map<String, Object> body = baseBodyWithCollection();
        body.put("data", data);
        body.put("annsField", vectorField);
        body.put("limit", topK);
        body.put("outputFields", List.of("*"));
        body.put("searchParams", HttpJson.mapOf(
            "metricType", metric.milvus(),
            "params", Map.of()
        ));
        Object resp = http.post("/v2/vectordb/entities/search", body);
        Object rawData = HttpJson.dig(resp, "data");
        List<Object> outer = HttpJson.asList(rawData);
        List<VectorSearchResult> out = new ArrayList<>(queries.size());
        // If server returns one flat list for single-vector only, handle multi
        boolean nested = !outer.isEmpty() && outer.get(0) instanceof List<?>;
        if (!nested) {
            // single block — only one query worth
            out.add(parseMilvusHits(outer, topK));
            while (out.size() < queries.size()) out.add(VectorSearchResult.empty());
            return out;
        }
        for (Object block : outer) {
            out.add(parseMilvusHits(HttpJson.asList(block), topK));
        }
        while (out.size() < queries.size()) out.add(VectorSearchResult.empty());
        return out;
    }

    private VectorSearchResult parseMilvusHits(List<Object> inner, int topK) {
        List<VectorHit> hits = new ArrayList<>(inner.size());
        for (Object row : inner) {
            Map<String, Object> m = HttpJson.asMap(row);
            Object idObj = m.get(idField);
            if (idObj == null) idObj = m.get("id");
            String id = HttpJson.asString(idObj);
            float distance = HttpJson.asFloat(m.get("distance"), HttpJson.asFloat(m.get("score"), 0f));
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : m.entrySet()) {
                String k = e.getKey();
                if (idField.equals(k) || vectorField.equals(k)
                    || "distance".equals(k) || "score".equals(k) || "id".equals(k)) continue;
                payload.put(k, e.getValue());
            }
            long numId = HttpJson.asLong(idObj, -1L);
            hits.add(new VectorHit(id, numId, numId >= 0, distance, distance, null, payload));
            if (hits.size() >= topK) break;
        }
        return new VectorSearchResult(hits);
    }

    @Override
    public void close() {
        http.close();
    }

    private Map<String, Object> baseBody() {
        Map<String, Object> m = new LinkedHashMap<>();
        if (dbName != null && !dbName.isEmpty() && !"default".equals(dbName)) {
            m.put("dbName", dbName);
        }
        return m;
    }

    private Map<String, Object> baseBodyWithCollection() {
        Map<String, Object> m = baseBody();
        m.put("collectionName", collection);
        return m;
    }

    private static long hashId(String s) {
        // stable positive long from string (FNV-1a 64)
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001b3L;
        }
        return h == Long.MIN_VALUE ? 0L : Math.abs(h);
    }

    public static final class Builder {
        private final String url;
        private String collection = "vectors";
        private String dbName = "default";
        private int dim;
        private VectorMetric metric = VectorMetric.L2;
        private String token;
        private String apiKey;
        private Duration timeout = Duration.ofSeconds(30);
        private String idField = "id";
        private String vectorField = "vector";
        private String indexType = "AUTOINDEX";

        Builder(String url) { this.url = url; }

        public Builder collection(String c) { this.collection = c; return this; }
        public Builder dbName(String d) { this.dbName = d; return this; }
        public Builder dim(int d) { this.dim = d; return this; }
        public Builder metric(VectorMetric m) { this.metric = m; return this; }
        public Builder token(String t) { this.token = t; return this; }
        public Builder apiKey(String k) { this.apiKey = k; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }
        public Builder idField(String f) { this.idField = f; return this; }
        public Builder vectorField(String f) { this.vectorField = f; return this; }
        public Builder indexType(String t) { this.indexType = t; return this; }

        public MilvusVectorStore build() {
            return new MilvusVectorStore(this);
        }
    }
}
