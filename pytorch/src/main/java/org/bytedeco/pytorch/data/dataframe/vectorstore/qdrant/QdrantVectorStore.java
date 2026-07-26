package org.bytedeco.pytorch.data.dataframe.vectorstore.qdrant;

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
import java.util.UUID;

/**
 * Qdrant adapter over the public REST API — no {@code qdrant-client} dependency.
 *
 * <p>Default endpoint: {@code http://localhost:6333}.
 * Auth: optional API key via {@code api-key} header.
 *
 * <pre>{@code
 * try (VectorStore vs = QdrantVectorStore.builder("http://localhost:6333")
 *         .collection("clips").dim(768).metric(VectorMetric.COSINE).build()) {
 *     vs.ensureCollection();
 *     vs.upsert(VectorRecord.of("a", emb));
 *     vs.search(emb, 10);
 * }
 * }</pre>
 *
 * @see <a href="https://qdrant.tech/documentation/concepts/points/">Qdrant points API</a>
 */
public final class QdrantVectorStore implements VectorStore {

    private final HttpJson http;
    private final String collection;
    private final int dim;
    private final VectorMetric metric;
    private final boolean onDisk;
    private final int shardNumber;

    private QdrantVectorStore(Builder b) {
        this.collection = Objects.requireNonNull(b.collection, "collection");
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.onDisk = b.onDisk;
        this.shardNumber = b.shardNumber;
        HttpJson.Builder hb = HttpJson.builder(b.url)
            .backend("qdrant")
            .timeout(b.timeout);
        if (b.apiKey != null && !b.apiKey.isEmpty()) {
            hb.header("api-key", b.apiKey);
        }
        this.http = hb.build();
    }

    public static Builder builder(String url) { return new Builder(url); }

    @Override public String backend() { return "qdrant"; }
    @Override public String name() { return collection; }
    @Override public int dim() { return dim; }
    @Override public VectorMetric metric() { return metric; }

    @Override
    public void ensureCollection() {
        // GET /collections/{name} — 404 → create
        try {
            http.get("/collections/" + enc(collection));
            return; // exists
        } catch (VectorStoreException e) {
            if (e.status() != 404) throw e;
        }
        if (dim <= 0) {
            throw new VectorStoreException("dim required to create Qdrant collection", -1, backend());
        }
        Map<String, Object> vectors = HttpJson.mapOf(
            "size", dim,
            "distance", metric.qdrant(),
            "on_disk", onDisk
        );
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("vectors", vectors);
        if (shardNumber > 0) body.put("shard_number", shardNumber);
        http.put("/collections/" + enc(collection), body);
    }

    @Override
    public void dropCollection() {
        try {
            http.delete("/collections/" + enc(collection));
        } catch (VectorStoreException e) {
            if (e.status() != 404) throw e;
        }
    }

    @Override
    public long count() {
        Object resp = http.get("/collections/" + enc(collection));
        Object n = HttpJson.dig(resp, "result", "points_count");
        if (n == null) n = HttpJson.dig(resp, "result", "indexed_vectors_count");
        return HttpJson.asLong(n, -1L);
    }

    @Override
    public void upsert(Collection<VectorRecord> records) {
        if (records == null || records.isEmpty()) return;
        List<Map<String, Object>> points = new ArrayList<>(records.size());
        for (VectorRecord r : records) {
            Map<String, Object> p = new LinkedHashMap<>();
            // Qdrant accepts unsigned int or UUID string
            Object id = resolveQdrantId(r);
            p.put("id", id);
            p.put("vector", HttpJson.toDoubleList(r.vector()));
            if (r.payload() != null && !r.payload().isEmpty()) {
                p.put("payload", r.payload());
            }
            points.add(p);
        }
        // batch in chunks of 256
        final int chunk = 256;
        for (int i = 0; i < points.size(); i += chunk) {
            List<Map<String, Object>> slice = points.subList(i, Math.min(i + chunk, points.size()));
            Map<String, Object> body = HttpJson.mapOf("points", slice);
            http.put("/collections/" + enc(collection) + "/points?wait=true", body);
        }
    }

    @Override
    public void delete(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return;
        List<Object> list = new ArrayList<>(ids.size());
        for (String id : ids) list.add(parseId(id));
        Map<String, Object> body = HttpJson.mapOf("points", list);
        http.post("/collections/" + enc(collection) + "/points/delete?wait=true", body);
    }

    @Override
    @SuppressWarnings("unchecked")
    public VectorSearchResult search(VectorQuery query) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("vector", HttpJson.toDoubleList(query.vector()));
        body.put("limit", query.topK());
        body.put("with_payload", query.includePayload());
        body.put("with_vector", query.includeVector());
        Integer ef = query.option("ef", null);
        if (ef != null) {
            body.put("params", HttpJson.mapOf("hnsw_ef", ef));
        }
        if (query.filter() != null) {
            body.put("filter", query.filter());
        }
        Object resp = http.post("/collections/" + enc(collection) + "/points/search", body);
        List<Object> result = HttpJson.asList(HttpJson.dig(resp, "result"));
        List<VectorHit> hits = new ArrayList<>(result.size());
        for (Object row : result) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get("id"));
            float score = HttpJson.asFloat(m.get("score"), 0f);
            // Qdrant score is similarity for Cosine/Dot (higher better), distance for Euclid
            Float distance = null;
            if (metric == VectorMetric.L2) {
                distance = score;
            } else {
                // approximate: distance = 1 - score for cosine-ish; keep raw as score
                distance = metric == VectorMetric.COSINE ? (1f - score) : -score;
            }
            float[] vec = null;
            if (query.includeVector()) {
                vec = HttpJson.asFloatArray(m.get("vector"));
            }
            Map<String, Object> payload = Map.of();
            if (query.includePayload() && m.get("payload") instanceof Map<?, ?> pm) {
                payload = new LinkedHashMap<>((Map<String, Object>) pm);
            }
            long numId = -1L;
            boolean hasNum = false;
            try {
                numId = Long.parseLong(id);
                hasNum = true;
            } catch (Exception ignored) {}
            hits.add(new VectorHit(id, numId, hasNum, score, distance, vec, payload));
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<VectorRecord> fetch(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return List.of();
        List<Object> idList = new ArrayList<>(ids.size());
        for (String id : ids) idList.add(parseId(id));
        Map<String, Object> body = HttpJson.mapOf(
            "ids", idList,
            "with_payload", true,
            "with_vector", true
        );
        Object resp = http.post("/collections/" + enc(collection) + "/points", body);
        List<Object> result = HttpJson.asList(HttpJson.dig(resp, "result"));
        List<VectorRecord> out = new ArrayList<>(result.size());
        for (Object row : result) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get("id"));
            float[] vec = HttpJson.asFloatArray(m.get("vector"));
            if (vec == null) continue;
            Map<String, Object> payload = Map.of();
            if (m.get("payload") instanceof Map<?, ?> pm) {
                payload = new LinkedHashMap<>((Map<String, Object>) pm);
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
        body.put("limit", lim);
        body.put("with_payload", true);
        body.put("with_vector", true);
        if (cursor != null) body.put("offset", cursor);
        Object resp = http.post("/collections/" + enc(collection) + "/points/scroll", body);
        List<Object> result = HttpJson.asList(HttpJson.dig(resp, "result", "points"));
        // some versions nest differently
        if (result.isEmpty()) {
            Object alt = HttpJson.dig(resp, "result");
            if (alt instanceof Map<?, ?> rm) {
                result = HttpJson.asList(rm.get("points"));
            } else if (alt instanceof List<?>) {
                result = HttpJson.asList(alt);
            }
        }
        List<VectorRecord> page = new ArrayList<>(result.size());
        for (Object row : result) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get("id"));
            float[] vec = HttpJson.asFloatArray(m.get("vector"));
            if (vec == null) vec = new float[Math.max(dim, 0)];
            Map<String, Object> payload = Map.of();
            if (m.get("payload") instanceof Map<?, ?> pm) {
                payload = new LinkedHashMap<>((Map<String, Object>) pm);
            }
            page.add(VectorRecord.of(id, vec, payload));
        }
        Object next = HttpJson.dig(resp, "result", "next_page_offset");
        if (next == null && page.size() < lim) next = null;
        else if (next == null && !page.isEmpty()) {
            // no explicit cursor — stop
            next = null;
        }
        return new ScrollPage(page, next);
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<VectorSearchResult> searchBatch(List<VectorQuery> queries) {
        if (queries == null || queries.isEmpty()) return List.of();
        // Qdrant /points/search/batch
        List<Map<String, Object>> batch = new ArrayList<>(queries.size());
        for (VectorQuery q : queries) {
            Map<String, Object> one = new LinkedHashMap<>();
            one.put("vector", HttpJson.toDoubleList(q.vector()));
            one.put("limit", q.topK());
            one.put("with_payload", q.includePayload());
            one.put("with_vector", q.includeVector());
            if (q.filter() != null) one.put("filter", q.filter());
            Integer ef = q.option("ef", null);
            if (ef != null) one.put("params", HttpJson.mapOf("hnsw_ef", ef));
            batch.add(one);
        }
        Object resp = http.post("/collections/" + enc(collection) + "/points/search/batch",
            HttpJson.mapOf("searches", batch));
        List<Object> result = HttpJson.asList(HttpJson.dig(resp, "result"));
        List<VectorSearchResult> out = new ArrayList<>(result.size());
        for (Object block : result) {
            List<Object> rows = HttpJson.asList(block);
            List<VectorHit> hits = new ArrayList<>(rows.size());
            for (Object row : rows) {
                Map<String, Object> m = HttpJson.asMap(row);
                String id = HttpJson.asString(m.get("id"));
                float score = HttpJson.asFloat(m.get("score"), 0f);
                Float distance = metric == VectorMetric.L2 ? score
                    : (metric == VectorMetric.COSINE ? (1f - score) : -score);
                float[] vec = HttpJson.asFloatArray(m.get("vector"));
                Map<String, Object> payload = Map.of();
                if (m.get("payload") instanceof Map<?, ?> pm) {
                    payload = new LinkedHashMap<>((Map<String, Object>) pm);
                }
                hits.add(new VectorHit(id, -1L, false, score, distance, vec, payload));
            }
            out.add(new VectorSearchResult(hits));
        }
        // pad if server returned fewer
        while (out.size() < queries.size()) out.add(VectorSearchResult.empty());
        return out;
    }

    @Override
    public void close() {
        http.close();
    }

    private static Object resolveQdrantId(VectorRecord r) {
        if (r.hasNumericId() && r.numericId() >= 0) return r.numericId();
        String s = r.id();
        if (s == null || s.isEmpty()) {
            return UUID.randomUUID().toString();
        }
        // prefer numeric if parseable
        try {
            long v = Long.parseLong(s);
            if (v >= 0) return v;
        } catch (NumberFormatException ignored) {}
        // Qdrant requires UUID format for string ids — if not UUID, hash to UUID-ish
        if (isUuid(s)) return s;
        return UUID.nameUUIDFromBytes(s.getBytes(java.nio.charset.StandardCharsets.UTF_8)).toString();
    }

    private static Object parseId(String id) {
        if (id == null) return null;
        try {
            long v = Long.parseLong(id);
            if (v >= 0) return v;
        } catch (NumberFormatException ignored) {}
        if (isUuid(id)) return id;
        return UUID.nameUUIDFromBytes(id.getBytes(java.nio.charset.StandardCharsets.UTF_8)).toString();
    }

    private static boolean isUuid(String s) {
        try {
            UUID.fromString(s);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private static String enc(String s) {
        return java.net.URLEncoder.encode(s, java.nio.charset.StandardCharsets.UTF_8).replace("+", "%20");
    }

    public static final class Builder {
        private final String url;
        private String collection = "vectors";
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private String apiKey;
        private Duration timeout = Duration.ofSeconds(30);
        private boolean onDisk;
        private int shardNumber;

        Builder(String url) { this.url = url; }

        public Builder collection(String c) { this.collection = c; return this; }
        public Builder dim(int d) { this.dim = d; return this; }
        public Builder metric(VectorMetric m) { this.metric = m; return this; }
        public Builder apiKey(String k) { this.apiKey = k; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }
        public Builder onDisk(boolean v) { this.onDisk = v; return this; }
        public Builder shardNumber(int n) { this.shardNumber = n; return this; }

        public QdrantVectorStore build() {
            return new QdrantVectorStore(this);
        }
    }
}
