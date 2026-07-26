package org.bytedeco.pytorch.data.dataframe.vectorstore.mongo;

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
 * MongoDB Atlas Vector Search adapter over the <b>Atlas Data API</b> (HTTPS) —
 * no {@code mongodb-driver-sync} dependency.
 *
 * <p>Works with:
 * <ul>
 *   <li>Atlas Data API endpoint
 *       ({@code https://data.mongodb-api.com/app/&lt;app&gt;/endpoint/data/v1})</li>
 *   <li>Any HTTPS gateway that speaks the same
 *       {@code action/insertOne|updateOne|deleteOne|aggregate} protocol</li>
 * </ul>
 *
 * <p>Vector Search itself is configured in Atlas UI (search index on the vector
 * path). {@link #ensureCollection()} only upserts a trivial placeholder document
 * so the collection exists; it cannot create the Atlas Search index via Data API.
 *
 * <pre>{@code
 * try (VectorStore vs = MongoAtlasVectorStore.builder(dataApiUrl)
 *         .apiKey(key).dataSource("Cluster0").database("rag").collection("clips")
 *         .dim(768).metric(VectorMetric.COSINE)
 *         .vectorPath("embedding").indexName("vector_index").build()) {
 *     vs.upsert(records);
 *     vs.search(query, 10);
 * }
 * }</pre>
 *
 * @see <a href="https://www.mongodb.com/docs/atlas/api/data-api/">Atlas Data API</a>
 * @see <a href="https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/">Atlas Vector Search</a>
 */
public final class MongoAtlasVectorStore implements VectorStore {

    private final HttpJson http;
    private final String dataSource;
    private final String database;
    private final String collection;
    private final int dim;
    private final VectorMetric metric;
    private final String vectorPath;
    private final String indexName;
    private final String idField;

    private MongoAtlasVectorStore(Builder b) {
        this.dataSource = Objects.requireNonNull(b.dataSource, "dataSource");
        this.database = Objects.requireNonNull(b.database, "database");
        this.collection = Objects.requireNonNull(b.collection, "collection");
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.vectorPath = b.vectorPath == null ? "embedding" : b.vectorPath;
        this.indexName = b.indexName == null ? "vector_index" : b.indexName;
        this.idField = b.idField == null ? "_id" : b.idField;

        HttpJson.Builder hb = HttpJson.builder(b.url)
            .backend("mongo")
            .timeout(b.timeout)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");
        if (b.apiKey != null && !b.apiKey.isEmpty()) {
            hb.header("api-key", b.apiKey);
        }
        this.http = hb.build();
    }

    public static Builder builder(String dataApiBaseUrl) {
        return new Builder(dataApiBaseUrl);
    }

    @Override public String backend() { return "mongo"; }
    @Override public String name() { return database + "." + collection; }
    @Override public int dim() { return dim; }
    @Override public VectorMetric metric() { return metric; }

    @Override
    public void ensureCollection() {
        // Data API cannot create Atlas Search indexes. Insert a no-op doc then delete
        // so the collection materializes; vector index must be created in Atlas UI / Admin API.
        Map<String, Object> doc = new LinkedHashMap<>();
        doc.put(idField, "__vectorstore_init__");
        doc.put(vectorPath, zeroVector(Math.max(dim, 1)));
        doc.put("_init", true);
        try {
            action("insertOne", HttpJson.mapOf("document", doc));
            action("deleteOne", HttpJson.mapOf(
                "filter", HttpJson.mapOf(idField, "__vectorstore_init__")));
        } catch (VectorStoreException e) {
            // collection may already exist / duplicate key — fine
            String msg = String.valueOf(e.getMessage()).toLowerCase();
            if (!msg.contains("duplicate") && e.status() != 409) {
                // still OK if delete failed after insert
            }
        }
    }

    @Override
    public void dropCollection() {
        // delete all documents (cannot drop collection via standard Data API on all tiers)
        action("deleteMany", HttpJson.mapOf("filter", Map.of()));
    }

    @Override
    public long count() {
        try {
            Object resp = action("aggregate", HttpJson.mapOf(
                "pipeline", List.of(Map.of("$count", "n"))
            ));
            List<Object> docs = HttpJson.asList(HttpJson.dig(resp, "documents"));
            if (!docs.isEmpty()) {
                return HttpJson.asLong(HttpJson.asMap(docs.get(0)).get("n"), -1L);
            }
        } catch (VectorStoreException e) {
            return -1L;
        }
        return 0L;
    }

    @Override
    public void upsert(Collection<VectorRecord> records) {
        if (records == null || records.isEmpty()) return;
        // Atlas Data API has no bulkWrite; chunk updateOne. For pure inserts of new ids,
        // try insertMany first (fast path), fall back to per-doc upsert on conflict.
        List<VectorRecord> list = records instanceof List
            ? (List<VectorRecord>) records
            : new ArrayList<>(records);
        final int chunk = 100;
        for (int i = 0; i < list.size(); i += chunk) {
            List<VectorRecord> slice = list.subList(i, Math.min(i + chunk, list.size()));
            // Prefer per-doc upsert for correctness (insert-or-replace)
            for (VectorRecord r : slice) {
                String id = r.resolvedId();
                Map<String, Object> doc = toDoc(r);
                Map<String, Object> body = new LinkedHashMap<>();
                body.put("filter", HttpJson.mapOf(idField, id));
                body.put("update", HttpJson.mapOf("$set", doc));
                body.put("upsert", true);
                action("updateOne", body);
            }
        }
    }

    private Map<String, Object> toDoc(VectorRecord r) {
        Map<String, Object> doc = new LinkedHashMap<>();
        doc.put(idField, r.resolvedId());
        doc.put(vectorPath, HttpJson.toDoubleList(r.vector()));
        if (r.payload() != null) {
            for (Map.Entry<String, Object> e : r.payload().entrySet()) {
                if (idField.equals(e.getKey()) || vectorPath.equals(e.getKey())) continue;
                doc.put(e.getKey(), e.getValue());
            }
        }
        return doc;
    }

    @Override
    public void delete(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return;
        // deleteMany with $in (one round-trip)
        List<String> list = new ArrayList<>();
        for (String id : ids) if (id != null) list.add(id);
        if (list.isEmpty()) return;
        action("deleteMany", HttpJson.mapOf(
            "filter", HttpJson.mapOf(idField, HttpJson.mapOf("$in", list))));
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<VectorRecord> fetch(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return List.of();
        List<String> list = new ArrayList<>();
        for (String id : ids) if (id != null) list.add(id);
        if (list.isEmpty()) return List.of();
        Object resp = action("find", HttpJson.mapOf(
            "filter", HttpJson.mapOf(idField, HttpJson.mapOf("$in", list)),
            "limit", list.size()
        ));
        return parseDocs(HttpJson.asList(HttpJson.dig(resp, "documents")));
    }

    @Override
    @SuppressWarnings("unchecked")
    public ScrollPage scroll(int limit, Object cursor) {
        int lim = Math.max(1, limit);
        int skip = 0;
        if (cursor instanceof Number n) skip = Math.max(0, n.intValue());
        else if (cursor instanceof String s) {
            try { skip = Integer.parseInt(s); } catch (NumberFormatException ignored) {}
        }
        Object resp = action("find", HttpJson.mapOf(
            "filter", Map.of(),
            "limit", lim,
            "skip", skip
        ));
        List<VectorRecord> page = parseDocs(HttpJson.asList(HttpJson.dig(resp, "documents")));
        Object next = page.size() < lim ? null : Integer.valueOf(skip + page.size());
        return new ScrollPage(page, next);
    }

    private List<VectorRecord> parseDocs(List<Object> docs) {
        List<VectorRecord> out = new ArrayList<>(docs.size());
        for (Object row : docs) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get(idField));
            float[] vec = HttpJson.asFloatArray(m.get(vectorPath));
            if (vec == null) continue;
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : m.entrySet()) {
                String key = e.getKey();
                if (idField.equals(key) || vectorPath.equals(key)) continue;
                payload.put(key, e.getValue());
            }
            out.add(VectorRecord.of(id, vec, payload));
        }
        return out;
    }

    @Override
    @SuppressWarnings("unchecked")
    public VectorSearchResult search(VectorQuery query) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        int k = query.topK();
        int numCandidates = query.option("num_candidates", Math.max(k * 10, 100));

        Map<String, Object> vectorSearch = new LinkedHashMap<>();
        vectorSearch.put("index", indexName);
        vectorSearch.put("path", vectorPath);
        vectorSearch.put("queryVector", HttpJson.toDoubleList(query.vector()));
        vectorSearch.put("numCandidates", numCandidates);
        vectorSearch.put("limit", k);
        if (query.filter() instanceof Map<?, ?> f) {
            vectorSearch.put("filter", f);
        }
        // similarity is controlled by the Atlas index definition; metric field is informational here

        List<Map<String, Object>> pipeline = new ArrayList<>();
        pipeline.add(HttpJson.mapOf("$vectorSearch", vectorSearch));
        pipeline.add(HttpJson.mapOf("$addFields", HttpJson.mapOf(
            "score", HttpJson.mapOf("$meta", "vectorSearchScore")
        )));
        if (!query.includeVector()) {
            pipeline.add(HttpJson.mapOf("$project", HttpJson.mapOf(vectorPath, 0)));
        }

        Object resp = action("aggregate", HttpJson.mapOf("pipeline", pipeline));
        List<Object> docs = HttpJson.asList(HttpJson.dig(resp, "documents"));
        List<VectorHit> hits = new ArrayList<>(docs.size());
        for (Object row : docs) {
            Map<String, Object> m = HttpJson.asMap(row);
            String id = HttpJson.asString(m.get(idField));
            float score = HttpJson.asFloat(m.get("score"), 0f);
            float[] vec = null;
            if (query.includeVector()) {
                vec = HttpJson.asFloatArray(m.get(vectorPath));
            }
            Map<String, Object> payload = new LinkedHashMap<>();
            if (query.includePayload()) {
                for (Map.Entry<String, Object> e : m.entrySet()) {
                    String key = e.getKey();
                    if (idField.equals(key) || vectorPath.equals(key) || "score".equals(key)) continue;
                    payload.put(key, e.getValue());
                }
            }
            // Atlas vectorSearchScore is similarity (higher better)
            Float distance = metric == VectorMetric.COSINE ? (1f - score) : -score;
            hits.add(new VectorHit(id, -1L, false, score, distance, vec, payload));
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    @Override
    public void close() {
        http.close();
    }

    private Object action(String actionName, Map<String, Object> extra) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("dataSource", dataSource);
        body.put("database", database);
        body.put("collection", collection);
        if (extra != null) body.putAll(extra);
        return http.post("/action/" + actionName, body);
    }

    private static List<Double> zeroVector(int d) {
        List<Double> v = new ArrayList<>(d);
        for (int i = 0; i < d; i++) v.add(0.0);
        return v;
    }

    public static final class Builder {
        private final String url;
        private String apiKey;
        private String dataSource = "Cluster0";
        private String database = "test";
        private String collection = "vectors";
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private String vectorPath = "embedding";
        private String indexName = "vector_index";
        private String idField = "_id";
        private Duration timeout = Duration.ofSeconds(30);

        Builder(String url) { this.url = url; }

        public Builder apiKey(String k) { this.apiKey = k; return this; }
        public Builder dataSource(String ds) { this.dataSource = ds; return this; }
        public Builder database(String db) { this.database = db; return this; }
        public Builder collection(String c) { this.collection = c; return this; }
        public Builder dim(int d) { this.dim = d; return this; }
        public Builder metric(VectorMetric m) { this.metric = m; return this; }
        public Builder vectorPath(String p) { this.vectorPath = p; return this; }
        public Builder indexName(String n) { this.indexName = n; return this; }
        public Builder idField(String f) { this.idField = f; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }

        public MongoAtlasVectorStore build() {
            return new MongoAtlasVectorStore(this);
        }
    }
}
