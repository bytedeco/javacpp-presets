/*
 * Milvus embedding store adapter — production ANN for item/user tower embeddings.
 *
 * Uses existing pure-REST MilvusVectorStore (no milvus-sdk-java).
 * Namespace maps to collection name suffix.
 *
 * Feast / Tecton / Alibaba / ByteDance: online features in Redis/KV;
 * retrieval embeddings in Milvus / Faiss / proprietary ANN.
 */
package org.bytedeco.pytorch.utils.feature.store;

import org.bytedeco.pytorch.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.milvus.MilvusVectorStore;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Milvus-backed {@link EmbeddingStore}. */
public final class MilvusEmbeddingStore implements EmbeddingStore {

    private final String url;
    private final String token;
    private final String collectionPrefix;
    private final int dim;
    private final VectorMetric metric;
    private final boolean ownsStores;
    /** namespace → VectorStore */
    private final ConcurrentHashMap<String, VectorStore> stores = new ConcurrentHashMap<>();
    /** Local write-through cache for get() (Milvus get-by-id varies by deployment). */
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, float[]>> cache =
            new ConcurrentHashMap<>();

    public MilvusEmbeddingStore(String url, String token, String collectionPrefix,
                                int dim, VectorMetric metric) {
        this.url = url != null ? url : "http://127.0.0.1:9091";
        this.token = token != null ? token : "";
        this.collectionPrefix = collectionPrefix != null ? collectionPrefix : "feature_embeddings";
        this.dim = Math.max(1, dim);
        this.metric = metric != null ? metric : VectorMetric.COSINE;
        this.ownsStores = true;
    }

    public static MilvusEmbeddingStore connect(String url, int dim) {
        return new MilvusEmbeddingStore(url, "", "feature_embeddings", dim, VectorMetric.COSINE);
    }

    public static MilvusEmbeddingStore connect(StoreConfig cfg) {
        VectorMetric m = parseMetric(cfg.embeddingMetric());
        return new MilvusEmbeddingStore(
                cfg.milvusUrl(),
                cfg.milvusToken(),
                cfg.milvusCollection(),
                cfg.embeddingDim(),
                m);
    }

    /** Probe REST liveness without failing construction. */
    public boolean available() {
        try {
            VectorStore vs = storeFor("default");
            vs.ensureCollection();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private VectorStore storeFor(String namespace) {
        String ns = namespace == null || namespace.isEmpty() ? "default" : namespace;
        return stores.computeIfAbsent(ns, n -> {
            String coll = collectionPrefix + (n.equals("default") ? "" : "_" + sanitize(n));
            MilvusVectorStore.Builder b = MilvusVectorStore.builder(url)
                    .collection(coll)
                    .dim(dim)
                    .metric(metric);
            if (token != null && !token.isEmpty()) {
                b.token(token);
            }
            VectorStore vs = b.build();
            vs.ensureCollection();
            return vs;
        });
    }

    private ConcurrentHashMap<String, float[]> cacheNs(String namespace) {
        String ns = namespace == null || namespace.isEmpty() ? "default" : namespace;
        return cache.computeIfAbsent(ns, k -> new ConcurrentHashMap<>());
    }

    private static String sanitize(String n) {
        return n.replaceAll("[^A-Za-z0-9_]", "_");
    }

    private static VectorMetric parseMetric(String raw) {
        if (raw == null) return VectorMetric.COSINE;
        try {
            return VectorMetric.valueOf(raw.trim().toUpperCase());
        } catch (Exception e) {
            return VectorMetric.COSINE;
        }
    }

    @Override
    public String backend() {
        return "milvus";
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public void put(String namespace, String id, float[] vector) {
        put(namespace, id, vector, null);
    }

    @Override
    public void put(String namespace, String id, float[] vector, Map<String, Object> meta) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(vector, "vector");
        if (vector.length != dim) {
            throw new IllegalArgumentException("embedding dim " + vector.length + " != " + dim);
        }
        VectorStore vs = storeFor(namespace);
        Map<String, Object> payload = meta != null ? new LinkedHashMap<>(meta) : new LinkedHashMap<>();
        payload.put("ns", namespace == null ? "default" : namespace);
        vs.upsert(VectorRecord.of(id, vector, payload));
        cacheNs(namespace).put(id, vector.clone());
    }

    @Override
    public void putBatch(String namespace, Map<String, float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) return;
        VectorStore vs = storeFor(namespace);
        List<VectorRecord> recs = new ArrayList<>(vectors.size());
        for (Map.Entry<String, float[]> e : vectors.entrySet()) {
            if (e.getValue() == null || e.getValue().length != dim) continue;
            recs.add(VectorRecord.of(e.getKey(), e.getValue()));
            cacheNs(namespace).put(e.getKey(), e.getValue().clone());
        }
        if (!recs.isEmpty()) vs.upsert(recs);
    }

    @Override
    public Optional<float[]> get(String namespace, String id) {
        float[] cached = cacheNs(namespace).get(id);
        if (cached != null) return Optional.of(cached.clone());
        // Milvus primary get path — best-effort via cache after put.
        // Production callers also materialize embeddings into online KV.
        return Optional.empty();
    }

    @Override
    public Map<String, float[]> getBatch(String namespace, Collection<String> ids) {
        Map<String, float[]> out = new LinkedHashMap<>();
        if (ids == null) return out;
        ConcurrentHashMap<String, float[]> c = cacheNs(namespace);
        for (String id : ids) {
            float[] v = c.get(id);
            if (v != null) out.put(id, v.clone());
        }
        return out;
    }

    @Override
    public void delete(String namespace, String id) {
        storeFor(namespace).delete(id);
        cacheNs(namespace).remove(id);
    }

    @Override
    public long count(String namespace) {
        try {
            return storeFor(namespace).count();
        } catch (Exception e) {
            return cacheNs(namespace).size();
        }
    }

    @Override
    public List<EmbeddingHit> search(String namespace, float[] query, int topK) {
        if (query == null || topK <= 0) return List.of();
        VectorSearchResult result = storeFor(namespace).search(VectorQuery.of(query, topK));
        List<EmbeddingHit> hits = new ArrayList<>();
        if (result == null || result.hits() == null) return hits;
        for (VectorHit h : result.hits()) {
            float[] vec = cacheNs(namespace).get(h.id());
            hits.add(new EmbeddingHit(h.id(), h.score(), vec != null ? vec.clone() : null));
        }
        return hits;
    }

    @Override
    public void close() {
        if (ownsStores) {
            for (VectorStore vs : stores.values()) {
                try {
                    vs.close();
                } catch (Exception ignored) {
                }
            }
        }
        stores.clear();
        cache.clear();
    }
}
