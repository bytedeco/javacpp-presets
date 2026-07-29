/*
 * Redis Stack / RediSearch vector embedding store adapter.
 * Uses RedisVectorStore pure-RESP client (no Jedis).
 */
package org.bytedeco.pytorch.utils.feature.store;

import org.bytedeco.pytorch.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.redis.RedisVectorStore;

import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Redis-vector {@link EmbeddingStore}. */
public final class RedisVectorEmbeddingStore implements EmbeddingStore {

    private final String host;
    private final int port;
    private final String indexPrefix;
    private final int dim;
    private final VectorMetric metric;
    private final ConcurrentHashMap<String, VectorStore> stores = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, float[]>> cache =
            new ConcurrentHashMap<>();

    public RedisVectorEmbeddingStore(String host, int port, String indexPrefix,
                                     int dim, VectorMetric metric) {
        this.host = host != null ? host : "127.0.0.1";
        this.port = port > 0 ? port : 6379;
        this.indexPrefix = indexPrefix != null ? indexPrefix : "fs:emb:";
        this.dim = Math.max(1, dim);
        this.metric = metric != null ? metric : VectorMetric.COSINE;
    }

    public static RedisVectorEmbeddingStore connect(String redisUri, int dim) {
        String host = "127.0.0.1";
        int port = 6379;
        try {
            String u = redisUri == null ? "redis://127.0.0.1:6379" : redisUri;
            if (!u.contains("://")) u = "redis://" + u;
            URI uri = URI.create(u);
            if (uri.getHost() != null) host = uri.getHost();
            if (uri.getPort() > 0) port = uri.getPort();
        } catch (Exception ignored) {
        }
        return new RedisVectorEmbeddingStore(host, port, "fs:emb:", dim, VectorMetric.COSINE);
    }

    public static RedisVectorEmbeddingStore connect(StoreConfig cfg) {
        return connect(cfg.redisUri(), cfg.embeddingDim());
    }

    public boolean available() {
        try {
            storeFor("default").ensureCollection();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private VectorStore storeFor(String namespace) {
        String ns = namespace == null || namespace.isEmpty() ? "default" : namespace;
        return stores.computeIfAbsent(ns, n -> {
            String index = indexPrefix + n;
            VectorStore vs = RedisVectorStore.builder()
                    .host(host)
                    .port(port)
                    .index(index)
                    .prefix(index + ":")
                    .dim(dim)
                    .metric(metric)
                    .build();
            vs.ensureCollection();
            return vs;
        });
    }

    private ConcurrentHashMap<String, float[]> cacheNs(String namespace) {
        String ns = namespace == null || namespace.isEmpty() ? "default" : namespace;
        return cache.computeIfAbsent(ns, k -> new ConcurrentHashMap<>());
    }

    @Override
    public String backend() {
        return "redis_vector";
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public void put(String namespace, String id, float[] vector) {
        if (id == null || vector == null) return;
        storeFor(namespace).upsert(VectorRecord.of(id, vector));
        cacheNs(namespace).put(id, vector.clone());
    }

    @Override
    public void putBatch(String namespace, Map<String, float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) return;
        List<VectorRecord> recs = new ArrayList<>();
        for (Map.Entry<String, float[]> e : vectors.entrySet()) {
            if (e.getValue() == null) continue;
            recs.add(VectorRecord.of(e.getKey(), e.getValue()));
            cacheNs(namespace).put(e.getKey(), e.getValue().clone());
        }
        if (!recs.isEmpty()) storeFor(namespace).upsert(recs);
    }

    @Override
    public Optional<float[]> get(String namespace, String id) {
        float[] v = cacheNs(namespace).get(id);
        return v == null ? Optional.empty() : Optional.of(v.clone());
    }

    @Override
    public Map<String, float[]> getBatch(String namespace, Collection<String> ids) {
        Map<String, float[]> out = new LinkedHashMap<>();
        if (ids == null) return out;
        for (String id : ids) {
            get(namespace, id).ifPresent(v -> out.put(id, v));
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
        try {
            VectorSearchResult result = storeFor(namespace).search(VectorQuery.of(query, topK));
            List<EmbeddingHit> hits = new ArrayList<>();
            if (result == null) return hits;
            for (VectorHit h : result.hits()) {
                float[] vec = h.vector() != null ? h.vector() : cacheNs(namespace).get(h.id());
                hits.add(new EmbeddingHit(h.id(), h.score(), vec != null ? vec.clone() : null));
            }
            return hits;
        } catch (Exception e) {
            return List.of();
        }
    }

    @Override
    public void close() {
        for (VectorStore vs : stores.values()) {
            try {
                vs.close();
            } catch (Exception ignored) {
            }
        }
        stores.clear();
        cache.clear();
    }
}
