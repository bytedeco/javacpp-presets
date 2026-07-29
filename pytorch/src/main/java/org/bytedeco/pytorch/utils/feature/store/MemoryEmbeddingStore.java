/*
 * In-process embedding store (ConcurrentHashMap) — default for tests / demos.
 */
package org.bytedeco.pytorch.utils.feature.store;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Memory embedding KV with optional brute-force cosine top-K. */
public final class MemoryEmbeddingStore implements EmbeddingStore {

    private final int dim;
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, float[]>> data =
            new ConcurrentHashMap<>();

    public MemoryEmbeddingStore() {
        this(0);
    }

    public MemoryEmbeddingStore(int dim) {
        this.dim = Math.max(0, dim);
    }

    private ConcurrentHashMap<String, float[]> ns(String namespace) {
        String n = namespace == null || namespace.isEmpty() ? "default" : namespace;
        return data.computeIfAbsent(n, k -> new ConcurrentHashMap<>());
    }

    @Override
    public String backend() {
        return "memory";
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public void put(String namespace, String id, float[] vector) {
        if (id == null || vector == null) return;
        ns(namespace).put(id, vector.clone());
    }

    @Override
    public Optional<float[]> get(String namespace, String id) {
        float[] v = ns(namespace).get(id);
        return v == null ? Optional.empty() : Optional.of(v.clone());
    }

    @Override
    public Map<String, float[]> getBatch(String namespace, Collection<String> ids) {
        Map<String, float[]> out = new LinkedHashMap<>();
        if (ids == null) return out;
        ConcurrentHashMap<String, float[]> m = ns(namespace);
        for (String id : ids) {
            float[] v = m.get(id);
            if (v != null) out.put(id, v.clone());
        }
        return out;
    }

    @Override
    public void delete(String namespace, String id) {
        ns(namespace).remove(id);
    }

    @Override
    public long count(String namespace) {
        return ns(namespace).size();
    }

    @Override
    public List<EmbeddingHit> search(String namespace, float[] query, int topK) {
        if (query == null || topK <= 0) return List.of();
        List<EmbeddingHit> hits = new ArrayList<>();
        double qn = norm(query);
        for (Map.Entry<String, float[]> e : ns(namespace).entrySet()) {
            float[] v = e.getValue();
            if (v.length != query.length) continue;
            double score = qn > 0 && norm(v) > 0 ? dot(query, v) / (qn * norm(v)) : dot(query, v);
            hits.add(new EmbeddingHit(e.getKey(), score, v.clone()));
        }
        hits.sort(Comparator.comparingDouble((EmbeddingHit h) -> h.score).reversed());
        if (hits.size() > topK) return hits.subList(0, topK);
        return hits;
    }

    public void clear() {
        data.clear();
    }

    private static double dot(float[] a, float[] b) {
        double s = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) s += (double) a[i] * b[i];
        return s;
    }

    private static double norm(float[] a) {
        double s = 0;
        for (float x : a) s += (double) x * x;
        return Math.sqrt(s);
    }
}
