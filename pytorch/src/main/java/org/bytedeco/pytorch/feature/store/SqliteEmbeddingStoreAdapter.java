/*
 * SQLite embedding store adapter — wraps utils.sqlite.SQLiteEmbeddingStore
 * with String ids (stable hash) for EmbeddingStore SPI.
 */
package org.bytedeco.pytorch.feature.store;

import org.bytedeco.pytorch.utils.sqlite.SQLiteEmbeddingStore;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** SQLite-backed embeddings with string id mapping. */
public final class SqliteEmbeddingStoreAdapter implements EmbeddingStore {

    private final SQLiteEmbeddingStore store;
    private final boolean owns;
    private final int dim;
    /** ns|stringId → numeric id */
    private final ConcurrentHashMap<String, Long> idMap = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, String> revMap = new ConcurrentHashMap<>();

    public SqliteEmbeddingStoreAdapter(SQLiteEmbeddingStore store, boolean owns) {
        this.store = store;
        this.owns = owns;
        this.dim = store.defaultDim();
    }

    public static SqliteEmbeddingStoreAdapter open(Path dbFile, int dim) {
        try {
            return new SqliteEmbeddingStoreAdapter(SQLiteEmbeddingStore.open(dbFile, dim), true);
        } catch (Exception e) {
            throw new IllegalStateException("cannot open SqliteEmbeddingStore at " + dbFile, e);
        }
    }

    public static SqliteEmbeddingStoreAdapter inMemory(int dim) {
        try {
            return new SqliteEmbeddingStoreAdapter(SQLiteEmbeddingStore.inMemory(dim), true);
        } catch (Exception e) {
            throw new IllegalStateException("cannot open in-memory SqliteEmbeddingStore", e);
        }
    }

    public SQLiteEmbeddingStore raw() {
        return store;
    }

    private static long stableId(String id) {
        // non-negative stable hash
        long h = id.hashCode();
        // mix for fewer collisions across short ids
        h = (h * 0x9E3779B97F4A7C15L) ^ (id.length() * 0xC2B2AE3D27D4EB4FL);
        if (h == Long.MIN_VALUE) return 0L;
        return h < 0 ? -h : h;
    }

    private long mapId(String namespace, String id) {
        String key = (namespace == null ? "default" : namespace) + "|" + id;
        return idMap.computeIfAbsent(key, k -> {
            long nid = stableId(k);
            revMap.put(nid, id);
            return nid;
        });
    }

    @Override
    public String backend() {
        return "sqlite";
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public void put(String namespace, String id, float[] vector) {
        try {
            store.put(namespace == null ? "default" : namespace, mapId(namespace, id), vector);
        } catch (Exception e) {
            throw new IllegalStateException("sqlite embedding put failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Optional<float[]> get(String namespace, String id) {
        try {
            float[] v = store.get(namespace == null ? "default" : namespace, mapId(namespace, id));
            return v == null ? Optional.empty() : Optional.of(v);
        } catch (Exception e) {
            throw new IllegalStateException("sqlite embedding get failed: " + e.getMessage(), e);
        }
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
        try {
            store.delete(namespace == null ? "default" : namespace, mapId(namespace, id));
        } catch (Exception e) {
            throw new IllegalStateException("sqlite embedding delete failed: " + e.getMessage(), e);
        }
    }

    @Override
    public long count(String namespace) {
        try {
            return store.count(namespace == null ? "default" : namespace);
        } catch (Exception e) {
            return -1L;
        }
    }

    @Override
    public List<EmbeddingHit> search(String namespace, float[] query, int topK) {
        try {
            List<SQLiteEmbeddingStore.Hit> raw =
                    store.topKCosine(namespace == null ? "default" : namespace, query, topK);
            List<EmbeddingHit> out = new ArrayList<>();
            for (SQLiteEmbeddingStore.Hit h : raw) {
                String sid = revMap.getOrDefault(h.id, String.valueOf(h.id));
                float[] vec = store.get(namespace == null ? "default" : namespace, h.id);
                out.add(new EmbeddingHit(sid, h.score, vec));
            }
            return out;
        } catch (Exception e) {
            return List.of();
        }
    }

    @Override
    public void close() {
        if (owns) {
            try {
                store.close();
            } catch (Exception ignored) {
            }
        }
    }
}
