package org.bytedeco.pytorch.dataframe.vectorstore.memory;

import org.bytedeco.pytorch.dataframe.ann.AnnSearchResult;
import org.bytedeco.pytorch.dataframe.ann.HnswIndex;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-process {@link VectorStore} backed by pure-Java {@link HnswIndex}.
 * Zero network, zero native deps — useful for unit tests and local RAG demos.
 *
 * <p>Deletes are soft (tombstones): search skips them; {@link #compact()} rebuilds
 * the HNSW without deleted points.
 */
public final class InMemoryVectorStore implements VectorStore {

    private final String name;
    private final int dim;
    private final VectorMetric metric;
    private final int M;
    private final int efConstruction;
    private final boolean normalize;

    private final Object lock = new Object();
    private HnswIndex index;
    private final Map<String, Entry> byId = new ConcurrentHashMap<>();
    private final AtomicLong autoId = new AtomicLong(0);
    private volatile boolean closed;

    public InMemoryVectorStore(String name, int dim, VectorMetric metric) {
        this(name, dim, metric, 16, 200, false);
    }

    public InMemoryVectorStore(String name, int dim, VectorMetric metric,
                               int M, int efConstruction, boolean normalize) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        this.name = name == null ? "memory" : name;
        this.dim = dim;
        this.metric = metric == null ? VectorMetric.L2 : metric;
        this.M = Math.max(2, M);
        this.efConstruction = Math.max(efConstruction, this.M);
        this.normalize = normalize;
        this.index = HnswIndex.builder(dim)
            .M(this.M)
            .efConstruction(this.efConstruction)
            .space(this.metric.toDistance())
            .normalize(this.normalize)
            .build();
    }

    public static Builder builder(int dim) { return new Builder(dim); }

    @Override public String backend() { return "memory"; }
    @Override public String name() { return name; }
    @Override public int dim() { return dim; }
    @Override public VectorMetric metric() { return metric; }

    @Override
    public void ensureCollection() {
        checkOpen();
        // always ready
    }

    @Override
    public void dropCollection() {
        checkOpen();
        synchronized (lock) {
            byId.clear();
            index = HnswIndex.builder(dim)
                .M(M).efConstruction(efConstruction)
                .space(metric.toDistance()).normalize(normalize).build();
            autoId.set(0);
        }
    }

    @Override
    public long count() {
        checkOpen();
        long n = 0;
        for (Entry e : byId.values()) if (!e.deleted) n++;
        return n;
    }

    @Override
    public void upsert(Collection<VectorRecord> records) {
        checkOpen();
        if (records == null || records.isEmpty()) return;
        synchronized (lock) {
            for (VectorRecord r : records) {
                if (r.vector().length != dim) {
                    throw new VectorStoreException(
                        "dim mismatch: got " + r.vector().length + ", expected " + dim, -1, backend());
                }
                String id;
                try {
                    id = r.id() != null && !r.id().isEmpty() ? r.id() : r.resolvedId();
                } catch (IllegalStateException ex) {
                    id = Long.toString(autoId.getAndIncrement());
                }
                Entry prev = byId.get(id);
                if (prev != null) prev.deleted = true;

                long numeric = r.hasNumericId() ? r.numericId() : autoId.getAndIncrement();
                float[] vec = r.vector().clone();
                int pos = index.size();
                index.addOne(vec, numeric);
                Entry e = new Entry(id, numeric, pos, vec, new LinkedHashMap<>(r.payload()));
                byId.put(id, e);
            }
        }
    }

    @Override
    public void delete(Collection<String> ids) {
        checkOpen();
        if (ids == null || ids.isEmpty()) return;
        for (String id : ids) {
            Entry e = byId.get(id);
            if (e != null) e.deleted = true;
        }
    }

    @Override
    public VectorSearchResult search(VectorQuery query) {
        checkOpen();
        Objects.requireNonNull(query, "query");
        if (query.vector().length != dim) {
            throw new VectorStoreException(
                "query dim " + query.vector().length + " != " + dim, -1, backend());
        }
        long t0 = System.nanoTime();
        int ef = query.option("ef", Math.max(64, query.topK() * 2));
        // over-fetch to compensate for tombstones
        int fetch = Math.min(index.size(), Math.max(query.topK() * 4, query.topK() + 16));

        AnnSearchResult raw;
        synchronized (lock) {
            if (index.size() == 0) return VectorSearchResult.empty();
            raw = index.search(query.vector(), fetch, ef);
        }

        // Build reverse map: position → entry (live only)
        Map<Integer, Entry> byPos = new LinkedHashMap<>();
        for (Entry e : byId.values()) {
            if (!e.deleted) byPos.put(e.position, e);
        }

        List<VectorHit> hits = new ArrayList<>(query.topK());
        int[] indices = raw.indices();
        float[] dists = raw.distances();
        for (int i = 0; i < indices.length && hits.size() < query.topK(); i++) {
            Entry e = byPos.get(indices[i]);
            if (e == null) continue;
            float score = dists[i];
            // For IP, Hnsw stores distance = -dot, so score is already lower-is-better distance.
            float[] vecOut = query.includeVector() ? e.vector.clone() : null;
            Map<String, Object> payload = query.includePayload() ? e.payload : Map.of();
            hits.add(new VectorHit(e.id, e.numericId, true, score, score, vecOut, payload));
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    /** Rebuild HNSW without tombstoned points (optional maintenance). */
    public void compact() {
        checkOpen();
        synchronized (lock) {
            List<Entry> live = new ArrayList<>();
            for (Entry e : byId.values()) if (!e.deleted) live.add(e);
            HnswIndex fresh = HnswIndex.builder(dim)
                .M(M).efConstruction(efConstruction)
                .space(metric.toDistance()).normalize(normalize)
                .initialCapacity(Math.max(16, live.size()))
                .build();
            byId.clear();
            for (Entry e : live) {
                int pos = fresh.size();
                fresh.addOne(e.vector, e.numericId);
                Entry ne = new Entry(e.id, e.numericId, pos, e.vector, e.payload);
                byId.put(e.id, ne);
            }
            index = fresh;
        }
    }

    @Override
    public List<VectorRecord> fetch(Collection<String> ids) {
        checkOpen();
        if (ids == null || ids.isEmpty()) return List.of();
        List<VectorRecord> out = new ArrayList<>();
        for (String id : ids) {
            Entry e = byId.get(id);
            if (e == null || e.deleted) continue;
            out.add(VectorRecord.builder()
                .id(e.id)
                .id(e.numericId)
                .vector(e.vector.clone())
                .payload(e.payload)
                .build());
        }
        return out;
    }

    @Override
    public VectorStore.ScrollPage scroll(int limit, Object cursor) {
        checkOpen();
        int lim = Math.max(1, limit);
        int offset = 0;
        if (cursor instanceof Number n) offset = Math.max(0, n.intValue());
        else if (cursor instanceof String s) {
            try { offset = Integer.parseInt(s); } catch (NumberFormatException ignored) {}
        }
        List<VectorRecord> page = new ArrayList<>(Math.min(lim, 64));
        int i = 0;
        for (Entry e : byId.values()) {
            if (e.deleted) continue;
            if (i++ < offset) continue;
            page.add(VectorRecord.builder()
                .id(e.id)
                .id(e.numericId)
                .vector(e.vector.clone())
                .payload(e.payload)
                .build());
            if (page.size() >= lim) break;
        }
        int next = offset + page.size();
        long live = count();
        Object nextCur = next >= live ? null : Integer.valueOf(next);
        return new VectorStore.ScrollPage(page, nextCur);
    }

    @Override
    public void close() {
        closed = true;
        synchronized (lock) {
            byId.clear();
            index = null;
        }
    }

    private void checkOpen() {
        if (closed) throw new VectorStoreException("InMemoryVectorStore closed", -1, backend());
    }

    private static final class Entry {
        final String id;
        final long numericId;
        final int position;
        final float[] vector;
        final Map<String, Object> payload;
        volatile boolean deleted;

        Entry(String id, long numericId, int position, float[] vector, Map<String, Object> payload) {
            this.id = id;
            this.numericId = numericId;
            this.position = position;
            this.vector = vector;
            this.payload = payload;
        }
    }

    public static final class Builder {
        private final int dim;
        private String name = "memory";
        private VectorMetric metric = VectorMetric.L2;
        private int M = 16;
        private int efConstruction = 200;
        private boolean normalize;

        Builder(int dim) { this.dim = dim; }
        public Builder name(String n) { this.name = n; return this; }
        public Builder metric(VectorMetric m) { this.metric = m; return this; }
        public Builder M(int m) { this.M = m; return this; }
        public Builder efConstruction(int ef) { this.efConstruction = ef; return this; }
        public Builder normalize(boolean v) { this.normalize = v; return this; }
        public InMemoryVectorStore build() {
            return new InMemoryVectorStore(name, dim, metric, M, efConstruction, normalize);
        }
    }
}
