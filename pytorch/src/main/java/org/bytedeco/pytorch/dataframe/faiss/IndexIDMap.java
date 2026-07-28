package org.bytedeco.pytorch.dataframe.faiss;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * ID-mapping wrapper — mirrors {@code faiss.IndexIDMap(index)}.
 *
 * <pre>
 *   IndexHNSWFlat raw = new IndexHNSWFlat(d, 32);
 *   IndexIDMap index = new IndexIDMap(raw);
 *   index.add_with_ids(vecs, ids);
 *   SearchResult r = index.search(queries, k);  // I[] are business ids
 *   index.remove_ids(new IDSelectorArray(delIds));
 * </pre>
 *
 * <p>Also exposes the inner index as {@link #index} for FAISS-style
 * {@code loaded_index.index.hnsw.efSearch = 96}.
 */
public class IndexIDMap extends Index {
    private static final long serialVersionUID = 1L;

    /** Inner index (FAISS {@code index_with_id.index}). */
    public final Index index;

    /** id_map[i] = external id of vector at inner position i. */
    private long[] idMap;
    private int idSize;
    /** reverse: external id → inner position (first occurrence). */
    private final Map<Long, Integer> rev;

    public IndexIDMap(Index inner) {
        super(inner.d, inner.metric_type);
        this.index = inner;
        this.is_trained = inner.is_trained;
        this.idMap = new long[0];
        this.idSize = 0;
        this.rev = new HashMap<>();
        this.ntotal = inner.ntotal();
    }

    @Override
    public String indexType() {
        return "IDMap:" + index.indexType();
    }

    @Override
    public void train(float[] x, int n) {
        index.train(x, n);
        is_trained = index.is_trained();
    }

    @Override
    public synchronized void add(float[] x, int n) {
        // sequential ids starting at current ntotal
        long[] ids = new long[n];
        long base = ntotal;
        for (int i = 0; i < n; i++) ids[i] = base + i;
        add_with_ids(x, n, ids);
    }

    @Override
    public synchronized void add_with_ids(float[] x, int n, long[] ids) {
        if (n <= 0) return;
        if (ids == null || ids.length < n)
            throw new IllegalArgumentException("ids length < n");
        long innerBefore = index.ntotal();
        index.add(x, n);
        ensureIdCap(idSize + n);
        for (int i = 0; i < n; i++) {
            long ext = ids[i];
            int pos = (int) innerBefore + i;
            idMap[idSize++] = ext;
            rev.put(ext, pos);
        }
        ntotal = index.ntotal();
        is_trained = index.is_trained();
    }

    @Override
    public SearchResult search(float[] xq, int nq, int k) {
        SearchResult raw = index.search(xq, nq, k);
        // translate inner positions → external ids
        long[][] I = new long[raw.nq()][raw.k()];
        for (int q = 0; q < raw.nq(); q++) {
            for (int j = 0; j < raw.k(); j++) {
                long pos = raw.I[q][j];
                if (pos < 0 || pos >= idSize) I[q][j] = -1;
                else I[q][j] = idMap[(int) pos];
            }
        }
        return new SearchResult(raw.D, I);
    }

    @Override
    public RangeSearchResult range_search(float[] xq, int nq, float radius) {
        RangeSearchResult raw = index.range_search(xq, nq, radius);
        long[] I = new long[raw.I.length];
        for (int i = 0; i < raw.I.length; i++) {
            long pos = raw.I[i];
            I[i] = (pos < 0 || pos >= idSize) ? -1 : idMap[(int) pos];
        }
        return new RangeSearchResult(raw.lims, raw.D, I);
    }

    @Override
    public synchronized long remove_ids(IDSelector sel) {
        if (sel == null || idSize == 0) return 0;
        // Collect inner positions to remove
        // Prefer delegating to inner if it supports remove by positional id
        // Build selector on inner positions
        final java.util.BitSet drop = new java.util.BitSet(idSize);
        long removed = 0;
        for (int i = 0; i < idSize; i++) {
            if (sel.is_member(idMap[i])) {
                drop.set(i);
                removed++;
            }
        }
        if (removed == 0) return 0;

        // Rebuild: extract kept vectors via reconstruct and re-add
        // More correct than sparse tombstones for HNSW graph integrity.
        int keep = idSize - (int) removed;
        float[] keptX = new float[keep * d];
        long[] keptIds = new long[keep];
        int w = 0;
        float[] tmp = new float[d];
        for (int i = 0; i < idSize; i++) {
            if (drop.get(i)) continue;
            index.reconstruct(i, tmp);
            System.arraycopy(tmp, 0, keptX, w * d, d);
            keptIds[w] = idMap[i];
            w++;
        }
        // reset inner + id maps
        index.reset();
        // If inner is HNSW/Flat after reset, re-add
        idMap = new long[0];
        idSize = 0;
        rev.clear();
        ntotal = 0;
        if (keep > 0) {
            // retrain not needed for Flat/HNSW; IVFPQ would need care — re-add only
            index.add(keptX, keep);
            ensureIdCap(keep);
            for (int i = 0; i < keep; i++) {
                idMap[idSize] = keptIds[i];
                rev.put(keptIds[i], idSize);
                idSize++;
            }
            ntotal = index.ntotal();
        }
        return removed;
    }

    @Override
    public void reconstruct(long key, float[] recons) {
        // key is external id
        Integer pos = rev.get(key);
        if (pos == null) throw new IllegalArgumentException("unknown id: " + key);
        index.reconstruct(pos, recons);
    }

    @Override
    public synchronized void reset() {
        index.reset();
        idMap = new long[0];
        idSize = 0;
        rev.clear();
        ntotal = 0;
    }

    @Override
    public void to_gpu_storage(int device) {
        index.to_gpu_storage(device);
        markGpu(device);
    }

    @Override
    public void to_cpu_storage() {
        index.to_cpu_storage();
        markCpu();
    }

    public long[] id_map() {
        return Arrays.copyOf(idMap, idSize);
    }

    /**
     * Wrap an already-populated inner index with an external id map
     * (used by {@link NativeFaissIO} when reading {@code IxMp}/{@code IxM2}).
     */
    static IndexIDMap wrapExisting(Index inner, long[] ids) {
        if (inner == null) throw new IllegalArgumentException("inner is null");
        IndexIDMap map = new IndexIDMap(inner);
        if (ids != null && ids.length > 0) {
            map.idMap = Arrays.copyOf(ids, ids.length);
            map.idSize = ids.length;
            map.rev.clear();
            for (int i = 0; i < ids.length; i++) map.rev.put(ids[i], i);
        }
        map.ntotal = inner.ntotal();
        map.is_trained = inner.is_trained();
        map.metric_type = inner.metric_type;
        return map;
    }

    private void ensureIdCap(int need) {
        if (need <= idMap.length) return;
        int nc = Math.max(Math.max(16, idMap.length * 2), need);
        idMap = Arrays.copyOf(idMap, nc);
    }
}
