package org.bytedeco.pytorch.dataframe.vectorstore;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Ranked neighbor list from a vector search.
 */
public final class VectorSearchResult {
    private final List<VectorHit> hits;
    private final long tookMs;

    public VectorSearchResult(List<VectorHit> hits) {
        this(hits, -1L);
    }

    public VectorSearchResult(List<VectorHit> hits, long tookMs) {
        this.hits = hits == null
            ? List.of()
            : Collections.unmodifiableList(new ArrayList<>(hits));
        this.tookMs = tookMs;
    }

    public static VectorSearchResult empty() {
        return new VectorSearchResult(List.of(), 0L);
    }

    public List<VectorHit> hits() { return hits; }
    public int size() { return hits.size(); }
    public boolean isEmpty() { return hits.isEmpty(); }
    public long tookMs() { return tookMs; }

    public VectorHit get(int i) { return hits.get(i); }

    public String[] ids() {
        String[] out = new String[hits.size()];
        for (int i = 0; i < hits.size(); i++) out[i] = hits.get(i).id();
        return out;
    }

    public float[] scores() {
        float[] out = new float[hits.size()];
        for (int i = 0; i < hits.size(); i++) out[i] = hits.get(i).score();
        return out;
    }

    /**
     * Materialize hits as a DataFrame with columns
     * {@code id}, {@code score}, optional {@code distance}, {@code rank}, plus flattened payload keys.
     */
    public DataFrame toDataFrame() {
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("score", Column.DType.FLOAT64);
        df.addColumn("distance", Column.DType.FLOAT64);
        df.addColumn("rank", Column.DType.INT64);

        // union of payload keys (stable insertion order from first occurrence)
        List<String> payloadKeys = new ArrayList<>();
        for (VectorHit h : hits) {
            for (String k : h.payload().keySet()) {
                if (!payloadKeys.contains(k)) payloadKeys.add(k);
            }
        }
        for (String k : payloadKeys) {
            // free-form payload — STRING is the portable "any cell" bucket across DType revisions
            df.addColumn(k, Column.DType.STRING);
        }

        for (int i = 0; i < hits.size(); i++) {
            VectorHit h = hits.get(i);
            int row = df.addEmptyRow();
            df.set(row, "id", h.id());
            df.set(row, "score", (double) h.score());
            df.set(row, "distance", h.distance() == null ? null : h.distance().doubleValue());
            df.set(row, "rank", (long) (i + 1));
            Map<String, Object> p = h.payload();
            for (String k : payloadKeys) {
                df.set(row, k, p.get(k));
            }
        }
        return df;
    }

    @Override
    public String toString() {
        return "VectorSearchResult{k=" + size()
            + (tookMs >= 0 ? ", tookMs=" + tookMs : "")
            + ", hits=" + hits + "}";
    }
}
