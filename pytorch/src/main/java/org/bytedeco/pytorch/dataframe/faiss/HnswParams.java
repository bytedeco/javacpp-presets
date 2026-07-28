package org.bytedeco.pytorch.dataframe.faiss;

import java.io.Serializable;

/**
 * Mutable HNSW hyper-parameters — mirrors {@code index.hnsw.efConstruction / efSearch}.
 *
 * <pre>
 *   IndexHNSWFlat index = new IndexHNSWFlat(d, 32);
 *   index.hnsw.efConstruction = 128;
 *   index.hnsw.efSearch = 64;
 * </pre>
 */
public final class HnswParams implements Serializable {
    private static final long serialVersionUID = 1L;

    /** Max degree on layers &gt; 0 (FAISS {@code M}). */
    public final int M;
    /** Max degree on layer 0 (= 2*M in FAISS). */
    public final int maxM0;
    /** Build-time beam width. */
    public int efConstruction;
    /** Query-time beam width (hot-updatable). */
    public int efSearch;

    public HnswParams(int M) {
        this(M, 40, 16);
    }

    public HnswParams(int M, int efConstruction, int efSearch) {
        if (M < 2) throw new IllegalArgumentException("M must be >= 2");
        this.M = M;
        this.maxM0 = M * 2;
        this.efConstruction = Math.max(efConstruction, M);
        this.efSearch = Math.max(efSearch, 1);
    }
}
