package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Result of {@link Index#search(float[], int, int)} — mirrors FAISS {@code (D, I)}.
 *
 * <p>{@code D[q][j]} is the distance/score of neighbor {@code j} for query {@code q};
 * {@code I[q][j]} is the corresponding id (or row index when no ID map).
 */
public final class SearchResult {
    /** Distances / scores, shape {@code [nq][k]}. */
    public final float[][] D;
    /** Ids / indices, shape {@code [nq][k]}. Missing slots are {@code -1}. */
    public final long[][] I;

    public SearchResult(float[][] D, long[][] I) {
        this.D = D;
        this.I = I;
    }

    public int nq() {
        return D == null ? 0 : D.length;
    }

    public int k() {
        return D == null || D.length == 0 || D[0] == null ? 0 : D[0].length;
    }

    /** Single-query convenience: distances of query 0. */
    public float[] distances() {
        return nq() == 0 ? new float[0] : D[0];
    }

    /** Single-query convenience: ids of query 0. */
    public long[] ids() {
        return nq() == 0 ? new long[0] : I[0];
    }

    /** Pack as row-major flat arrays (FAISS C-style layout helpers). */
    public float[] flatD() {
        int n = nq(), k = k();
        float[] out = new float[n * k];
        for (int q = 0; q < n; q++) {
            System.arraycopy(D[q], 0, out, q * k, k);
        }
        return out;
    }

    public long[] flatI() {
        int n = nq(), k = k();
        long[] out = new long[n * k];
        for (int q = 0; q < n; q++) {
            System.arraycopy(I[q], 0, out, q * k, k);
        }
        return out;
    }
}
