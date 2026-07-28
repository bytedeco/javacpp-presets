package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Result of {@link Index#range_search(float[], int, float)} — mirrors FAISS
 * {@code lims, D, I} from {@code range_search}.
 *
 * <p>For query {@code q}, results occupy {@code [lims[q], lims[q+1])}.
 */
public final class RangeSearchResult {
    /** Length {@code nq + 1}; result slice for q is {@code [lims[q], lims[q+1])}. */
    public final long[] lims;
    /** Concatenated distances. */
    public final float[] D;
    /** Concatenated ids. */
    public final long[] I;

    public RangeSearchResult(long[] lims, float[] D, long[] I) {
        this.lims = lims;
        this.D = D;
        this.I = I;
    }

    public int nq() {
        return lims == null || lims.length == 0 ? 0 : lims.length - 1;
    }

    public int count(int q) {
        return (int) (lims[q + 1] - lims[q]);
    }

    public float[] distances(int q) {
        int start = (int) lims[q];
        int end = (int) lims[q + 1];
        float[] out = new float[end - start];
        System.arraycopy(D, start, out, 0, out.length);
        return out;
    }

    public long[] ids(int q) {
        int start = (int) lims[q];
        int end = (int) lims[q + 1];
        long[] out = new long[end - start];
        System.arraycopy(I, start, out, 0, out.length);
        return out;
    }
}
