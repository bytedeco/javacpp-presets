package org.bytedeco.pytorch.data.dataframe.ann;

import java.util.Arrays;

/**
 * Result of an ANN k-NN search.
 */
public final class AnnSearchResult {
    private final int[] indices;     // row indices into original DataFrame / vector array
    private final long[] ids;        // optional external ids (may be null)
    private final float[] distances; // distance values (lower is better for L2/COSINE; -IP for IP)

    public AnnSearchResult(int[] indices, float[] distances, long[] ids) {
        this.indices = indices;
        this.distances = distances;
        this.ids = ids;
    }

    public int[] indices() { return indices; }
    public float[] distances() { return distances; }
    public long[] ids() { return ids; }
    public int size() { return indices == null ? 0 : indices.length; }

    @Override
    public String toString() {
        return "AnnSearchResult{k=" + size()
            + ", indices=" + Arrays.toString(indices)
            + ", distances=" + Arrays.toString(distances) + "}";
    }
}
