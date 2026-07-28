package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Bounded top-k collector. For L2 (lower better) keeps smallest k;
 * for IP (higher better) keeps largest k.
 */
public final class TopK {
    private final int k;
    private final boolean lowerIsBetter;
    private final float[] dist;
    private final long[] id;
    private int size;

    public TopK(int k, boolean lowerIsBetter) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0");
        this.k = k;
        this.lowerIsBetter = lowerIsBetter;
        this.dist = new float[k];
        this.id = new long[k];
        this.size = 0;
        float init = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            dist[i] = init;
            id[i] = -1;
        }
    }

    public void offer(long idx, float d) {
        if (size < k) {
            dist[size] = d;
            id[size] = idx;
            size++;
            // bubble up toward correct order: maintain worst at end for quick reject
            siftUpNew();
            return;
        }
        // full: compare against worst
        if (lowerIsBetter) {
            if (d >= dist[size - 1]) return;
        } else {
            if (d <= dist[size - 1]) return;
        }
        dist[size - 1] = d;
        id[size - 1] = idx;
        // re-sort insertion into position
        int i = size - 1;
        if (lowerIsBetter) {
            while (i > 0 && dist[i] < dist[i - 1]) {
                swap(i, i - 1);
                i--;
            }
        } else {
            while (i > 0 && dist[i] > dist[i - 1]) {
                swap(i, i - 1);
                i--;
            }
        }
    }

    private void siftUpNew() {
        int i = size - 1;
        if (lowerIsBetter) {
            while (i > 0 && dist[i] < dist[i - 1]) {
                swap(i, i - 1);
                i--;
            }
        } else {
            while (i > 0 && dist[i] > dist[i - 1]) {
                swap(i, i - 1);
                i--;
            }
        }
    }

    private void swap(int a, int b) {
        float td = dist[a]; dist[a] = dist[b]; dist[b] = td;
        long ti = id[a]; id[a] = id[b]; id[b] = ti;
    }

    /** Worst (boundary) distance currently held; used for early reject. */
    public float worst() {
        if (size == 0) {
            return lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }
        return dist[size - 1];
    }

    public boolean isFull() { return size >= k; }

    public int size() { return size; }

    /** Sorted best→worst into out arrays (length k, pad with init / -1). */
    public void export(float[] outD, long[] outI) {
        int n = Math.min(k, outD.length);
        for (int i = 0; i < n; i++) {
            if (i < size) {
                outD[i] = dist[i];
                outI[i] = id[i];
            } else {
                outD[i] = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
                outI[i] = -1;
            }
        }
    }

    public static SearchResult toSearchResult(TopK[] perQuery, int k) {
        int nq = perQuery.length;
        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];
        for (int q = 0; q < nq; q++) {
            perQuery[q].export(D[q], I[q]);
        }
        return new SearchResult(D, I);
    }
}
