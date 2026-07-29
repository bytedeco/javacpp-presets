/*
 * RQ-KMeans — residual K-Means for Semantic ID construction (OneRec industrial path).
 *
 * Reference:
 *   - OneRec uses RQ-KMeans / residual quantization over multimodal+CF embeddings
 *     to produce hierarchical item tokens (typically L=3).
 *   - MiniOneRec https://github.com/AkaliKong/MiniOneRec (rq/rqkmeans_faiss.py)
 *
 * Unlike {@link RQVAE}, this is a pure offline clustering routine (no neural encoder).
 * Fit on item embedding matrix [N, D], then {@link #predict} yields codes [N, L].
 *
 * Implementation: successive residual K-Means with Lloyd iterations on CPU float arrays
 * (no faiss dependency). Suitable for medium catalogs; for million-scale items prefer
 * faiss / external indexing then load codes into {@link SemanticID.Trie}.
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

import java.util.Arrays;
import java.util.Random;

public final class RQKMeans {

    private final int numLevels;
    private final int codebookSize;
    private final int dim;
    private final int maxIters;
    private final long seed;
    /** [L][K][D] cluster centers. */
    private float[][][] centers;
    private boolean fitted;

    public RQKMeans(int numLevels, int codebookSize, int dim) {
        this(numLevels, codebookSize, dim, 25, 42L);
    }

    public RQKMeans(int numLevels, int codebookSize, int dim, int maxIters, long seed) {
        if (numLevels <= 0 || codebookSize <= 0 || dim <= 0) {
            throw new IllegalArgumentException("numLevels/codebookSize/dim must be positive");
        }
        this.numLevels = numLevels;
        this.codebookSize = codebookSize;
        this.dim = dim;
        this.maxIters = Math.max(1, maxIters);
        this.seed = seed;
        this.fitted = false;
    }

    public int numLevels() { return numLevels; }
    public int codebookSize() { return codebookSize; }
    public int dim() { return dim; }
    public boolean isFitted() { return fitted; }

    /** Fit on row-major embeddings [N*D] with N rows. */
    public RQKMeans fit(float[] flatEmbeddings, int numItems) {
        return fit(flatEmbeddings, numItems, 0);
    }

    /**
     * @param fitMaxItems if &gt; 0 and &lt; numItems, fit centers on a random subset then
     *                    assign all items (much faster for 100k+ catalogs).
     */
    public RQKMeans fit(float[] flatEmbeddings, int numItems, int fitMaxItems) {
        if (flatEmbeddings == null || flatEmbeddings.length < numItems * dim) {
            throw new IllegalArgumentException("embeddings too short");
        }
        int fitN = numItems;
        int[] fitIndex = null;
        if (fitMaxItems > 0 && fitMaxItems < numItems) {
            fitN = fitMaxItems;
            fitIndex = new int[fitN];
            // reservoir-ish: stride sample with seed for stability + a bit of jitter
            Random rng = new Random(seed);
            // first fill evenly then shuffle swap
            double step = numItems / (double) fitN;
            for (int i = 0; i < fitN; i++) {
                int idx = Math.min(numItems - 1, (int) (i * step) + rng.nextInt(Math.max(1, (int) step)));
                fitIndex[i] = idx;
            }
        }

        float[][] residual = new float[fitN][dim];
        for (int i = 0; i < fitN; i++) {
            int src = fitIndex == null ? i : fitIndex[i];
            System.arraycopy(flatEmbeddings, src * dim, residual[i], 0, dim);
        }
        centers = new float[numLevels][][];
        Random rng = new Random(seed);
        for (int l = 0; l < numLevels; l++) {
            centers[l] = kmeans(residual, codebookSize, maxIters, rng);
            for (int i = 0; i < fitN; i++) {
                int c = nearest(residual[i], centers[l]);
                for (int d = 0; d < dim; d++) {
                    residual[i][d] -= centers[l][c][d];
                }
            }
        }
        fitted = true;
        return this;
    }

    /** Fit from Tensor [N, D] float. */
    public RQKMeans fit(Tensor embeddings) {
        Tensor cpu = embeddings.to(ScalarType.Float).contiguous().cpu();
        int n = (int) cpu.size(0);
        int d = (int) cpu.size(1);
        if (d != dim) {
            throw new IllegalArgumentException("embedding dim " + d + " != " + dim);
        }
        float[] flat = TensorHelpers.toFloatArray(cpu);
        return fit(flat, n);
    }

    /** Predict codes [numItems][L] for the same (or new) embeddings [N*D]. */
    public int[][] predict(float[] flatEmbeddings, int numItems) {
        ensureFitted();
        float[][] residual = new float[numItems][dim];
        for (int i = 0; i < numItems; i++) {
            System.arraycopy(flatEmbeddings, i * dim, residual[i], 0, dim);
        }
        int[][] codes = new int[numItems][numLevels];
        for (int l = 0; l < numLevels; l++) {
            for (int i = 0; i < numItems; i++) {
                int c = nearest(residual[i], centers[l]);
                codes[i][l] = c;
                for (int d = 0; d < dim; d++) {
                    residual[i][d] -= centers[l][c][d];
                }
            }
        }
        return codes;
    }

    public int[][] predict(Tensor embeddings) {
        Tensor cpu = embeddings.to(ScalarType.Float).contiguous().cpu();
        int n = (int) cpu.size(0);
        float[] flat = TensorHelpers.toFloatArray(cpu);
        return predict(flat, n);
    }

    /** Build a {@link SemanticID.Trie} over all predicted item SIDs. */
    public SemanticID.Trie toTrie(int[][] codes) {
        SemanticID.Trie trie = new SemanticID.Trie(numLevels, codebookSize);
        trie.insertAll(codes);
        return trie;
    }

    private void ensureFitted() {
        if (!fitted) throw new IllegalStateException("RQKMeans not fitted");
    }

    private static int nearest(float[] x, float[][] cents) {
        int best = 0;
        float bestD = Float.POSITIVE_INFINITY;
        for (int k = 0; k < cents.length; k++) {
            float dist = 0f;
            float[] c = cents[k];
            for (int d = 0; d < x.length; d++) {
                float diff = x[d] - c[d];
                dist += diff * diff;
            }
            if (dist < bestD) {
                bestD = dist;
                best = k;
            }
        }
        return best;
    }

    /** Lloyd K-Means; returns [K][D] centers. */
    private static float[][] kmeans(float[][] data, int k, int iters, Random rng) {
        int n = data.length;
        int d = data[0].length;
        k = Math.min(k, n);
        float[][] cents = new float[k][d];
        // init: random distinct points
        int[] chosen = new int[k];
        Arrays.fill(chosen, -1);
        for (int i = 0; i < k; i++) {
            int idx;
            boolean ok;
            do {
                idx = rng.nextInt(n);
                ok = true;
                for (int j = 0; j < i; j++) if (chosen[j] == idx) { ok = false; break; }
            } while (!ok);
            chosen[i] = idx;
            System.arraycopy(data[idx], 0, cents[i], 0, d);
        }
        int[] assign = new int[n];
        float[][] sum = new float[k][d];
        int[] count = new int[k];
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < n; i++) assign[i] = nearest(data[i], cents);
            for (int c = 0; c < k; c++) {
                Arrays.fill(sum[c], 0f);
                count[c] = 0;
            }
            for (int i = 0; i < n; i++) {
                int c = assign[i];
                count[c]++;
                for (int j = 0; j < d; j++) sum[c][j] += data[i][j];
            }
            for (int c = 0; c < k; c++) {
                if (count[c] == 0) {
                    // re-seed empty cluster
                    System.arraycopy(data[rng.nextInt(n)], 0, cents[c], 0, d);
                } else {
                    for (int j = 0; j < d; j++) cents[c][j] = sum[c][j] / count[c];
                }
            }
        }
        return cents;
    }
}
