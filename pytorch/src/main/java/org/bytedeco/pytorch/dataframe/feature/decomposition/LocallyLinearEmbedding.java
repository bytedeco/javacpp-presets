package org.bytedeco.pytorch.dataframe.feature.decomposition;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.ml.regression.LinearRegression;

import java.util.*;

/**
 * 局部线性嵌入（LLE – Locally Linear Embedding）
 * 对应 sklearn LocallyLinearEmbedding，非线性流形学习
 */
public class LocallyLinearEmbedding extends BaseTransformer {
    private int nComponents;
    private int nNeighbors;
    private int maxIter;

    private double[][] embedding; // fitted embedding (only for transform on same data)

    public LocallyLinearEmbedding(int nComponents) { this(nComponents, 5, 100); }
    public LocallyLinearEmbedding(int nComponents, int nNeighbors, int maxIter) {
        super(); this.nComponents = nComponents; this.nNeighbors = nNeighbors; this.maxIter = maxIter;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        double[][] data = X.to_numpy();
        embedding = computeEmbedding(data);
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("LocallyLinearEmbedding not fitted");
        // For new data, use the stored embedding (exact transform requires more computation)
        double[][] data = X.to_numpy();
        if (data.length == embedding.length) return toDataFrame(embedding);
        // Approximate: re-compute embedding
        return toDataFrame(computeEmbedding(data));
    }

    private double[][] computeEmbedding(double[][] X) {
        int n = X.length, d = X[0].length;
        int k = Math.min(nNeighbors, n - 1);

        // Step 1: Find k nearest neighbors
        int[][] knn = new int[n][k];
        for (int i = 0; i < n; i++) {
            double[] dists = new double[n];
            for (int j = 0; j < n; j++) dists[j] = dist(X[i], X[j]);
            Integer[] idx = new Integer[n]; for (int j = 0; j < n; j++) idx[j] = j;
            Arrays.sort(idx, Comparator.comparingDouble(a -> dists[a]));
            for (int j = 0; j < k; j++) knn[i][j] = idx[j + 1]; // skip self
        }

        // Step 2: Compute reconstruction weights
        double[][] W = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[][] C = new double[k][k]; // local covariance
            double[] z = new double[d];
            for (int j = 0; j < k; j++) for (int jj = 0; jj < d; jj++) z[jj] += X[knn[i][j]][jj] / k;
            for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) {
                for (int jj = 0; jj < d; jj++)
                    C[a][b] += (X[knn[i][a]][jj] - X[i][jj]) * (X[knn[i][b]][jj] - X[i][jj]);
                if (a == b) C[a][b] += 1e-3; // regularize
            }
            double[] w = solveUniform(C, k); // all-ones RHS
            double wSum = 0; for (double v : w) wSum += v;
            for (int j = 0; j < k; j++) W[i][knn[i][j]] = wSum == 0 ? 1.0/k : w[j] / wSum;
        }

        // Step 3: Compute low-dim embedding via bottom eigenvectors of M = (I-W)^T (I-W)
        // Use power iteration on M^T M
        double[][] result = new double[n][nComponents];
        Random rng = new Random(42);
        for (int c = 0; c < nComponents; c++) {
            double[] v = randomUnit(n, rng);
            for (int iter = 0; iter < maxIter; iter++) {
                double[] Mv = applyM(W, v, n);
                double norm = 0; for (double x : Mv) norm += x*x; norm = Math.sqrt(norm + 1e-12);
                for (int i = 0; i < n; i++) v[i] = Mv[i] / norm;
            }
            for (int i = 0; i < n; i++) result[i][c] = v[i];
        }
        return result;
    }

    private double[] applyM(double[][] W, double[] v, int n) {
        // (I - W)v
        double[] r = new double[n];
        for (int i = 0; i < n; i++) { r[i] = v[i]; for (int j = 0; j < n; j++) r[i] -= W[i][j] * v[j]; }
        return r;
    }

    private double[] solveUniform(double[][] C, int k) {
        double[] b = new double[k]; Arrays.fill(b, 1.0);
        // Gaussian elimination
        return LinearRegression.gaussianElimination(C, b);
    }

    private double dist(double[] a, double[] b) {
        double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]); return Math.sqrt(s);
    }

    private double[] randomUnit(int n, Random rng) {
        double[] v = new double[n]; double norm = 0;
        for (int i = 0; i < n; i++) { v[i] = rng.nextGaussian(); norm += v[i]*v[i]; }
        norm = Math.sqrt(norm); for (int i = 0; i < n; i++) v[i] /= norm;
        return v;
    }

    private DataFrame toDataFrame(double[][] embed) throws Exception {
        DataFrame out = DataFrame.create();
        for (int c = 0; c < embed[0].length; c++) {
            final int ci = c; List<Double> col = new ArrayList<>();
            for (double[] row : embed) col.add(row[ci]);
            out = out.withColumnForDouble("lle_" + c, col);
        }
        return out;
    }

    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_components", nComponents); p.put("n_neighbors", nNeighbors); return p;
    }
}

