package org.bytedeco.pytorch.dataframe.feature.decomposition;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.DenseLinalg;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.Random;

/**
 * Non-negative Matrix Factorization (sklearn-style multiplicative updates).
 *
 * <p>Factorizes {@code X ≈ W @ H} with W,H ≥ 0.
 * {@code fit} learns H (components); {@code transform} solves for W on new X with H fixed.
 */
public class NMF extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int nComponents;
    private int maxIter = 200;
    private double tol = 1e-4;
    private long randomState = 42L;
    private double eps = 1e-10;

    /** components_ : [nComponents, nFeatures]  (== H) */
    private double[][] components;
    /** reconstruction error on training set after fit */
    private double reconstructionErr = Double.NaN;

    public NMF(int nComponents, String... columns) {
        super(columns);
        this.nComponents = Math.max(1, nComponents);
    }

    public NMF setMaxIter(int maxIter) { this.maxIter = Math.max(1, maxIter); return this; }
    public NMF setTol(double tol) { this.tol = tol; return this; }
    public NMF setRandomState(long seed) { this.randomState = seed; return this; }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        double[][] V = extractNonNeg(X);
        int n = V.length;
        int d = V[0].length;
        int k = Math.min(nComponents, Math.min(n, d));
        nComponents = k;

        Random rng = new Random(randomState);
        // W: [n, k], H: [k, d]
        double[][] W = new double[n][k];
        double[][] H = new double[k][d];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < k; c++)
                W[i][c] = Math.abs(rng.nextGaussian()) + eps;
        for (int c = 0; c < k; c++)
            for (int j = 0; j < d; j++)
                H[c][j] = Math.abs(rng.nextGaussian()) + eps;

        double prevErr = Double.POSITIVE_INFINITY;
        for (int iter = 0; iter < maxIter; iter++) {
            // H <- H * (W^T V) / (W^T W H)
            updateH(V, W, H);
            // W <- W * (V H^T) / (W H H^T)
            updateW(V, W, H);

            if (iter % 10 == 0 || iter == maxIter - 1) {
                double err = frobeniusLoss(V, W, H);
                if (Math.abs(prevErr - err) / (prevErr + 1e-12) < tol) {
                    prevErr = err;
                    break;
                }
                prevErr = err;
            }
        }
        reconstructionErr = prevErr;
        components = H; // [k, d]
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        double[][] V = extractNonNeg(X);
        int n = V.length;
        int k = components.length;
        int d = components[0].length;

        // Solve W with H fixed via multiplicative updates (warm start random)
        Random rng = new Random(randomState + 7);
        double[][] W = new double[n][k];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < k; c++)
                W[i][c] = Math.abs(rng.nextGaussian()) + eps;

        double[][] H = components;
        int tIter = Math.min(maxIter, 100);
        for (int iter = 0; iter < tIter; iter++) {
            updateW(V, W, H);
        }

        DataFrame result = X.copy();
        for (int c = 0; c < k; c++) {
            String name = FeatureMatrices.uniqueName(result, "NMF_" + (c + 1));
            result.addColumn(name, Column.DType.FLOAT64);
            Column col = result.column(name);
            while (col.size() < n) col.add(null);
            for (int i = 0; i < n; i++) col.set(i, W[i][c]);
        }
        return result;
    }

    /**
     * Multiplicative update for H (Lee & Seung):
     * H <- H ⊙ (WᵀV) ⊘ (WᵀWH + eps)
     */
    private void updateH(double[][] V, double[][] W, double[][] H) {
        int n = V.length, d = V[0].length, k = H.length;
        // numerator = W^T V   -> [k, d]
        double[][] num = DenseLinalg.matmulAT(W, V);
        // WtW = W^T W -> [k, k]
        double[][] WtW = DenseLinalg.matmulAT(W, W);
        // den = WtW @ H -> [k, d]
        double[][] den = DenseLinalg.matmul(WtW, H);
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < d; j++) {
                H[c][j] *= num[c][j] / (den[c][j] + eps);
                if (H[c][j] < eps) H[c][j] = eps;
            }
        }
    }

    /**
     * Multiplicative update for W:
     * W <- W ⊙ (V Hᵀ) ⊘ (W H Hᵀ + eps)
     */
    private void updateW(double[][] V, double[][] W, double[][] H) {
        int n = V.length, k = W[0].length;
        // numerator = V @ H^T -> [n, k]
        double[][] num = DenseLinalg.matmulBT(V, H);
        // HHt = H @ H^T -> [k, k]
        double[][] HHt = DenseLinalg.matmulBT(H, H);
        // den = W @ HHt -> [n, k]
        double[][] den = DenseLinalg.matmul(W, HHt);
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) {
                W[i][c] *= num[i][c] / (den[i][c] + eps);
                if (W[i][c] < eps) W[i][c] = eps;
            }
        }
    }

    private double frobeniusLoss(double[][] V, double[][] W, double[][] H) {
        // ||V - WH||_F^2
        double[][] WH = DenseLinalg.matmul(W, H);
        double s = 0;
        for (int i = 0; i < V.length; i++) {
            for (int j = 0; j < V[0].length; j++) {
                double d = V[i][j] - WH[i][j];
                s += d * d;
            }
        }
        return s;
    }

    private double[][] extractNonNeg(DataFrame X) {
        String[] cols = columns.toArray(new String[0]);
        double[][] m = FeatureMatrices.fromDf(X, cols);
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                double v = m[i][j];
                if (Double.isNaN(v) || v < 0) m[i][j] = 0;
            }
        }
        return m;
    }

    public double[][] getComponents() { return components; }
    public double getReconstructionErr() { return reconstructionErr; }
    public int getNComponents() { return nComponents; }
}
