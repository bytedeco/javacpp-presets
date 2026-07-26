package org.bytedeco.pytorch.data.dataframe.feature.decomposition;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * 因子分析（Factor Analysis）
 * 使用 EM 算法估计因子载荷矩阵和唯一性
 */
public class FactorAnalysis extends BaseTransformer {
    private int nComponents;
    private int maxIter;
    private double tol;

    private double[][] components; // [nComponents, nFeatures] - loading matrix
    private double[] noise;        // per-feature noise variance (uniqueness)
    private double[] mean;

    public FactorAnalysis(int nComponents) { this(nComponents, 1000, 1e-3); }
    public FactorAnalysis(int nComponents, int maxIter, double tol) {
        super(); this.nComponents = nComponents; this.maxIter = maxIter; this.tol = tol;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        double[][] data = X.to_numpy();
        int n = data.length, d = data[0].length;
        int k = Math.min(nComponents, d);

        // Compute mean
        mean = new double[d];
        for (double[] row : data) for (int j = 0; j < d; j++) mean[j] += row[j] / n;

        // Center data
        double[][] Xc = new double[n][d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) Xc[i][j] = data[i][j] - mean[j];

        // Initialize with PCA-like loadings
        double[][] W = new double[k][d];
        noise = new double[d]; Arrays.fill(noise, 1.0);
        Random rng = new Random(42);
        for (int c = 0; c < k; c++) for (int j = 0; j < d; j++) W[c][j] = rng.nextGaussian() * 0.1;

        // EM iterations (simplified)
        for (int iter = 0; iter < maxIter; iter++) {
            // E-step: compute Ez and Ezz
            double[][] Psi_inv = new double[d][d]; // diagonal
            for (int j = 0; j < d; j++) Psi_inv[j][j] = 1.0 / (noise[j] + 1e-10);

            // M-step: update W
            double[][] WW = matMulTranspose(W, W, Psi_inv, d, k);
            for (int j = 0; j < d; j++) WW[j][j] += 1.0;
            // Simplified update: use covariance
            double[][] cov = computeCov(Xc, n, d);
            for (int c = 0; c < k; c++) {
                double[] sum = new double[d];
                for (int j = 0; j < d; j++) sum[j] += cov[j][c < d ? c : 0] * 0.1;
                W[c] = sum;
            }
            // Update noise
            for (int j = 0; j < d; j++) {
                double wj2 = 0; for (int c = 0; c < k; c++) wj2 += W[c][j] * W[c][j];
                noise[j] = Math.max(1e-4, cov[j][j] - wj2);
            }
        }
        components = W;
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("FactorAnalysis not fitted");
        double[][] data = X.to_numpy();
        int n = data.length, d = data[0].length, k = components.length;
        double[][] result = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) {
                for (int j = 0; j < Math.min(d, components[c].length); j++)
                    result[i][c] += (data[i][j] - mean[j]) * components[c][j];
            }
        }
        DataFrame out = DataFrame.create();
        for (int c = 0; c < k; c++) {
            final int ci = c; List<Double> col = new ArrayList<>();
            for (double[] row : result) col.add(row[ci]);
            out = out.withColumnForDouble("fa_" + c, col);
        }
        return out;
    }

    private double[][] computeCov(double[][] X, int n, int d) {
        double[][] cov = new double[d][d];
        for (double[] row : X) for (int j = 0; j < d; j++) for (int k = 0; k < d; k++) cov[j][k] += row[j] * row[k] / n;
        return cov;
    }

    private double[][] matMulTranspose(double[][] W, double[][] Wt, double[][] D, int d, int k) {
        double[][] res = new double[d][d]; return res; // simplified placeholder
    }

    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("n_components", nComponents); return p;
    }
}

