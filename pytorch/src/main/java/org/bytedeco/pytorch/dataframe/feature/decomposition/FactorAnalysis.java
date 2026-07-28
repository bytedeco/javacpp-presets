package org.bytedeco.pytorch.dataframe.feature.decomposition;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.DenseLinalg;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * Factor Analysis with EM updates (sklearn-compatible core).
 *
 * <p>Model: {@code x = W z + μ + ε}, {@code z ~ N(0,I)}, {@code ε ~ N(0, diag(ψ))}.
 * Uses covariance-based EM with real matrix multiplies (no placeholder stubs).
 */
public class FactorAnalysis extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int nComponents;
    private int maxIter;
    private double tol;
    private long randomState = 42L;

    /** loadings_ [nComponents, nFeatures] */
    private double[][] components;
    /** noise variance per feature (ψ) */
    private double[] noiseVariance;
    private double[] mean;
    private double loglike = Double.NaN;

    public FactorAnalysis(int nComponents) {
        this(nComponents, 500, 1e-4);
    }

    public FactorAnalysis(int nComponents, int maxIter, double tol) {
        super();
        this.nComponents = Math.max(1, nComponents);
        this.maxIter = Math.max(1, maxIter);
        this.tol = tol;
    }

    public FactorAnalysis(int nComponents, String... columns) {
        super(columns);
        this.nComponents = Math.max(1, nComponents);
        this.maxIter = 500;
        this.tol = 1e-4;
    }

    public FactorAnalysis setMaxIter(int maxIter) { this.maxIter = maxIter; return this; }
    public FactorAnalysis setTol(double tol) { this.tol = tol; return this; }
    public FactorAnalysis setRandomState(long seed) { this.randomState = seed; return this; }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        String[] cols = columns.toArray(new String[0]);
        double[][] data = FeatureMatrices.fromDf(X, cols);
        int n = data.length;
        int d = cols.length;
        int k = Math.min(nComponents, d);
        nComponents = k;
        if (n < 2) throw new IllegalStateException("FactorAnalysis needs >= 2 samples");

        // NaN -> col mean
        mean = new double[d];
        int[] cnt = new int[d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) {
                double v = data[i][j];
                if (!Double.isNaN(v)) { mean[j] += v; cnt[j]++; }
            }
        for (int j = 0; j < d; j++) mean[j] = cnt[j] == 0 ? 0 : mean[j] / cnt[j];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                if (Double.isNaN(data[i][j])) data[i][j] = mean[j];
        mean = DenseLinalg.mean(data);
        double[][] Xc = DenseLinalg.center(data, mean);

        // Sample covariance S
        double[][] S = DenseLinalg.covariance(Xc, true);

        // Initialize W with top-k PCA loadings * sqrt(eigenvalue)
        DenseLinalg.EigenResult eig = DenseLinalg.eighSymmetric(S);
        double[][] W = new double[d][k]; // sklearn stores components_ as [k,d] but EM easier with [d,k]
        Random rng = new Random(randomState);
        for (int c = 0; c < k; c++) {
            double lam = Math.max(eig.eigenvalues[c], 1e-8);
            double scale = Math.sqrt(lam);
            for (int j = 0; j < d; j++) {
                // eig.vectors are rows
                W[j][c] = eig.vectors[c][j] * scale;
            }
        }
        // slight jitter if degenerate
        for (int j = 0; j < d; j++)
            for (int c = 0; c < k; c++)
                if (Math.abs(W[j][c]) < 1e-15) W[j][c] = 0.01 * rng.nextGaussian();

        noiseVariance = new double[d];
        for (int j = 0; j < d; j++) {
            double w2 = 0;
            for (int c = 0; c < k; c++) w2 += W[j][c] * W[j][c];
            noiseVariance[j] = Math.max(1e-6, S[j][j] - w2);
        }

        double prevLl = Double.NEGATIVE_INFINITY;
        for (int iter = 0; iter < maxIter; iter++) {
            // Ψ^{-1} (diagonal)
            double[] psiInv = new double[d];
            for (int j = 0; j < d; j++) psiInv[j] = 1.0 / (noiseVariance[j] + 1e-12);

            // G = I + W^T Ψ^{-1} W   [k,k]
            double[][] G = new double[k][k];
            for (int a = 0; a < k; a++) {
                for (int b = a; b < k; b++) {
                    double s = (a == b) ? 1.0 : 0.0;
                    for (int j = 0; j < d; j++) s += W[j][a] * psiInv[j] * W[j][b];
                    G[a][b] = G[b][a] = s;
                }
            }
            // Ginv
            double[][] Ginv = invertSPD(G);

            // β = Ginv W^T Ψ^{-1}   [k,d]
            double[][] WtPsi = new double[k][d];
            for (int c = 0; c < k; c++)
                for (int j = 0; j < d; j++)
                    WtPsi[c][j] = W[j][c] * psiInv[j];
            double[][] beta = DenseLinalg.matmul(Ginv, WtPsi); // [k,d]

            // Ezz = Ginv + beta S beta^T
            double[][] betaS = DenseLinalg.matmul(beta, S); // [k,d]
            double[][] betaSbt = DenseLinalg.matmulBT(betaS, beta); // [k,k]
            double[][] Ezz = new double[k][k];
            for (int a = 0; a < k; a++)
                for (int b = 0; b < k; b++)
                    Ezz[a][b] = Ginv[a][b] + betaSbt[a][b];

            // Exz^T = S beta^T  => [d,k]
            double[][] ExzT = DenseLinalg.matmulBT(S, beta); // S @ beta^T = [d,k]

            // W_new = ExzT @ Ezz^{-1}
            double[][] EzzInv = invertSPD(Ezz);
            double[][] Wnew = DenseLinalg.matmul(ExzT, EzzInv); // [d,k]

            // ψ_j = S_jj - (ExzT @ W_new^T)_jj  = S_jj - sum_c ExzT[j,c] * Wnew[j,c]
            double[] psiNew = new double[d];
            for (int j = 0; j < d; j++) {
                double s = S[j][j];
                for (int c = 0; c < k; c++) s -= ExzT[j][c] * Wnew[j][c];
                psiNew[j] = Math.max(1e-6, s);
            }

            W = Wnew;
            noiseVariance = psiNew;

            // rough loglik monitor: -0.5 * n * (log det Σ + tr(Σ^{-1} S)) with Σ = WW^T + Ψ
            double ll = approxLoglik(S, W, noiseVariance, n);
            if (iter > 0 && Math.abs(ll - prevLl) / (Math.abs(prevLl) + 1e-12) < tol) {
                loglike = ll;
                break;
            }
            prevLl = ll;
            loglike = ll;
        }

        // store components as [k, d]
        components = new double[k][d];
        for (int c = 0; c < k; c++)
            for (int j = 0; j < d; j++)
                components[c][j] = W[j][c];

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        String[] cols = columns.toArray(new String[0]);
        double[][] data = FeatureMatrices.fromDf(X, cols);
        int n = data.length;
        int d = cols.length;
        int k = components.length;

        // posterior mean of z: β (x-μ) with β = Ginv W^T Ψinv
        double[] psiInv = new double[d];
        for (int j = 0; j < d; j++) psiInv[j] = 1.0 / (noiseVariance[j] + 1e-12);

        // W is [k,d] in components; build W_dk [d,k]
        double[][] W = new double[d][k];
        for (int c = 0; c < k; c++)
            for (int j = 0; j < d; j++) W[j][c] = components[c][j];

        double[][] G = new double[k][k];
        for (int a = 0; a < k; a++) {
            for (int b = a; b < k; b++) {
                double s = (a == b) ? 1.0 : 0.0;
                for (int j = 0; j < d; j++) s += W[j][a] * psiInv[j] * W[j][b];
                G[a][b] = G[b][a] = s;
            }
        }
        double[][] Ginv = invertSPD(G);
        double[][] WtPsi = new double[k][d];
        for (int c = 0; c < k; c++)
            for (int j = 0; j < d; j++)
                WtPsi[c][j] = W[j][c] * psiInv[j];
        double[][] beta = DenseLinalg.matmul(Ginv, WtPsi); // [k,d]

        DataFrame result = X.copy();
        for (int c = 0; c < k; c++) {
            String name = FeatureMatrices.uniqueName(result, "FA" + (c + 1));
            result.addColumn(name, Column.DType.FLOAT64);
            Column col = result.column(name);
            while (col.size() < n) col.add(null);
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < d; j++) {
                    double v = data[i][j];
                    if (Double.isNaN(v)) v = mean[j];
                    s += beta[c][j] * (v - mean[j]);
                }
                col.set(i, s);
            }
        }
        return result;
    }

    private static double[][] invertSPD(double[][] A) {
        int n = A.length;
        double[][] inv = new double[n][n];
        for (int j = 0; j < n; j++) {
            double[] e = new double[n];
            e[j] = 1.0;
            double[] col = DenseLinalg.solve(A, e);
            for (int i = 0; i < n; i++) inv[i][j] = col[i];
        }
        return inv;
    }

    /** Cheap log-likelihood proxy for convergence (not exact constant terms). */
    private static double approxLoglik(double[][] S, double[][] W, double[] psi, int n) {
        int d = psi.length;
        int k = W[0].length;
        // Σ ≈ WW^T + Ψ
        double[][] Sigma = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                double s = (i == j) ? psi[i] : 0.0;
                for (int c = 0; c < k; c++) s += W[i][c] * W[j][c];
                Sigma[i][j] = Sigma[j][i] = s;
            }
        }
        // log det via eigh
        DenseLinalg.EigenResult eig = DenseLinalg.eighSymmetric(Sigma);
        double logdet = 0;
        for (double ev : eig.eigenvalues) logdet += Math.log(Math.max(ev, 1e-12));
        // tr(Σ^{-1} S) via solve each column
        double tr = 0;
        for (int j = 0; j < d; j++) {
            double[] col = new double[d];
            for (int i = 0; i < d; i++) col[i] = S[i][j];
            double[] solved = DenseLinalg.solve(Sigma, col);
            tr += solved[j];
        }
        return -0.5 * n * (logdet + tr);
    }

    public double[][] getComponents() { return components; }
    public double[] getNoiseVariance() { return noiseVariance; }
    public double[] getMean() { return mean; }
    public double getLoglike() { return loglike; }
    public int getNComponents() { return nComponents; }

    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_components", nComponents);
        p.put("max_iter", maxIter);
        p.put("tol", tol);
        return p;
    }
}
