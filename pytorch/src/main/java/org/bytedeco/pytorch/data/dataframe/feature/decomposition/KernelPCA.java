package org.bytedeco.pytorch.data.dataframe.feature.decomposition;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.data.dataframe.ml.regression.LinearRegression;

import java.util.*;

/**
 * 核主成分分析（Kernel PCA）
 * 对应 sklearn KernelPCA，支持 rbf/linear/poly kernel
 */
public class KernelPCA extends BaseTransformer {
    private int nComponents;
    private String kernel; // "rbf" | "linear" | "poly"
    private double gamma;
    private double degree;
    private double coef0;

    private double[][] trainX;
    private double[][] alphas;  // eigenvectors in kernel space
    private double[] lambdas;   // eigenvalues

    public KernelPCA(int nComponents) { this(nComponents, "rbf", -1, 3, 1); }
    public KernelPCA(int nComponents, String kernel, double gamma, double degree, double coef0) {
        super();
        this.nComponents = nComponents; this.kernel = kernel;
        this.gamma = gamma; this.degree = degree; this.coef0 = coef0;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        trainX = X.to_numpy();
        int n = trainX.length;
        double g = gamma < 0 ? 1.0 / trainX[0].length : gamma;

        // Compute kernel matrix
        double[][] K = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) K[i][j] = kernelFunc(trainX[i], trainX[j], g);
        // Center kernel matrix
        double[] rowMeans = new double[n]; double totalMean = 0;
        for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) rowMeans[i] += K[i][j]; rowMeans[i] /= n; totalMean += rowMeans[i]; }
        totalMean /= n;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
            K[i][j] = K[i][j] - rowMeans[i] - rowMeans[j] + totalMean;

        // Power iteration for top nComponents eigenvectors
        int k = Math.min(nComponents, n);
        alphas = new double[k][n]; lambdas = new double[k];
        double[][] deflated = deepCopy(K);
        Random rng = new Random(42);
        for (int c = 0; c < k; c++) {
            double[] v = randomUnit(n, rng);
            for (int iter = 0; iter < 100; iter++) {
                double[] Kv = matVec(deflated, v);
                double norm = 0; for (double x : Kv) norm += x * x; norm = Math.sqrt(norm);
                if (norm < 1e-12) break;
                for (int i = 0; i < n; i++) v[i] = Kv[i] / norm;
            }
            double[] Kv = matVec(deflated, v);
            lambdas[c] = dotVec(v, Kv);
            alphas[c] = v.clone();
            // Deflate
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) deflated[i][j] -= lambdas[c] * v[i] * v[j];
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("KernelPCA not fitted");
        double[][] newX = X.to_numpy();
        int n = newX.length;
        double g = gamma < 0 ? 1.0 / trainX[0].length : gamma;

        double[][] result = new double[n][nComponents];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < nComponents; c++) {
                double proj = 0;
                for (int j = 0; j < trainX.length; j++) proj += alphas[c][j] * kernelFunc(newX[i], trainX[j], g);
                result[i][c] = lambdas[c] > 0 ? proj / Math.sqrt(Math.abs(lambdas[c])) : proj;
            }
        }
        DataFrame out = DataFrame.create();
        for (int c = 0; c < nComponents; c++) {
            List<Double> col = new ArrayList<>();
            for (double[] row : result) col.add(row[c]);
            out = out.withColumnForDouble("kpca_" + c, col);
        }
        return out;
    }

    private double kernelFunc(double[] a, double[] b, double g) {
        return switch (kernel) {
            case "linear" -> { double s = 0; for (int i = 0; i < a.length; i++) s += a[i]*b[i]; yield s; }
            case "poly"   -> { double s = 0; for (int i = 0; i < a.length; i++) s += a[i]*b[i]; yield Math.pow(g*s + coef0, degree); }
            default       -> { double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]); yield Math.exp(-g*s); }
        };
    }

    private double[] matVec(double[][] A, double[] v) {
        int n = A.length; double[] r = new double[n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) r[i] += A[i][j] * v[j];
        return r;
    }

    private double dotVec(double[] a, double[] b) { double s = 0; for (int i = 0; i < a.length; i++) s += a[i]*b[i]; return s; }

    private double[] randomUnit(int n, Random rng) {
        double[] v = new double[n]; double norm = 0;
        for (int i = 0; i < n; i++) { v[i] = rng.nextGaussian(); norm += v[i]*v[i]; }
        norm = Math.sqrt(norm); for (int i = 0; i < n; i++) v[i] /= norm;
        return v;
    }

    private double[][] deepCopy(double[][] A) {
        double[][] B = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) B[i] = A[i].clone();
        return B;
    }

    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_components", nComponents); p.put("kernel", kernel); p.put("gamma", gamma); return p;
    }
}

