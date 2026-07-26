package org.bytedeco.pytorch.data.dataframe.ml.anomaly;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.data.dataframe.ml.regression.LinearRegression;
import java.util.*;

/** 椭圆包络（Mahalanobis 距离异常检测） */
public class EllipticEnvelope extends BaseClassifier {
    private double contamination; private double threshold;
    private double[] mean; private double[][] precisionMatrix;

    public EllipticEnvelope() { this(0.1); }
    public EllipticEnvelope(double contamination) { this.contamination = contamination; }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        mean = new double[d];
        for (double[] row : X) for (int j = 0; j < d; j++) mean[j] += row[j] / n;
        // Compute covariance matrix
        double[][] cov = new double[d][d];
        for (double[] row : X) {
            for (int j = 0; j < d; j++) for (int k = 0; k < d; k++)
                cov[j][k] += (row[j] - mean[j]) * (row[k] - mean[k]) / (n - 1);
        }
        // Add regularization
        for (int j = 0; j < d; j++) cov[j][j] += 1e-6;
        // Compute precision matrix (inverse of covariance)
        precisionMatrix = invert(cov);
        // Compute Mahalanobis distances for training data
        double[] dists = mahalanobisAll(X);
        Arrays.sort(dists);
        int tIdx = (int)((1 - contamination) * n); tIdx = Math.max(0, Math.min(tIdx, n-1));
        threshold = dists[tIdx];
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] dists = mahalanobisAll(X);
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) result[i] = dists[i] <= threshold ? 1.0 : -1.0;
        return result;
    }

    private double[] mahalanobisAll(double[][] X) {
        double[] dists = new double[X.length];
        for (int i = 0; i < X.length; i++) dists[i] = mahalanobis(X[i]);
        return dists;
    }

    private double mahalanobis(double[] x) {
        int d = x.length;
        double[] diff = new double[d]; for (int j = 0; j < d; j++) diff[j] = x[j] - mean[j];
        double dist = 0;
        for (int j = 0; j < d; j++) {
            double sum = 0; for (int k = 0; k < d; k++) sum += precisionMatrix[j][k] * diff[k];
            dist += diff[j] * sum;
        }
        return Math.sqrt(Math.max(0, dist));
    }

    private double[][] invert(double[][] A) {
        int n = A.length;
        // Gauss-Jordan
        double[][] aug = new double[n][2*n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
            aug[i][n + i] = 1;
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col+1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
            double[] tmp = aug[col]; aug[col] = aug[pivot]; aug[pivot] = tmp;
            double p = aug[col][col];
            if (Math.abs(p) < 1e-12) continue;
            for (int j = 0; j < 2*n; j++) aug[col][j] /= p;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = aug[row][col];
                for (int j = 0; j < 2*n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) inv[i][j] = aug[i][n+j];
        return inv;
    }

    @Override
    public Map<String, Object> getParams() {
        return new LinkedHashMap<>(Map.of("contamination", contamination));
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("contamination")) contamination = ((Number) params.get("contamination")).doubleValue();
    }
}

