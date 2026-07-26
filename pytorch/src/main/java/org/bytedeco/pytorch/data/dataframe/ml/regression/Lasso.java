package org.bytedeco.pytorch.data.dataframe.ml.regression;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/**
 * Lasso 回归（L1 正则化，坐标下降法）
 */
public class Lasso extends BaseRegressor {
    private double alpha;
    private int maxIter;
    private double tol;
    private boolean fitIntercept;
    private double[] coef;
    private double intercept;

    public Lasso() { this(1.0, 1000, 1e-4, true); }
    public Lasso(double alpha, int maxIter, double tol, boolean fitIntercept) {
        this.alpha = alpha; this.maxIter = maxIter; this.tol = tol; this.fitIntercept = fitIntercept;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        coef = new double[d]; intercept = 0;
        // Center X and y if fitIntercept
        double[] xMean = new double[d]; double yMean = 0;
        if (fitIntercept) {
            for (double v : y) yMean += v; yMean /= n;
            for (int j = 0; j < d; j++) { for (double[] row : X) xMean[j] += row[j]; xMean[j] /= n; }
        }
        double[][] Xc = new double[n][d];
        double[] yc = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) Xc[i][j] = X[i][j] - xMean[j];
            yc[i] = y[i] - yMean;
        }
        // Coordinate descent
        double[] xSqNorm = new double[d];
        for (int j = 0; j < d; j++) for (double[] row : Xc) xSqNorm[j] += row[j]*row[j];

        for (int iter = 0; iter < maxIter; iter++) {
            double maxChange = 0;
            for (int j = 0; j < d; j++) {
                if (xSqNorm[j] < 1e-15) continue;
                double rho = 0;
                for (int i = 0; i < n; i++) {
                    double res = yc[i]; for (int k = 0; k < d; k++) if (k != j) res -= coef[k] * Xc[i][k];
                    rho += Xc[i][j] * res;
                }
                double newCoef = softThreshold(rho / xSqNorm[j], alpha * n / xSqNorm[j]);
                maxChange = Math.max(maxChange, Math.abs(newCoef - coef[j]));
                coef[j] = newCoef;
            }
            if (maxChange < tol) break;
        }
        if (fitIntercept) { intercept = yMean; for (int j = 0; j < d; j++) intercept -= coef[j] * xMean[j]; }
        fitted = true; return this;
    }

    private double softThreshold(double z, double lambda) {
        if (z > lambda) return z - lambda;
        if (z < -lambda) return z + lambda;
        return 0;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            p[i] = intercept; for (int j = 0; j < coef.length; j++) p[i] += coef[j] * X[i][j];
        }
        return p;
    }

    public double[] getCoef() { return coef; }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("alpha", alpha); p.put("max_iter", maxIter); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue();
    }
}

