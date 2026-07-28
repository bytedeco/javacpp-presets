package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/**
 * ElasticNet 回归（L1 + L2 混合正则化，坐标下降法）
 */
public class ElasticNet extends BaseRegressor {
    private double alpha;
    private double l1Ratio;   // mix: l1_ratio=1 → Lasso, l1_ratio=0 → Ridge
    private int maxIter;
    private double tol;
    private boolean fitIntercept;
    private double[] coef;
    private double intercept;

    public ElasticNet() { this(1.0, 0.5, 1000, 1e-4, true); }
    public ElasticNet(double alpha, double l1Ratio, int maxIter, double tol, boolean fitIntercept) {
        this.alpha = alpha; this.l1Ratio = l1Ratio; this.maxIter = maxIter;
        this.tol = tol; this.fitIntercept = fitIntercept;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        coef = new double[d]; intercept = 0;

        double[] xMean = new double[d]; double yMean = 0;
        if (fitIntercept) {
            for (double v : y) yMean += v; yMean /= n;
            for (int j = 0; j < d; j++) { for (double[] row : X) xMean[j] += row[j]; xMean[j] /= n; }
        }
        double[][] Xc = new double[n][d]; double[] yc = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) Xc[i][j] = X[i][j] - xMean[j];
            yc[i] = y[i] - yMean;
        }

        double[] xSqNorm = new double[d];
        for (int j = 0; j < d; j++) for (double[] row : Xc) xSqNorm[j] += row[j] * row[j];

        double lambda1 = alpha * l1Ratio;
        double lambda2 = alpha * (1 - l1Ratio);

        for (int iter = 0; iter < maxIter; iter++) {
            double maxChange = 0;
            for (int j = 0; j < d; j++) {
                double denom = xSqNorm[j] + lambda2 * n;
                if (denom < 1e-15) continue;
                double rho = 0;
                for (int i = 0; i < n; i++) {
                    double res = yc[i]; for (int k = 0; k < d; k++) if (k != j) res -= coef[k] * Xc[i][k];
                    rho += Xc[i][j] * res;
                }
                double newCoef = softThreshold(rho / denom, lambda1 * n / denom);
                maxChange = Math.max(maxChange, Math.abs(newCoef - coef[j]));
                coef[j] = newCoef;
            }
            if (maxChange < tol) break;
        }
        if (fitIntercept) { intercept = yMean; for (int j = 0; j < d; j++) intercept -= coef[j] * xMean[j]; }
        fitted = true; return this;
    }

    private double softThreshold(double z, double lam) {
        if (z > lam) return z - lam; if (z < -lam) return z + lam; return 0;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            p[i] = intercept; for (int j = 0; j < coef.length; j++) p[i] += coef[j] * X[i][j];
        }
        return p;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("alpha", alpha); p.put("l1_ratio", l1Ratio); p.put("max_iter", maxIter); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue();
        if (params.containsKey("l1_ratio")) l1Ratio = ((Number) params.get("l1_ratio")).doubleValue();
    }
}

