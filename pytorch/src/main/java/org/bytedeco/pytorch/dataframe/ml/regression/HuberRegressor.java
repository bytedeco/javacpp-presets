package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** Huber 回归（鲁棒回归，对异常值不敏感） */
public class HuberRegressor extends BaseRegressor {
    private double epsilon; // threshold between L1 and L2
    private double alpha;   // L2 reg
    private int maxIter; private double tol;
    private double[] coef; private double intercept;

    public HuberRegressor() { this(1.35, 1e-4, 100, 1e-5); }
    public HuberRegressor(double epsilon, double alpha, int maxIter, double tol) {
        this.epsilon = epsilon; this.alpha = alpha; this.maxIter = maxIter; this.tol = tol;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        coef = new double[d]; intercept = 0;
        double lr = 0.01;
        for (int iter = 0; iter < maxIter; iter++) {
            double[] gradW = new double[d]; double gradB = 0;
            for (int i = 0; i < n; i++) {
                double pred = intercept; for (int j = 0; j < d; j++) pred += coef[j] * X[i][j];
                double err = pred - y[i];
                double grad = Math.abs(err) <= epsilon ? err : epsilon * Math.signum(err);
                for (int j = 0; j < d; j++) gradW[j] += grad * X[i][j] + alpha * coef[j];
                gradB += grad;
            }
            double maxChange = 0;
            for (int j = 0; j < d; j++) { double d2 = lr * gradW[j] / n; coef[j] -= d2; maxChange = Math.max(maxChange, Math.abs(d2)); }
            intercept -= lr * gradB / n;
            if (maxChange < tol) break;
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) { p[i] = intercept; for (int j = 0; j < coef.length; j++) p[i] += coef[j]*X[i][j]; }
        return p;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("epsilon", epsilon); p.put("alpha", alpha); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("epsilon")) epsilon = ((Number) params.get("epsilon")).doubleValue();
    }
}

