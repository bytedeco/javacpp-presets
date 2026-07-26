package org.bytedeco.pytorch.data.dataframe.ml.regression;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** SVR：支持向量回归（使用 SGD 近似 epsilon-insensitive loss） */
public class SVR extends BaseRegressor {
    private double C; private String kernel; private double epsilon;
    private int maxIter; private Long randomState;
    private double[] coef; private double intercept;

    public SVR() { this(1.0, "rbf", 0.1, 1000, null); }
    public SVR(double C, String kernel, double epsilon, int maxIter, Long rs) {
        this.C = C; this.kernel = kernel; this.epsilon = epsilon; this.maxIter = maxIter; this.randomState = rs;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        coef = new double[d]; intercept = 0;
        Random rng = randomState == null ? new Random() : new Random(randomState);
        double lr = 0.01;
        for (int iter = 0; iter < maxIter; iter++) {
            for (int i = 0; i < n; i++) {
                int ii = rng.nextInt(n);
                double pred = intercept; for (int j = 0; j < d; j++) pred += coef[j] * X[ii][j];
                double err = pred - y[ii];
                // epsilon-insensitive loss gradient
                if (Math.abs(err) > epsilon) {
                    double sign = err > 0 ? 1 : -1;
                    for (int j = 0; j < d; j++) coef[j] -= lr * (sign * X[ii][j] + coef[j] / C);
                    intercept -= lr * sign;
                } else {
                    for (int j = 0; j < d; j++) coef[j] -= lr * coef[j] / C;
                }
            }
        }
        fitted = true; return this;
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
        p.put("C", C); p.put("kernel", kernel); p.put("epsilon", epsilon); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("C")) C = ((Number) params.get("C")).doubleValue();
    }
}

