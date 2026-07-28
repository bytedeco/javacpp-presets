package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/**
 * Ridge 回归（L2 正则化线性回归）
 */
public class Ridge extends BaseRegressor {
    private double alpha;
    private boolean fitIntercept;
    private double[] coef;
    private double intercept;

    public Ridge() { this(1.0, true); }
    public Ridge(double alpha, boolean fitIntercept) { this.alpha = alpha; this.fitIntercept = fitIntercept; }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        int cols = fitIntercept ? d + 1 : d;
        double[][] A = new double[n][cols];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) A[i][j] = X[i][j];
            if (fitIntercept) A[i][d] = 1.0;
        }
        double[][] ATA = new double[cols][cols];
        double[]   ATy = new double[cols];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < cols; j++) {
                ATy[j] += A[i][j] * y[i];
                for (int k = 0; k < cols; k++) ATA[j][k] += A[i][j] * A[i][k];
            }
        }
        // Add L2 regularization (don't regularize bias)
        for (int j = 0; j < d; j++) ATA[j][j] += alpha;
        double[] w = LinearRegression.gaussianElimination(ATA, ATy);
        coef = Arrays.copyOfRange(w, 0, d);
        intercept = fitIntercept ? w[d] : 0.0;
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double s = intercept;
            for (int j = 0; j < coef.length; j++) s += coef[j] * X[i][j];
            p[i] = s;
        }
        return p;
    }

    public double[] getCoef() { return coef; }
    public double getIntercept() { return intercept; }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("alpha", alpha); p.put("fit_intercept", fitIntercept); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue();
    }
}

