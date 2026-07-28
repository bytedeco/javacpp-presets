package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** KernelRidge 回归（核岭回归） */
public class KernelRidge extends BaseRegressor {
    private double alpha; private String kernel; private double gamma;
    private double[][] trainX; private double[] dualCoef;

    public KernelRidge() { this(1.0, "rbf", -1); }
    public KernelRidge(double alpha, String kernel, double gamma) { this.alpha = alpha; this.kernel = kernel; this.gamma = gamma; }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length;
        double g = gamma < 0 ? 1.0 / X[0].length : gamma;
        double[][] K = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) K[i][j] = kernelFunc(X[i], X[j], g);
        for (int i = 0; i < n; i++) K[i][i] += alpha;
        dualCoef = LinearRegression.gaussianElimination(K, y);
        trainX = X; fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        double g = gamma < 0 ? 1.0 / trainX[0].length : gamma;
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < trainX.length; j++) p[i] += dualCoef[j] * kernelFunc(X[i], trainX[j], g);
        }
        return p;
    }

    private double kernelFunc(double[] a, double[] b, double g) {
        if ("linear".equals(kernel)) { double s = 0; for (int i = 0; i < a.length; i++) s += a[i]*b[i]; return s; }
        double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]);
        return Math.exp(-g * s);
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("alpha", alpha); p.put("kernel", kernel); return p;
    }
    @Override public void setParams(Map<String, Object> params) { if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue(); }
}

