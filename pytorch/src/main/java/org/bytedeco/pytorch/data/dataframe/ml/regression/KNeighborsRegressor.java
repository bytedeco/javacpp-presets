package org.bytedeco.pytorch.data.dataframe.ml.regression;
import org.bytedeco.pytorch.enumtype.*;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** K 近邻回归器 */
public class KNeighborsRegressor extends BaseRegressor {
    private int k; private String metric;
    private double[][] trainX; private double[] trainY;

    public KNeighborsRegressor() { this(5, "euclidean"); }
    public KNeighborsRegressor(int k) { this(k, "euclidean"); }
    public KNeighborsRegressor(int k, String metric) { this.k = k; this.metric = metric; }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) { trainX = X; trainY = y; fitted = true; return this; }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int[] nn = kNearest(X[i]);
            double s = 0; for (int n : nn) s += trainY[n];
            p[i] = s / k;
        }
        return p;
    }

    private int[] kNearest(double[] x) {
        double[] dists = new double[trainX.length];
        for (int i = 0; i < trainX.length; i++) dists[i] = dist(x, trainX[i]);
        Integer[] idx = new Integer[trainX.length]; for (int i = 0; i < trainX.length; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> dists[i]));
        int[] res = new int[k]; for (int i = 0; i < k; i++) res[i] = idx[i]; return res;
    }

    private double dist(double[] a, double[] b) {
        double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]); return Math.sqrt(s);
    }

    @Override
    public Map<String, Object> getParams() {
        return new LinkedHashMap<>(Map.of("n_neighbors", k, "metric", metric));
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_neighbors")) k = ((Number) params.get("n_neighbors")).intValue();
    }
}

