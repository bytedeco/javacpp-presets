package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** 直方图梯度提升回归器 */
public class HistGradientBoostingRegressor extends BaseRegressor {
    private int maxIter; private double learningRate; private int maxDepth; private int maxBins; private Long randomState;
    private GradientBoostingRegressor delegate;

    public HistGradientBoostingRegressor() { this(100, 0.1, 3, 255, null); }
    public HistGradientBoostingRegressor(int maxIter, double lr, int depth, int bins, Long rs) {
        this.maxIter = maxIter; this.learningRate = lr; this.maxDepth = depth; this.maxBins = bins; this.randomState = rs;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        double[][] binned = binFeatures(X);
        delegate = new GradientBoostingRegressor(maxIter, learningRate, maxDepth, 1.0, randomState);
        delegate.fit(binned, y); fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) { return delegate.predict(binFeatures(X)); }

    private double[][] binFeatures(double[][] X) {
        int n = X.length, d = X[0].length;
        double[][] binned = new double[n][d];
        for (int j = 0; j < d; j++) {
            double min = X[0][j], max = X[0][j];
            for (double[] row : X) { min = Math.min(min, row[j]); max = Math.max(max, row[j]); }
            for (int i = 0; i < n; i++)
                binned[i][j] = max == min ? 0 : Math.floor((X[i][j] - min) / (max - min) * (maxBins - 1));
        }
        return binned;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_iter", maxIter); p.put("learning_rate", learningRate); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("max_iter")) maxIter = ((Number) params.get("max_iter")).intValue();
    }
}

