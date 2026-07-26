package org.bytedeco.pytorch.data.dataframe.ml.classification;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 直方图梯度提升分类器（bin-based GBDT，高效处理大数据集）
 */
public class HistGradientBoostingClassifier extends BaseClassifier {
    private int maxIter;
    private double learningRate;
    private int maxDepth;
    private int maxBins;
    private Long randomState;
    private double[] classes;

    // Delegate to GradientBoostingClassifier for now
    private GradientBoostingClassifier delegate;

    public HistGradientBoostingClassifier() { this(100, 0.1, 3, 255, null); }
    public HistGradientBoostingClassifier(int maxIter, double lr, int maxDepth, int maxBins, Long rs) {
        this.maxIter = maxIter; this.learningRate = lr; this.maxDepth = maxDepth;
        this.maxBins = maxBins; this.randomState = rs;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        // Bin the features for speed
        double[][] binned = binFeatures(X, maxBins);
        delegate = new GradientBoostingClassifier(maxIter, learningRate, maxDepth, 1.0, randomState);
        delegate.fit(binned, y);
        TreeSet<Double> cs = new TreeSet<>(); for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) { return delegate.predict(binFeatures(X, maxBins)); }

    @Override
    public double[][] predictProba(double[][] X) { return delegate.predictProba(binFeatures(X, maxBins)); }

    private double[][] binFeatures(double[][] X, int bins) {
        int n = X.length, d = X[0].length;
        double[][] binned = new double[n][d];
        for (int j = 0; j < d; j++) {
            double min = X[0][j], max = X[0][j];
            for (double[] row : X) { min = Math.min(min, row[j]); max = Math.max(max, row[j]); }
            for (int i = 0; i < n; i++) {
                binned[i][j] = max == min ? 0 : Math.floor((X[i][j] - min) / (max - min) * (bins - 1));
            }
        }
        return binned;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_iter", maxIter); p.put("learning_rate", learningRate);
        p.put("max_depth", maxDepth); p.put("max_bins", maxBins); p.put("random_state", randomState); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("max_iter")) maxIter = ((Number) params.get("max_iter")).intValue();
        if (params.containsKey("learning_rate")) learningRate = ((Number) params.get("learning_rate")).doubleValue();
    }
}

