package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/** 线性 SVM 分类器（hinge loss + L2 正则化） */
public class LinearSVC extends BaseClassifier {
    private double C; private int maxIter; private Long randomState;
    private SGDClassifier _delegate;

    public LinearSVC() { this(1.0, 1000, null); }
    public LinearSVC(double C, int maxIter, Long randomState) { this.C = C; this.maxIter = maxIter; this.randomState = randomState; }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        _delegate = new SGDClassifier("hinge", "l2", 1.0/C, maxIter, 1e-4, 0.01, randomState);
        _delegate.fit(X, y); fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) { return _delegate.predict(X); }

    @Override
    public Map<String, Object> getParams() {
        return new LinkedHashMap<>(Map.of("C", C, "max_iter", maxIter, "random_state", randomState));
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("C")) C = ((Number) params.get("C")).doubleValue();
    }
}

