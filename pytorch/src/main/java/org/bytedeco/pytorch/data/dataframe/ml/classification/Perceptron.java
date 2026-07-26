package org.bytedeco.pytorch.data.dataframe.ml.classification;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 感知机分类器（Perceptron）
 */
public class Perceptron extends BaseClassifier {
    private int maxIter; private double tol; private Long randomState;
    private double[][] weights; private double[] biases; private double[] classes;

    public Perceptron() { this(1000, 1e-3, null); }
    public Perceptron(int maxIter, double tol, Long randomState) {
        this.maxIter = maxIter; this.tol = tol; this.randomState = randomState;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        // Delegate to SGDClassifier with perceptron loss
        SGDClassifier sgd = new SGDClassifier("perceptron", "none", 0, maxIter, tol, 1.0, randomState);
        sgd.fit(X, y);
        // copy state
        TreeSet<Double> cs = new TreeSet<>();
        for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        this.fitted = true;
        this._sgd = sgd;
        return this;
    }

    private SGDClassifier _sgd;

    @Override
    public double[] predict(double[][] X) { return _sgd.predict(X); }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_iter", maxIter); p.put("tol", tol); p.put("random_state", randomState); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("max_iter")) maxIter = ((Number) params.get("max_iter")).intValue();
    }
}

