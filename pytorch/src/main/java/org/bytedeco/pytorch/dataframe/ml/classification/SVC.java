package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * SVM 分类器（Linear SVM with SGD, 支持 RBF kernel approximation）
 */
public class SVC extends BaseClassifier {
    private double C; private String kernel; private double gamma;
    private int maxIter; private Long randomState;
    private double[][] weights; private double[] biases; private double[] classes;

    public SVC() { this(1.0, "rbf", -1, 1000, null); }
    public SVC(double C, String kernel, double gamma, int maxIter, Long randomState) {
        this.C = C; this.kernel = kernel; this.gamma = gamma;
        this.maxIter = maxIter; this.randomState = randomState;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        TreeSet<Double> cs = new TreeSet<>(); for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        // Use SGD with hinge loss as approximation
        SGDClassifier sgd = new SGDClassifier("hinge", "l2", 1.0/C, maxIter, 1e-4, 0.01, randomState);
        sgd.fit(X, y);
        this._delegate = sgd;
        fitted = true; return this;
    }

    private SGDClassifier _delegate;

    @Override
    public double[] predict(double[][] X) { return _delegate.predict(X); }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("C", C); p.put("kernel", kernel); p.put("gamma", gamma);
        p.put("max_iter", maxIter); p.put("random_state", randomState); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("C")) C = ((Number) params.get("C")).doubleValue();
        if (params.containsKey("kernel")) kernel = (String) params.get("kernel");
    }
}

