package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** 线性 SVR */
public class LinearSVR extends BaseRegressor {
    private double C; private double epsilon; private int maxIter; private Long randomState;
    private SVR _delegate;
    public LinearSVR() { this(1.0, 0.1, 1000, null); }
    public LinearSVR(double C, double epsilon, int maxIter, Long rs) { this.C=C; this.epsilon=epsilon; this.maxIter=maxIter; this.randomState=rs; }

    @Override public BaseRegressor fit(double[][] X, double[] y) {
        _delegate = new SVR(C, "linear", epsilon, maxIter, randomState); _delegate.fit(X, y); fitted=true; return this;
    }
    @Override public double[] predict(double[][] X) { return _delegate.predict(X); }
    @Override public Map<String, Object> getParams() { return new LinkedHashMap<>(Map.of("C", C, "epsilon", epsilon)); }
    @Override public void setParams(Map<String, Object> params) { if (params.containsKey("C")) C=((Number)params.get("C")).doubleValue(); }
}

