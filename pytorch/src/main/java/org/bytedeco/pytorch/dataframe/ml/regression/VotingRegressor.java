package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** 投票回归器（均值集成多个回归器） */
public class VotingRegressor extends BaseRegressor {
    private List<String> names = new ArrayList<>();
    private List<BaseRegressor> estimators = new ArrayList<>();

    public VotingRegressor addEstimator(String name, BaseRegressor r) { names.add(name); estimators.add(r); return this; }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        for (BaseRegressor e : estimators) e.fit(X, y); fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] sum = new double[X.length];
        for (BaseRegressor e : estimators) { double[] p = e.predict(X); for (int i = 0; i < X.length; i++) sum[i] += p[i]; }
        for (int i = 0; i < X.length; i++) sum[i] /= estimators.size();
        return sum;
    }

    @Override
    public Map<String, Object> getParams() { return new LinkedHashMap<>(); }
    @Override public void setParams(Map<String, Object> params) {}
}

