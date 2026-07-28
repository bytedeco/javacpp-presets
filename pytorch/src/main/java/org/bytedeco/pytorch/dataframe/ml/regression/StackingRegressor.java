package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** Stacking 回归器 */
public class StackingRegressor extends BaseRegressor {
    private List<BaseRegressor> estimators; private BaseRegressor finalEstimator;

    public StackingRegressor(List<BaseRegressor> estimators) { this(estimators, new Ridge()); }
    public StackingRegressor(List<BaseRegressor> estimators, BaseRegressor finalEstimator) {
        this.estimators = new ArrayList<>(estimators); this.finalEstimator = finalEstimator;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length;
        double[][] meta = new double[n][estimators.size()];
        for (int e = 0; e < estimators.size(); e++) {
            estimators.get(e).fit(X, y);
            double[] p = estimators.get(e).predict(X);
            for (int i = 0; i < n; i++) meta[i][e] = p[i];
        }
        finalEstimator.fit(meta, y); fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] meta = new double[X.length][estimators.size()];
        for (int e = 0; e < estimators.size(); e++) {
            double[] p = estimators.get(e).predict(X); for (int i = 0; i < X.length; i++) meta[i][e] = p[i];
        }
        return finalEstimator.predict(meta);
    }

    @Override
    public Map<String, Object> getParams() { return new LinkedHashMap<>(); }
    @Override public void setParams(Map<String, Object> params) {}
}

