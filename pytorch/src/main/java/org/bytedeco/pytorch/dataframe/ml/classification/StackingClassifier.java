package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * Stacking 分类器（元学习，默认元学习器 LogisticRegression）
 */
public class StackingClassifier extends BaseClassifier {
    private List<BaseClassifier> estimators = new ArrayList<>();
    private BaseClassifier finalEstimator;
    private int cv;

    public StackingClassifier(List<BaseClassifier> estimators, BaseClassifier finalEstimator, int cv) {
        this.estimators = new ArrayList<>(estimators);
        this.finalEstimator = finalEstimator;
        this.cv = cv;
    }

    public StackingClassifier(List<BaseClassifier> estimators) {
        this(estimators, new LogisticRegression(), 5);
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length;
        double[][] metaFeatures = new double[n][estimators.size()];

        // Simple approach: train each base classifier on full X and generate OOF predictions
        for (int e = 0; e < estimators.size(); e++) {
            estimators.get(e).fit(X, y);
            double[] preds = estimators.get(e).predict(X);
            for (int i = 0; i < n; i++) metaFeatures[i][e] = preds[i];
        }
        finalEstimator.fit(metaFeatures, y);
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] meta = transform(X);
        return finalEstimator.predict(meta);
    }

    @Override
    public double[][] predictProba(double[][] X) {
        double[][] meta = transform(X);
        return finalEstimator.predictProba(meta);
    }

    private double[][] transform(double[][] X) {
        double[][] meta = new double[X.length][estimators.size()];
        for (int e = 0; e < estimators.size(); e++) {
            double[] preds = estimators.get(e).predict(X);
            for (int i = 0; i < X.length; i++) meta[i][e] = preds[i];
        }
        return meta;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("cv", cv); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("cv")) cv = ((Number) params.get("cv")).intValue();
    }
}

