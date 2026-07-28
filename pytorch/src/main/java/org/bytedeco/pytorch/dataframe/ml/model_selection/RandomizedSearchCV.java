package org.bytedeco.pytorch.dataframe.ml.model_selection;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.dataframe.feature.base.BaseEstimator;
import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import org.bytedeco.pytorch.dataframe.ml.metrics.ClassificationMetrics;
import org.bytedeco.pytorch.dataframe.ml.metrics.RegressionMetrics;

import java.util.*;

/**
 * 随机搜索交叉验证（对应 sklearn RandomizedSearchCV）
 * 从参数分布中随机采样 nIter 次，找最佳超参组合
 */
public class RandomizedSearchCV {
    private final BaseEstimator estimator;
    private final Map<String, Object[]> paramDistributions;
    private final int nIter;
    private final int cv;
    private final String scoring;
    private final Long randomState;

    private Map<String, Object> bestParams;
    private double bestScore = Double.NEGATIVE_INFINITY;
    private BaseEstimator bestEstimator;
    private List<Map<String, Object>> cvResults = new ArrayList<>();

    public RandomizedSearchCV(BaseEstimator estimator, Map<String, Object[]> paramDistributions,
                               int nIter, int cv, Long randomState) {
        this(estimator, paramDistributions, nIter, cv, "accuracy", randomState);
    }

    public RandomizedSearchCV(BaseEstimator estimator, Map<String, Object[]> paramDistributions,
                               int nIter, int cv, String scoring, Long randomState) {
        this.estimator = estimator;
        this.paramDistributions = paramDistributions;
        this.nIter = nIter;
        this.cv = cv;
        this.scoring = scoring;
        this.randomState = randomState;
    }

    public RandomizedSearchCV fit(double[][] X, double[] y) {
        Random rng = randomState == null ? new Random() : new Random(randomState);
        KFold kf = new KFold(cv, true, randomState);
        List<KFold.Split> splits = kf.split(X, y);

        for (int iter = 0; iter < nIter; iter++) {
            // Random sample from each param distribution
            Map<String, Object> params = new LinkedHashMap<>();
            for (Map.Entry<String, Object[]> e : paramDistributions.entrySet()) {
                params.put(e.getKey(), e.getValue()[rng.nextInt(e.getValue().length)]);
            }

            double totalScore = 0;
            for (KFold.Split s : splits) {
                BaseEstimator candidate = cloneAndSet(params);
                if (candidate instanceof BaseClassifier clf) {
                    clf.fit(s.trainX(X), s.trainY(y));
                    totalScore += scoreClf(clf.predict(s.testX(X)), s.testY(y));
                } else if (candidate instanceof BaseRegressor reg) {
                    reg.fit(s.trainX(X), s.trainY(y));
                    totalScore += scoreReg(reg.predict(s.testX(X)), s.testY(y));
                }
            }
            double meanScore = totalScore / cv;
            Map<String, Object> r = new LinkedHashMap<>(params);
            r.put("mean_test_score", meanScore);
            cvResults.add(r);

            if (meanScore > bestScore) {
                bestScore = meanScore;
                bestParams = new LinkedHashMap<>(params);
                bestEstimator = cloneAndSet(params);
            }
        }
        if (bestEstimator instanceof BaseClassifier clf) clf.fit(X, y);
        else if (bestEstimator instanceof BaseRegressor reg) reg.fit(X, y);
        return this;
    }

    private BaseEstimator cloneAndSet(Map<String, Object> params) {
        try {
            BaseEstimator copy = estimator.getClass().getDeclaredConstructor().newInstance();
            copy.setParams(params);
            return copy;
        } catch (Exception e) {
            estimator.setParams(params);
            return estimator;
        }
    }

    private double scoreClf(double[] preds, double[] y) {
        return switch (scoring.toLowerCase()) {
            case "f1"        -> ClassificationMetrics.f1Score(y, preds);
            case "precision" -> ClassificationMetrics.precisionScore(y, preds);
            case "recall"    -> ClassificationMetrics.recallScore(y, preds);
            default          -> ClassificationMetrics.accuracyScore(y, preds);
        };
    }

    private double scoreReg(double[] preds, double[] y) {
        return switch (scoring.toLowerCase()) {
            case "mse" -> -RegressionMetrics.meanSquaredError(y, preds);
            default    -> RegressionMetrics.r2Score(y, preds);
        };
    }

    public Map<String, Object>       getBestParams()   { return bestParams; }
    public double                    getBestScore()    { return bestScore; }
    public BaseEstimator             getBestEstimator(){ return bestEstimator; }
    public List<Map<String, Object>> getCvResults()    { return cvResults; }
}

