package org.bytedeco.pytorch.dataframe.ml.model_selection;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.dataframe.feature.base.BaseEstimator;
import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import org.bytedeco.pytorch.dataframe.ml.metrics.ClassificationMetrics;
import org.bytedeco.pytorch.dataframe.ml.metrics.RegressionMetrics;

import java.util.*;

/**
 * 网格搜索交叉验证（对应 sklearn GridSearchCV）
 *
 * <pre>
 * Map<String,Object[]> grid = new LinkedHashMap<>();
 * grid.put("C",       new Object[]{0.1, 1.0, 10.0});
 * grid.put("max_iter",new Object[]{100, 500});
 *
 * GridSearchCV gs = new GridSearchCV(new LogisticRegression(), grid, 5);
 * gs.fit(X, y);
 * System.out.println("Best params: " + gs.getBestParams());
 * System.out.println("Best score:  " + gs.getBestScore());
 * </pre>
 */
public class GridSearchCV {
    private final BaseEstimator estimator;
    private final Map<String, Object[]> paramGrid;
    private final int cv;
    private final String scoring;

    private Map<String, Object> bestParams;
    private double bestScore = Double.NEGATIVE_INFINITY;
    private BaseEstimator bestEstimator;
    private List<Map<String, Object>> cvResults = new ArrayList<>();

    public GridSearchCV(BaseEstimator estimator, Map<String, Object[]> paramGrid, int cv) {
        this(estimator, paramGrid, cv, "accuracy");
    }

    public GridSearchCV(BaseEstimator estimator, Map<String, Object[]> paramGrid, int cv, String scoring) {
        this.estimator = estimator;
        this.paramGrid = paramGrid;
        this.cv = cv;
        this.scoring = scoring;
    }

    public GridSearchCV fit(double[][] X, double[] y) {
        List<Map<String, Object>> allCombinations = cartesianProduct(paramGrid);
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);

        for (Map<String, Object> params : allCombinations) {
            double totalScore = 0;
            for (KFold.Split s : splits) {
                BaseEstimator candidate = cloneAndSet(params);
                if (candidate instanceof BaseClassifier clf) {
                    clf.fit(s.trainX(X), s.trainY(y));
                    double[] preds = clf.predict(s.testX(X));
                    totalScore += score(preds, s.testY(y));
                } else if (candidate instanceof BaseRegressor reg) {
                    reg.fit(s.trainX(X), s.trainY(y));
                    double[] preds = reg.predict(s.testX(X));
                    totalScore += scoreReg(preds, s.testY(y));
                }
            }
            double meanScore = totalScore / cv;
            Map<String, Object> result = new LinkedHashMap<>(params);
            result.put("mean_test_score", meanScore);
            cvResults.add(result);

            if (meanScore > bestScore) {
                bestScore = meanScore;
                bestParams = new LinkedHashMap<>(params);
                bestEstimator = cloneAndSet(params);
            }
        }
        // Final fit on all data with best params
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

    private double score(double[] preds, double[] y) {
        return switch (scoring.toLowerCase()) {
            case "accuracy"  -> ClassificationMetrics.accuracyScore(y, preds);
            case "f1"        -> ClassificationMetrics.f1Score(y, preds);
            case "precision" -> ClassificationMetrics.precisionScore(y, preds);
            case "recall"    -> ClassificationMetrics.recallScore(y, preds);
            default          -> ClassificationMetrics.accuracyScore(y, preds);
        };
    }

    private double scoreReg(double[] preds, double[] y) {
        return switch (scoring.toLowerCase()) {
            case "r2"  -> RegressionMetrics.r2Score(y, preds);
            case "mse" -> -RegressionMetrics.meanSquaredError(y, preds);
            default    -> RegressionMetrics.r2Score(y, preds);
        };
    }

    /** Generate Cartesian product of param grid */
    private List<Map<String, Object>> cartesianProduct(Map<String, Object[]> grid) {
        List<Map<String, Object>> result = new ArrayList<>();
        result.add(new LinkedHashMap<>());
        for (Map.Entry<String, Object[]> entry : grid.entrySet()) {
            List<Map<String, Object>> newResult = new ArrayList<>();
            for (Map<String, Object> existing : result) {
                for (Object val : entry.getValue()) {
                    Map<String, Object> combo = new LinkedHashMap<>(existing);
                    combo.put(entry.getKey(), val);
                    newResult.add(combo);
                }
            }
            result = newResult;
        }
        return result;
    }

    public Map<String, Object> getBestParams()      { return bestParams; }
    public double              getBestScore()        { return bestScore; }
    public BaseEstimator       getBestEstimator()    { return bestEstimator; }
    public List<Map<String, Object>> getCvResults()  { return cvResults; }
}

