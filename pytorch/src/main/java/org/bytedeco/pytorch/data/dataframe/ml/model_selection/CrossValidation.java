package org.bytedeco.pytorch.data.dataframe.ml.model_selection;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import org.bytedeco.pytorch.data.dataframe.ml.metrics.ClassificationMetrics;
import org.bytedeco.pytorch.data.dataframe.ml.metrics.RegressionMetrics;

import java.util.*;
import java.util.function.Function;

/**
 * 交叉验证工具（对应 sklearn cross_val_score / cross_val_predict / cross_validate）
 *
 * <pre>
 * double[] scores = CrossValidation.crossValScore(clf, X, y, 5);
 * </pre>
 */
public class CrossValidation {

    // ============ cross_val_score ============

    /** 分类器交叉验证评分（默认 accuracy） */
    public static double[] crossValScore(BaseClassifier clf, double[][] X, double[] y, int cv) {
        return crossValScore(clf, X, y, cv, "accuracy");
    }

    public static double[] crossValScore(BaseClassifier clf, double[][] X, double[] y, int cv, String scoring) {
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);
        double[] scores = new double[splits.size()];
        for (int i = 0; i < splits.size(); i++) {
            KFold.Split s = splits.get(i);
            clf.fit(s.trainX(X), s.trainY(y));
            double[] preds = clf.predict(s.testX(X));
            double[] trueY = s.testY(y);
            scores[i] = scoreClassification(preds, trueY, scoring);
        }
        return scores;
    }

    /** 回归器交叉验证评分（默认 r2） */
    public static double[] crossValScore(BaseRegressor reg, double[][] X, double[] y, int cv) {
        return crossValScore(reg, X, y, cv, "r2");
    }

    public static double[] crossValScore(BaseRegressor reg, double[][] X, double[] y, int cv, String scoring) {
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);
        double[] scores = new double[splits.size()];
        for (int i = 0; i < splits.size(); i++) {
            KFold.Split s = splits.get(i);
            reg.fit(s.trainX(X), s.trainY(y));
            double[] preds = reg.predict(s.testX(X));
            scores[i] = scoreRegression(preds, s.testY(y), scoring);
        }
        return scores;
    }

    // ============ cross_val_predict ============

    public static double[] crossValPredict(BaseClassifier clf, double[][] X, double[] y, int cv) {
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);
        double[] allPreds = new double[X.length];
        for (KFold.Split s : splits) {
            clf.fit(s.trainX(X), s.trainY(y));
            double[] preds = clf.predict(s.testX(X));
            for (int i = 0; i < s.testIndices.length; i++) allPreds[s.testIndices[i]] = preds[i];
        }
        return allPreds;
    }

    public static double[] crossValPredict(BaseRegressor reg, double[][] X, double[] y, int cv) {
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);
        double[] allPreds = new double[X.length];
        for (KFold.Split s : splits) {
            reg.fit(s.trainX(X), s.trainY(y));
            double[] preds = reg.predict(s.testX(X));
            for (int i = 0; i < s.testIndices.length; i++) allPreds[s.testIndices[i]] = preds[i];
        }
        return allPreds;
    }

    // ============ cross_validate (returns map) ============

    public static Map<String, double[]> crossValidate(BaseClassifier clf, double[][] X, double[] y,
                                                       int cv, String[] scoring) {
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);
        Map<String, List<Double>> results = new LinkedHashMap<>();
        for (String s : scoring) results.put("test_" + s, new ArrayList<>());
        results.put("fit_time", new ArrayList<>());
        results.put("score_time", new ArrayList<>());
        for (KFold.Split s : splits) {
            long t0 = System.currentTimeMillis();
            clf.fit(s.trainX(X), s.trainY(y));
            results.get("fit_time").add((System.currentTimeMillis() - t0) / 1000.0);
            t0 = System.currentTimeMillis();
            double[] preds = clf.predict(s.testX(X));
            results.get("score_time").add((System.currentTimeMillis() - t0) / 1000.0);
            for (String sc : scoring)
                results.get("test_" + sc).add(scoreClassification(preds, s.testY(y), sc));
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<Double>> e : results.entrySet())
            out.put(e.getKey(), e.getValue().stream().mapToDouble(Double::doubleValue).toArray());
        return out;
    }

    // ============ learning_curve ============

    public static double[][] learningCurve(BaseClassifier clf, double[][] X, double[] y,
                                             int cv, int[] trainSizes) {
        // Returns [trainSizes.length][cv] train scores and test scores
        double[][] trainScores = new double[trainSizes.length][cv];
        double[][] testScores  = new double[trainSizes.length][cv];
        KFold kf = new KFold(cv, true, null);
        List<KFold.Split> splits = kf.split(X, y);
        for (int k = 0; k < splits.size(); k++) {
            KFold.Split s = splits.get(k);
            double[][] testX = s.testX(X); double[] testY = s.testY(y);
            for (int si = 0; si < trainSizes.length; si++) {
                int sz = Math.min(trainSizes[si], s.trainIndices.length);
                double[][] trX = new double[sz][X[0].length]; double[] trY = new double[sz];
                for (int i = 0; i < sz; i++) { trX[i] = X[s.trainIndices[i]]; trY[i] = y[s.trainIndices[i]]; }
                clf.fit(trX, trY);
                trainScores[si][k] = clf.score(trX, trY);
                testScores[si][k]  = clf.score(testX, testY);
            }
        }
        // Return [trainScores_mean, testScores_mean] per trainSize
        double[][] result = new double[2][trainSizes.length];
        for (int si = 0; si < trainSizes.length; si++) {
            double ts = 0, tes = 0;
            for (int k = 0; k < cv; k++) { ts += trainScores[si][k]; tes += testScores[si][k]; }
            result[0][si] = ts / cv; result[1][si] = tes / cv;
        }
        return result; // result[0] = train scores mean, result[1] = test scores mean
    }

    // ============ helpers ============

    private static double scoreClassification(double[] preds, double[] y, String metric) {
        return switch (metric.toLowerCase()) {
            case "accuracy"  -> ClassificationMetrics.accuracyScore(y, preds);
            case "precision" -> ClassificationMetrics.precisionScore(y, preds);
            case "recall"    -> ClassificationMetrics.recallScore(y, preds);
            case "f1"        -> ClassificationMetrics.f1Score(y, preds);
            default          -> ClassificationMetrics.accuracyScore(y, preds);
        };
    }

    private static double scoreRegression(double[] preds, double[] y, String metric) {
        return switch (metric.toLowerCase()) {
            case "r2"  -> RegressionMetrics.r2Score(y, preds);
            case "mse" -> -RegressionMetrics.meanSquaredError(y, preds);
            case "mae" -> -RegressionMetrics.meanAbsoluteError(y, preds);
            default    -> RegressionMetrics.r2Score(y, preds);
        };
    }
}

