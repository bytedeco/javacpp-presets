package org.bytedeco.pytorch.data.dataframe.feature.selection;

import java.util.*;

/**
 * 特征评分函数工具（对应 sklearn f_classif, f_regression, chi2, mutual_info_classif 等）
 * 全部返回 double[]（每个特征的得分）
 */
public class ScoreFunctions {

    /**
     * F 统计量（用于回归）：特征与目标的线性相关性
     */
    public static double[] fRegression(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        double yMean = 0; for (double v : y) yMean += v; yMean /= n;
        double[] scores = new double[d];
        for (int j = 0; j < d; j++) {
            double xMean = 0; for (double[] row : X) xMean += row[j]; xMean /= n;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++) {
                sxy += (X[i][j] - xMean) * (y[i] - yMean);
                sxx += (X[i][j] - xMean) * (X[i][j] - xMean);
                syy += (y[i] - yMean) * (y[i] - yMean);
            }
            double r2 = sxx * syy == 0 ? 0 : (sxy * sxy) / (sxx * syy);
            scores[j] = n == 2 ? 0 : r2 * (n - 2) / (1 - r2 + 1e-15);
        }
        return scores;
    }

    /**
     * F 统计量（用于分类）：类内方差 vs 类间方差
     */
    public static double[] fClassif(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        Map<Double, List<Integer>> byClass = new HashMap<>();
        for (int i = 0; i < n; i++) byClass.computeIfAbsent(y[i], k -> new ArrayList<>()).add(i);
        int K = byClass.size();

        double[] scores = new double[d];
        for (int j = 0; j < d; j++) {
            double globalMean = 0; for (double[] row : X) globalMean += row[j]; globalMean /= n;
            double ssBetween = 0, ssWithin = 0;
            for (Map.Entry<Double, List<Integer>> e : byClass.entrySet()) {
                List<Integer> idx = e.getValue();
                double classMean = 0; for (int i : idx) classMean += X[i][j]; classMean /= idx.size();
                ssBetween += idx.size() * Math.pow(classMean - globalMean, 2);
                for (int i : idx) ssWithin += Math.pow(X[i][j] - classMean, 2);
            }
            double msBetween = ssBetween / (K - 1);
            double msWithin  = ssWithin / (n - K + 1e-10);
            scores[j] = msWithin == 0 ? 0 : msBetween / msWithin;
        }
        return scores;
    }

    /**
     * 卡方检验（非负特征 + 分类目标）
     */
    public static double[] chi2(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        Map<Double, Integer> classCount = new HashMap<>();
        for (double v : y) classCount.merge(v, 1, Integer::sum);
        int K = classCount.size();
        double[] scores = new double[d];
        for (int j = 0; j < d; j++) {
            // Build observed contingency table (bin values into buckets)
            double min = X[0][j], max = X[0][j];
            for (double[] row : X) { min = Math.min(min, row[j]); max = Math.max(max, row[j]); }
            int bins = Math.min(10, n);
            Map<Integer, Map<Double, Integer>> cont = new HashMap<>();
            for (int b = 0; b < bins; b++) cont.put(b, new HashMap<>());
            for (int i = 0; i < n; i++) {
                int bin = max == min ? 0 : (int) Math.min(bins - 1, Math.floor((X[i][j] - min) / (max - min) * bins));
                cont.get(bin).merge(y[i], 1, Integer::sum);
            }
            double chi2 = 0;
            for (Map.Entry<Integer, Map<Double, Integer>> b : cont.entrySet()) {
                int rowTotal = b.getValue().values().stream().mapToInt(v -> v).sum();
                for (Map.Entry<Double, Integer> e : b.getValue().entrySet()) {
                    double expected = (double) rowTotal * classCount.getOrDefault(e.getKey(), 0) / n;
                    if (expected > 0) chi2 += Math.pow(e.getValue() - expected, 2) / expected;
                }
            }
            scores[j] = chi2;
        }
        return scores;
    }

    /**
     * 互信息（分类）
     */
    public static double[] mutualInfoClassif(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        Map<Double, Integer> yCount = new HashMap<>();
        for (double v : y) yCount.merge(v, 1, Integer::sum);
        double[] scores = new double[d];
        for (int j = 0; j < d; j++) {
            // Discretize feature into bins
            int bins = Math.min(10, n);
            double min = X[0][j], max = X[0][j];
            for (double[] row : X) { min = Math.min(min, row[j]); max = Math.max(max, row[j]); }
            int[] xBins = new int[n];
            for (int i = 0; i < n; i++)
                xBins[i] = max == min ? 0 : (int) Math.min(bins - 1, Math.floor((X[i][j] - min) / (max - min) * bins));
            Map<String, Integer> joint = new HashMap<>();
            Map<Integer, Integer> xCount = new HashMap<>();
            for (int i = 0; i < n; i++) {
                String key = xBins[i] + "_" + y[i];
                joint.merge(key, 1, Integer::sum);
                xCount.merge(xBins[i], 1, Integer::sum);
            }
            double mi = 0;
            for (Map.Entry<String, Integer> e : joint.entrySet()) {
                String[] parts = e.getKey().split("_");
                int xb = Integer.parseInt(parts[0]); double yv = Double.parseDouble(parts[1]);
                double pxy = (double) e.getValue() / n;
                double px = (double) xCount.getOrDefault(xb, 0) / n;
                double py = (double) yCount.getOrDefault(yv, 0) / n;
                if (pxy > 0 && px > 0 && py > 0) mi += pxy * Math.log(pxy / (px * py));
            }
            scores[j] = Math.max(0, mi);
        }
        return scores;
    }

    /**
     * 互信息（回归）— 用离散化目标近似
     */
    public static double[] mutualInfoRegression(double[][] X, double[] y) {
        // Bin y into classes for approximate MI
        int bins = 10;
        double min = y[0], max = y[0];
        for (double v : y) { min = Math.min(min, v); max = Math.max(max, v); }
        double[] yBinned = new double[y.length];
        for (int i = 0; i < y.length; i++)
            yBinned[i] = max == min ? 0 : (int) Math.min(bins - 1, Math.floor((y[i] - min) / (max - min) * bins));
        return mutualInfoClassif(X, yBinned);
    }
}

