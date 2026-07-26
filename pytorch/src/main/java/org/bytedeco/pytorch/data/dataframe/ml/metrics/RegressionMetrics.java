package org.bytedeco.pytorch.data.dataframe.ml.metrics;

import java.util.Arrays;

/**
 * 回归评估指标（对应 sklearn.metrics 回归部分）
 */
public class RegressionMetrics {

    public static double meanSquaredError(double[] yTrue, double[] yPred) {
        double s = 0; for (int i = 0; i < yTrue.length; i++) s += Math.pow(yTrue[i] - yPred[i], 2);
        return s / yTrue.length;
    }

    public static double rootMeanSquaredError(double[] yTrue, double[] yPred) {
        return Math.sqrt(meanSquaredError(yTrue, yPred));
    }

    public static double meanAbsoluteError(double[] yTrue, double[] yPred) {
        double s = 0; for (int i = 0; i < yTrue.length; i++) s += Math.abs(yTrue[i] - yPred[i]);
        return s / yTrue.length;
    }

    public static double r2Score(double[] yTrue, double[] yPred) {
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < yTrue.length; i++) {
            ssTot += Math.pow(yTrue[i] - mean, 2);
            ssRes += Math.pow(yTrue[i] - yPred[i], 2);
        }
        return ssTot == 0 ? 1.0 : 1.0 - ssRes / ssTot;
    }

    public static double meanSquaredLogError(double[] yTrue, double[] yPred) {
        double s = 0;
        for (int i = 0; i < yTrue.length; i++)
            s += Math.pow(Math.log1p(Math.max(0, yTrue[i])) - Math.log1p(Math.max(0, yPred[i])), 2);
        return s / yTrue.length;
    }

    public static double explainedVarianceScore(double[] yTrue, double[] yPred) {
        double meanTrue = Arrays.stream(yTrue).average().orElse(0);
        double[] diff = new double[yTrue.length];
        for (int i = 0; i < yTrue.length; i++) diff[i] = yTrue[i] - yPred[i];
        double meanDiff = Arrays.stream(diff).average().orElse(0);
        double varDiff = 0, varTrue = 0;
        for (int i = 0; i < yTrue.length; i++) {
            varDiff += Math.pow(diff[i] - meanDiff, 2);
            varTrue += Math.pow(yTrue[i] - meanTrue, 2);
        }
        return varTrue == 0 ? 0 : 1 - varDiff / varTrue;
    }

    public static double maxError(double[] yTrue, double[] yPred) {
        double max = 0; for (int i = 0; i < yTrue.length; i++) max = Math.max(max, Math.abs(yTrue[i] - yPred[i]));
        return max;
    }

    public static double meanAbsolutePercentageError(double[] yTrue, double[] yPred) {
        double s = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (Math.abs(yTrue[i]) > 1e-10) s += Math.abs((yTrue[i] - yPred[i]) / yTrue[i]);
        }
        return s / yTrue.length;
    }

    /** Silhouette score – needs X and cluster labels */
    public static double silhouetteScore(double[][] X, double[] labels) {
        int n = X.length;
        double total = 0;
        for (int i = 0; i < n; i++) {
            double a = intraClusterDist(X, labels, i);
            double b = nearestClusterDist(X, labels, i);
            double denom = Math.max(a, b);
            total += denom == 0 ? 0 : (b - a) / denom;
        }
        return total / n;
    }

    public static double calinskiHarabaszScore(double[][] X, double[] labels) {
        int n = X.length, d = X[0].length;
        java.util.Map<Double, java.util.List<Integer>> clusters = new java.util.HashMap<>();
        for (int i = 0; i < n; i++) clusters.computeIfAbsent(labels[i], k -> new java.util.ArrayList<>()).add(i);
        int K = clusters.size(); if (K <= 1) return 0;
        double[] globalMean = new double[d];
        for (double[] row : X) for (int j = 0; j < d; j++) globalMean[j] += row[j] / n;

        double bcd = 0, wcd = 0;
        for (java.util.Map.Entry<Double, java.util.List<Integer>> e : clusters.entrySet()) {
            java.util.List<Integer> idx = e.getValue();
            double[] mu = new double[d];
            for (int i : idx) for (int j = 0; j < d; j++) mu[j] += X[i][j] / idx.size();
            double dist2 = 0; for (int j = 0; j < d; j++) dist2 += Math.pow(mu[j] - globalMean[j], 2);
            bcd += idx.size() * dist2;
            for (int i : idx) { double wd = 0; for (int j = 0; j < d; j++) wd += Math.pow(X[i][j] - mu[j], 2); wcd += wd; }
        }
        return (bcd / (K - 1)) / (wcd / (n - K) + 1e-10);
    }

    public static double daviesBouldinScore(double[][] X, double[] labels) {
        java.util.Map<Double, java.util.List<Integer>> clusters = new java.util.HashMap<>();
        for (int i = 0; i < X.length; i++) clusters.computeIfAbsent(labels[i], k -> new java.util.ArrayList<>()).add(i);
        Double[] keys = clusters.keySet().toArray(new Double[0]);
        int K = keys.length; if (K <= 1) return 0;
        int d = X[0].length;
        double[][] centroids = new double[K][d];
        double[] scatter = new double[K];
        for (int k = 0; k < K; k++) {
            java.util.List<Integer> idx = clusters.get(keys[k]);
            for (int i : idx) for (int j = 0; j < d; j++) centroids[k][j] += X[i][j] / idx.size();
            double s = 0; for (int i : idx) { double di = 0; for (int j = 0; j < d; j++) di += Math.pow(X[i][j]-centroids[k][j],2); s += Math.sqrt(di); }
            scatter[k] = s / idx.size();
        }
        double db = 0;
        for (int i = 0; i < K; i++) {
            double maxR = 0;
            for (int j = 0; j < K; j++) {
                if (i == j) continue;
                double dist = 0; for (int jj = 0; jj < d; jj++) dist += Math.pow(centroids[i][jj]-centroids[j][jj],2);
                double R = (scatter[i] + scatter[j]) / (Math.sqrt(dist) + 1e-10);
                maxR = Math.max(maxR, R);
            }
            db += maxR;
        }
        return db / K;
    }

    private static double intraClusterDist(double[][] X, double[] labels, int idx) {
        double sum = 0; int count = 0;
        for (int i = 0; i < X.length; i++) {
            if (i != idx && labels[i] == labels[idx]) { sum += euclidean(X[idx], X[i]); count++; }
        }
        return count == 0 ? 0 : sum / count;
    }

    private static double nearestClusterDist(double[][] X, double[] labels, int idx) {
        java.util.Map<Double, Double> clusterDist = new java.util.HashMap<>();
        java.util.Map<Double, Integer> clusterCount = new java.util.HashMap<>();
        for (int i = 0; i < X.length; i++) {
            if (labels[i] != labels[idx]) {
                clusterDist.merge(labels[i], euclidean(X[idx], X[i]), Double::sum);
                clusterCount.merge(labels[i], 1, Integer::sum);
            }
        }
        return clusterDist.entrySet().stream()
            .mapToDouble(e -> e.getValue() / clusterCount.get(e.getKey()))
            .min().orElse(0);
    }

    private static double euclidean(double[] a, double[] b) {
        double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]); return Math.sqrt(s);
    }
}

