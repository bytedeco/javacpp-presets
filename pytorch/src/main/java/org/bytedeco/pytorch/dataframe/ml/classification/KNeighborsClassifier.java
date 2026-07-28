package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;

import java.util.*;

/**
 * K 近邻分类器（支持 euclidean / manhattan / cosine 距离）
 */
public class KNeighborsClassifier extends BaseClassifier {
    private int k;
    private String metric; // "euclidean" | "manhattan" | "cosine"
    private double[][] trainX;
    private double[]   trainY;

    public KNeighborsClassifier() { this(5); }
    public KNeighborsClassifier(int k) { this(k, "euclidean"); }
    public KNeighborsClassifier(int k, String metric) { this.k = k; this.metric = metric; }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        this.trainX = X; this.trainY = y; this.fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int[] neighbors = kNearest(X[i]);
            preds[i] = majority(neighbors);
        }
        return preds;
    }

    @Override
    public double[][] predictProba(double[][] X) {
        TreeSet<Double> cs = new TreeSet<>();
        for (double v : trainY) cs.add(v);
        double[] classes = cs.stream().mapToDouble(d -> d).toArray();
        double[][] proba = new double[X.length][classes.length];
        for (int i = 0; i < X.length; i++) {
            int[] neighbors = kNearest(X[i]);
            Map<Double, Integer> votes = new HashMap<>();
            for (int n : neighbors) votes.merge(trainY[n], 1, Integer::sum);
            for (int c = 0; c < classes.length; c++)
                proba[i][c] = votes.getOrDefault(classes[c], 0) / (double) k;
        }
        return proba;
    }

    private int[] kNearest(double[] x) {
        int n = trainX.length;
        double[] dists = new double[n];
        for (int i = 0; i < n; i++) dists[i] = dist(x, trainX[i]);
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> dists[i]));
        int[] result = new int[k];
        for (int i = 0; i < k; i++) result[i] = idx[i];
        return result;
    }

    private double majority(int[] neighbors) {
        Map<Double, Integer> votes = new HashMap<>();
        for (int n : neighbors) votes.merge(trainY[n], 1, Integer::sum);
        return votes.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(0.0);
    }

    private double dist(double[] a, double[] b) {
        return switch (metric) {
            case "manhattan" -> { double s = 0; for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); yield s; }
            case "cosine"    -> { double dot = 0, na = 0, nb = 0;
                for (int i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
                yield 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10); }
            default          -> { double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]); yield Math.sqrt(s); }
        };
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_neighbors", k); p.put("metric", metric); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_neighbors")) k = ((Number) params.get("n_neighbors")).intValue();
        if (params.containsKey("metric")) metric = (String) params.get("metric");
    }
}

