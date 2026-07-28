package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 多项式朴素贝叶斯（适合计数/TF-IDF 特征，alpha 为拉普拉斯平滑）
 */
public class MultinomialNB extends BaseClassifier {
    private double alpha;
    private Map<Double, Double> classPriors = new HashMap<>();
    private Map<Double, double[]> featureLogProb = new HashMap<>();
    private double[] classes;

    public MultinomialNB() { this(1.0); }
    public MultinomialNB(double alpha) { this.alpha = alpha; }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        Map<Double, List<Integer>> idx = new HashMap<>();
        for (int i = 0; i < n; i++) idx.computeIfAbsent(y[i], k -> new ArrayList<>()).add(i);
        TreeSet<Double> cs = new TreeSet<>(idx.keySet());
        classes = cs.stream().mapToDouble(v -> v).toArray();
        for (double c : classes) {
            List<Integer> ci = idx.get(c);
            classPriors.put(c, Math.log((double) ci.size() / n));
            double[] counts = new double[d];
            double total = 0;
            for (int i : ci) for (int j = 0; j < d; j++) { counts[j] += X[i][j] + alpha; total += X[i][j] + alpha; }
            double[] logP = new double[d];
            for (int j = 0; j < d; j++) logP[j] = Math.log(counts[j] / total);
            featureLogProb.put(c, logP);
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double best = Double.NEGATIVE_INFINITY; double bestC = classes[0];
            for (double c : classes) {
                double s = classPriors.get(c);
                double[] lp = featureLogProb.get(c);
                for (int j = 0; j < X[i].length; j++) s += X[i][j] * lp[j];
                if (s > best) { best = s; bestC = c; }
            }
            preds[i] = bestC;
        }
        return preds;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("alpha", alpha); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue();
    }
}

