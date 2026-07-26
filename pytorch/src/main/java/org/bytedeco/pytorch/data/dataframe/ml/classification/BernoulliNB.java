package org.bytedeco.pytorch.data.dataframe.ml.classification;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 伯努利朴素贝叶斯（适合二值特征）
 */
public class BernoulliNB extends BaseClassifier {
    private double alpha;
    private double binarizeThreshold;
    private Map<Double, Double> classPriors = new HashMap<>();
    private Map<Double, double[]> featureLogProb  = new HashMap<>();
    private Map<Double, double[]> featureLogProb0 = new HashMap<>();
    private double[] classes;

    public BernoulliNB() { this(1.0, 0.0); }
    public BernoulliNB(double alpha, double binarize) { this.alpha = alpha; this.binarizeThreshold = binarize; }

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
            for (int i : ci) for (int j = 0; j < d; j++) if (X[i][j] > binarizeThreshold) counts[j]++;
            double[] lp1 = new double[d], lp0 = new double[d];
            for (int j = 0; j < d; j++) {
                double p = (counts[j] + alpha) / (ci.size() + 2 * alpha);
                lp1[j] = Math.log(p); lp0[j] = Math.log(1 - p);
            }
            featureLogProb.put(c, lp1); featureLogProb0.put(c, lp0);
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
                double[] lp1 = featureLogProb.get(c), lp0 = featureLogProb0.get(c);
                for (int j = 0; j < X[i].length; j++)
                    s += (X[i][j] > binarizeThreshold) ? lp1[j] : lp0[j];
                if (s > best) { best = s; bestC = c; }
            }
            preds[i] = bestC;
        }
        return preds;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("alpha", alpha); p.put("binarize", binarizeThreshold); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue();
        if (params.containsKey("binarize")) binarizeThreshold = ((Number) params.get("binarize")).doubleValue();
    }
}

