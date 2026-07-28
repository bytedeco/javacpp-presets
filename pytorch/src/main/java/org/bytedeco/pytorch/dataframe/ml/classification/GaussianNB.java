package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;

import java.util.*;

/**
 * 高斯朴素贝叶斯分类器
 */
public class GaussianNB extends BaseClassifier {
    private double varSmoothing;
    private Map<Double, Double> classPriors = new HashMap<>();
    private Map<Double, double[]> means = new HashMap<>();
    private Map<Double, double[]> vars  = new HashMap<>();
    private double[] classes;

    public GaussianNB() { this(1e-9); }
    public GaussianNB(double varSmoothing) { this.varSmoothing = varSmoothing; }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        Map<Double, List<Integer>> idx = new HashMap<>();
        for (int i = 0; i < n; i++) idx.computeIfAbsent(y[i], k -> new ArrayList<>()).add(i);
        TreeSet<Double> cs = new TreeSet<>(idx.keySet());
        classes = cs.stream().mapToDouble(v -> v).toArray();
        for (double c : classes) {
            List<Integer> ci = idx.get(c);
            classPriors.put(c, (double) ci.size() / n);
            double[] mean = new double[d], var = new double[d];
            for (int j = 0; j < d; j++) {
                double sum = 0;
                for (int i : ci) sum += X[i][j];
                mean[j] = sum / ci.size();
                double ss = 0;
                for (int i : ci) ss += Math.pow(X[i][j] - mean[j], 2);
                var[j] = ss / ci.size() + varSmoothing;
            }
            means.put(c, mean); vars.put(c, var);
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double bestLog = Double.NEGATIVE_INFINITY; double best = classes[0];
            for (double c : classes) {
                double logP = Math.log(classPriors.get(c));
                double[] mu = means.get(c); double[] v = vars.get(c);
                for (int j = 0; j < X[i].length; j++)
                    logP += -0.5 * Math.log(2 * Math.PI * v[j]) - 0.5 * Math.pow(X[i][j] - mu[j], 2) / v[j];
                if (logP > bestLog) { bestLog = logP; best = c; }
            }
            preds[i] = best;
        }
        return preds;
    }

    @Override
    public double[][] predictProba(double[][] X) {
        int n = X.length, K = classes.length;
        double[][] proba = new double[n][K];
        for (int i = 0; i < n; i++) {
            double[] logP = new double[K]; double maxLog = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < K; k++) {
                logP[k] = Math.log(classPriors.get(classes[k]));
                double[] mu = means.get(classes[k]); double[] v = vars.get(classes[k]);
                for (int j = 0; j < X[i].length; j++)
                    logP[k] += -0.5 * Math.log(2 * Math.PI * v[j]) - 0.5 * Math.pow(X[i][j] - mu[j], 2) / v[j];
                if (logP[k] > maxLog) maxLog = logP[k];
            }
            double sum = 0;
            for (int k = 0; k < K; k++) { proba[i][k] = Math.exp(logP[k] - maxLog); sum += proba[i][k]; }
            for (int k = 0; k < K; k++) proba[i][k] /= sum;
        }
        return proba;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("var_smoothing", varSmoothing); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("var_smoothing")) varSmoothing = ((Number) params.get("var_smoothing")).doubleValue();
    }
}

