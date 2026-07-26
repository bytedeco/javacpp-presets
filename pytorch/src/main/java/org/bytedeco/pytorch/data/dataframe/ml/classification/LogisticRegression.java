package org.bytedeco.pytorch.data.dataframe.ml.classification;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;

import java.util.*;

/**
 * 逻辑回归分类器（L2 正则化，梯度下降，支持二分类/多分类 OvR）
 */
public class LogisticRegression extends BaseClassifier {
    private String penalty;   // "l2" | "l1" | "none"
    private double C;         // 正则化强度倒数（C=1/lambda）
    private int maxIter;
    private double tol;
    private double learningRate;
    private Random random;
    private Long randomState;

    // learned params – OvR: one weight vector per class
    private double[][] weights; // [nClasses, nFeatures]
    private double[]   biases;  // [nClasses]
    private double[]   classes;

    public LogisticRegression() { this("l2", 1.0, 100, 1e-4, null); }
    public LogisticRegression(String penalty, double C, int maxIter, double tol, Long randomState) {
        this.penalty = penalty; this.C = C; this.maxIter = maxIter;
        this.tol = tol; this.randomState = randomState;
        this.learningRate = 0.1;
        this.random = randomState == null ? new Random() : new Random(randomState);
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        // unique classes
        TreeSet<Double> classSet = new TreeSet<>();
        for (double v : y) classSet.add(v);
        classes = classSet.stream().mapToDouble(Double::doubleValue).toArray();
        int K = classes.length;

        if (K == 2) {
            // binary logistic regression
            weights = new double[1][d];
            biases  = new double[1];
            double lambda = 1.0 / (C * n);
            for (int iter = 0; iter < maxIter; iter++) {
                double[] gradW = new double[d];
                double gradB = 0;
                double loss = 0;
                for (int i = 0; i < n; i++) {
                    double yBin = (y[i] == classes[1]) ? 1.0 : 0.0;
                    double pred = sigmoid(dot(weights[0], X[i]) + biases[0]);
                    double err  = pred - yBin;
                    loss += -yBin * Math.log(pred + 1e-15) - (1 - yBin) * Math.log(1 - pred + 1e-15);
                    for (int j = 0; j < d; j++) gradW[j] += err * X[i][j];
                    gradB += err;
                }
                // update
                for (int j = 0; j < d; j++) {
                    double reg = "none".equals(penalty) ? 0 : (lambda * weights[0][j]);
                    weights[0][j] -= learningRate * (gradW[j] / n + reg);
                }
                biases[0] -= learningRate * gradB / n;
                if (loss / n < tol) break;
            }
        } else {
            // OvR multiclass
            weights = new double[K][d];
            biases  = new double[K];
            double lambda = 1.0 / (C * n);
            for (int k = 0; k < K; k++) {
                double[] yBin = new double[n];
                for (int i = 0; i < n; i++) yBin[i] = (y[i] == classes[k]) ? 1.0 : 0.0;
                for (int iter = 0; iter < maxIter; iter++) {
                    double[] gradW = new double[d]; double gradB = 0;
                    for (int i = 0; i < n; i++) {
                        double pred = sigmoid(dot(weights[k], X[i]) + biases[k]);
                        double err  = pred - yBin[i];
                        for (int j = 0; j < d; j++) gradW[j] += err * X[i][j];
                        gradB += err;
                    }
                    for (int j = 0; j < d; j++) {
                        double reg = "none".equals(penalty) ? 0 : (lambda * weights[k][j]);
                        weights[k][j] -= learningRate * (gradW[j] / n + reg);
                    }
                    biases[k] -= learningRate * gradB / n;
                }
            }
        }
        fitted = true;
        return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] proba = predictProba(X);
        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int best = 0;
            for (int k = 1; k < proba[i].length; k++)
                if (proba[i][k] > proba[i][best]) best = k;
            preds[i] = classes[best];
        }
        return preds;
    }

    @Override
    public double[][] predictProba(double[][] X) {
        int n = X.length, K = classes.length;
        double[][] proba = new double[n][K];
        if (K == 2) {
            for (int i = 0; i < n; i++) {
                double p = sigmoid(dot(weights[0], X[i]) + biases[0]);
                proba[i][0] = 1 - p; proba[i][1] = p;
            }
        } else {
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int k = 0; k < K; k++) {
                    proba[i][k] = sigmoid(dot(weights[k], X[i]) + biases[k]);
                    sum += proba[i][k];
                }
                for (int k = 0; k < K; k++) proba[i][k] /= sum;
            }
        }
        return proba;
    }

    private double sigmoid(double z) { return 1.0 / (1.0 + Math.exp(-z)); }
    private double dot(double[] w, double[] x) {
        double s = 0; for (int i = 0; i < w.length; i++) s += w[i] * x[i]; return s;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("penalty", penalty); p.put("C", C); p.put("max_iter", maxIter);
        p.put("tol", tol); p.put("random_state", randomState);
        return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("C")) C = ((Number) params.get("C")).doubleValue();
        if (params.containsKey("penalty")) penalty = (String) params.get("penalty");
        if (params.containsKey("max_iter")) maxIter = ((Number) params.get("max_iter")).intValue();
        if (params.containsKey("tol")) tol = ((Number) params.get("tol")).doubleValue();
    }
}

