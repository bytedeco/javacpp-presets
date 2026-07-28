package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 随机梯度下降分类器
 * loss: "log_loss"(logistic) | "hinge"(linear SVM) | "perceptron"
 */
public class SGDClassifier extends BaseClassifier {
    private String loss;
    private String penalty; // "l2" | "l1" | "elasticnet" | "none"
    private double alpha;   // regularization strength
    private int maxIter;
    private double tol;
    private double eta0;    // initial learning rate
    private Long randomState;

    private double[][] weights;
    private double[]   biases;
    private double[]   classes;

    public SGDClassifier() { this("log_loss", "l2", 1e-4, 1000, 1e-3, 0.01, null); }
    public SGDClassifier(String loss, String penalty, double alpha, int maxIter, double tol, double eta0, Long randomState) {
        this.loss = loss; this.penalty = penalty; this.alpha = alpha;
        this.maxIter = maxIter; this.tol = tol; this.eta0 = eta0; this.randomState = randomState;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        TreeSet<Double> cs = new TreeSet<>();
        for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        int K = classes.length;
        Random rng = randomState == null ? new Random() : new Random(randomState);

        weights = new double[K][d]; biases = new double[K];
        int[] order = new int[n]; for (int i = 0; i < n; i++) order[i] = i;

        for (int iter = 0; iter < maxIter; iter++) {
            // shuffle
            for (int i = n - 1; i > 0; i--) { int j = rng.nextInt(i+1); int tmp = order[i]; order[i] = order[j]; order[j] = tmp; }
            double totalLoss = 0;
            for (int ii : order) {
                for (int k = 0; k < K; k++) {
                    double yBin = (y[ii] == classes[k]) ? 1.0 : -1.0;
                    double score = dot(weights[k], X[ii]) + biases[k];
                    double grad;
                    if ("hinge".equals(loss) || "perceptron".equals(loss)) {
                        double margin = yBin * score;
                        if (margin >= ("perceptron".equals(loss) ? 0 : 1)) continue;
                        grad = -yBin; totalLoss += Math.max(0, 1 - margin);
                    } else { // log_loss
                        double prob = 1.0 / (1 + Math.exp(-score));
                        double yPos = (yBin > 0) ? 1.0 : 0.0;
                        grad = prob - yPos;
                        totalLoss += -yPos * Math.log(prob + 1e-15) - (1 - yPos) * Math.log(1 - prob + 1e-15);
                    }
                    for (int j = 0; j < d; j++) {
                        double reg = "none".equals(penalty) ? 0 : alpha * weights[k][j];
                        weights[k][j] -= eta0 * (grad * X[ii][j] + reg);
                    }
                    biases[k] -= eta0 * grad;
                }
            }
            if (totalLoss / n < tol) break;
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int best = 0; double bestScore = dot(weights[0], X[i]) + biases[0];
            for (int k = 1; k < classes.length; k++) {
                double s = dot(weights[k], X[i]) + biases[k];
                if (s > bestScore) { bestScore = s; best = k; }
            }
            preds[i] = classes[best];
        }
        return preds;
    }

    private double dot(double[] w, double[] x) { double s = 0; for (int i = 0; i < w.length; i++) s += w[i]*x[i]; return s; }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("loss", loss); p.put("penalty", penalty); p.put("alpha", alpha);
        p.put("max_iter", maxIter); p.put("tol", tol); p.put("eta0", eta0);
        p.put("random_state", randomState); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("loss")) loss = (String) params.get("loss");
        if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue();
        if (params.containsKey("max_iter")) maxIter = ((Number) params.get("max_iter")).intValue();
    }
}

