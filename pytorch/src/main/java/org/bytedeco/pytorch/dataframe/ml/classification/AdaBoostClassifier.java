package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * AdaBoost 分类器（SAMME 算法，基学习器为决策树）
 */
public class AdaBoostClassifier extends BaseClassifier {
    private int nEstimators;
    private double learningRate;
    private Long randomState;

    private List<DecisionTreeClassifier> estimators = new ArrayList<>();
    private List<Double> estimatorWeights = new ArrayList<>();
    private double[] classes;

    public AdaBoostClassifier() { this(50, 1.0, null); }
    public AdaBoostClassifier(int nEstimators, double learningRate, Long randomState) {
        this.nEstimators = nEstimators; this.learningRate = learningRate; this.randomState = randomState;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length;
        TreeSet<Double> cs = new TreeSet<>();
        for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        int K = classes.length;

        double[] w = new double[n];
        Arrays.fill(w, 1.0 / n);
        Random rng = randomState == null ? new Random() : new Random(randomState);

        for (int t = 0; t < nEstimators; t++) {
            // Weighted bootstrap
            int[] sample = weightedSample(w, n, rng);
            double[][] sX = new double[n][X[0].length];
            double[]   sY = new double[n];
            for (int i = 0; i < n; i++) { sX[i] = X[sample[i]]; sY[i] = y[sample[i]]; }

            DecisionTreeClassifier tree = new DecisionTreeClassifier(1, 1, "gini", null);
            tree.fit(sX, sY);
            double[] preds = tree.predict(X);

            double err = 0;
            for (int i = 0; i < n; i++) if (preds[i] != y[i]) err += w[i];
            err = Math.max(err, 1e-10);
            if (err >= 1.0 - 1.0 / K) break;

            double alpha = learningRate * Math.log((1 - err) / err) + Math.log(K - 1);
            estimators.add(tree);
            estimatorWeights.add(alpha);

            // update weights
            for (int i = 0; i < n; i++) if (preds[i] != y[i]) w[i] *= Math.exp(alpha);
            double wSum = 0; for (double wi : w) wSum += wi;
            for (int i = 0; i < n; i++) w[i] /= wSum;
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] scores = new double[X.length][classes.length];
        for (int t = 0; t < estimators.size(); t++) {
            double alpha = estimatorWeights.get(t);
            double[] preds = estimators.get(t).predict(X);
            for (int i = 0; i < X.length; i++) {
                for (int c = 0; c < classes.length; c++) {
                    if (preds[i] == classes[c]) { scores[i][c] += alpha; break; }
                }
            }
        }
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int best = 0;
            for (int c = 1; c < classes.length; c++) if (scores[i][c] > scores[i][best]) best = c;
            result[i] = classes[best];
        }
        return result;
    }

    private int[] weightedSample(double[] w, int n, Random rng) {
        // build CDF
        double[] cdf = new double[n];
        cdf[0] = w[0];
        for (int i = 1; i < n; i++) cdf[i] = cdf[i-1] + w[i];
        int[] sample = new int[n];
        for (int i = 0; i < n; i++) {
            double r = rng.nextDouble() * cdf[n-1];
            int lo = 0, hi = n-1;
            while (lo < hi) { int mid = (lo+hi)/2; if (cdf[mid] < r) lo = mid+1; else hi = mid; }
            sample[i] = lo;
        }
        return sample;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("learning_rate", learningRate);
        p.put("random_state", randomState); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) nEstimators = ((Number) params.get("n_estimators")).intValue();
        if (params.containsKey("learning_rate")) learningRate = ((Number) params.get("learning_rate")).doubleValue();
    }
}

