package org.bytedeco.pytorch.data.dataframe.ml.regression;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** AdaBoost 回归器（AdaBoost.R2 算法） */
public class AdaBoostRegressor extends BaseRegressor {
    private int nEstimators; private double learningRate; private Long randomState;
    private List<DecisionTreeRegressor> estimators = new ArrayList<>();
    private List<Double> estimatorWeights = new ArrayList<>();

    public AdaBoostRegressor() { this(50, 1.0, null); }
    public AdaBoostRegressor(int n, double lr, Long rs) { nEstimators = n; learningRate = lr; randomState = rs; }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length;
        double[] w = new double[n]; Arrays.fill(w, 1.0 / n);
        Random rng = randomState == null ? new Random() : new Random(randomState);
        for (int t = 0; t < nEstimators; t++) {
            int[] sample = weightedSample(w, n, rng);
            double[][] sX = new double[n][X[0].length]; double[] sY = new double[n];
            for (int i = 0; i < n; i++) { sX[i] = X[sample[i]]; sY[i] = y[sample[i]]; }
            DecisionTreeRegressor tree = new DecisionTreeRegressor(3, 1, "mse", null);
            tree.fit(sX, sY); double[] preds = tree.predict(X);

            // Compute adjusted errors
            double maxErr = 0;
            double[] abserr = new double[n];
            for (int i = 0; i < n; i++) { abserr[i] = Math.abs(preds[i] - y[i]); maxErr = Math.max(maxErr, abserr[i]); }
            if (maxErr == 0) { estimators.add(tree); estimatorWeights.add(1.0); break; }
            double[] adjErr = new double[n];
            for (int i = 0; i < n; i++) adjErr[i] = abserr[i] / maxErr;
            double loss = 0; for (int i = 0; i < n; i++) loss += w[i] * adjErr[i];
            if (loss >= 0.5) break;
            double beta = loss / (1 - loss);
            double alpha = learningRate * Math.log(1 / (beta + 1e-10));
            estimators.add(tree); estimatorWeights.add(alpha);
            for (int i = 0; i < n; i++) w[i] *= Math.pow(beta, 1 - adjErr[i]);
            double wSum = 0; for (double wi : w) wSum += wi;
            for (int i = 0; i < n; i++) w[i] /= wSum;
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        if (estimators.isEmpty()) return new double[X.length];
        double[][] allPreds = new double[estimators.size()][X.length];
        for (int t = 0; t < estimators.size(); t++) allPreds[t] = estimators.get(t).predict(X);
        // Weighted median
        double[] result = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                double totalW = 0; for (double w : estimatorWeights) totalW += w;
                double target = totalW / 2;
                double cumW = 0;
                // sort by prediction using Integer array
                Integer[] order = new Integer[estimators.size()];
                for (int t = 0; t < estimators.size(); t++) order[t] = t;
                final int fi = i;
                Arrays.sort(order, (a,b) -> Double.compare(allPreds[a][fi], allPreds[b][fi]));
                for (int t : order) { cumW += estimatorWeights.get(t); if (cumW >= target) { result[i] = allPreds[t][i]; break; } }
            }
        return result;
    }

    private int[] weightedSample(double[] w, int n, Random rng) {
        double[] cdf = new double[n]; cdf[0] = w[0];
        for (int i = 1; i < n; i++) cdf[i] = cdf[i-1] + w[i];
        int[] s = new int[n];
        for (int i = 0; i < n; i++) {
            double r = rng.nextDouble() * cdf[n-1]; int lo = 0, hi = n-1;
            while (lo < hi) { int m = (lo+hi)/2; if (cdf[m] < r) lo = m+1; else hi = m; }
            s[i] = lo;
        }
        return s;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("learning_rate", learningRate); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) nEstimators = ((Number) params.get("n_estimators")).intValue();
    }
}

