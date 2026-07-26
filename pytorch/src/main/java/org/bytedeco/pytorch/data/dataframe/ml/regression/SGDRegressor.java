package org.bytedeco.pytorch.data.dataframe.ml.regression;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** SGD 回归器 */
public class SGDRegressor extends BaseRegressor {
    private String loss; // "squared_error" | "huber" | "epsilon_insensitive"
    private String penalty; private double alpha;
    private int maxIter; private double tol; private double eta0; private Long randomState;
    private double[] coef; private double intercept;

    public SGDRegressor() { this("squared_error","l2",1e-4,1000,1e-3,0.01,null); }
    public SGDRegressor(String loss, String penalty, double alpha, int maxIter, double tol, double eta0, Long rs) {
        this.loss = loss; this.penalty = penalty; this.alpha = alpha;
        this.maxIter = maxIter; this.tol = tol; this.eta0 = eta0; this.randomState = rs;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        coef = new double[d]; intercept = 0;
        Random rng = randomState == null ? new Random() : new Random(randomState);
        int[] order = new int[n]; for (int i = 0; i < n; i++) order[i] = i;
        for (int iter = 0; iter < maxIter; iter++) {
            for (int i = n-1; i > 0; i--) { int j = rng.nextInt(i+1); int tmp = order[i]; order[i] = order[j]; order[j] = tmp; }
            double totalLoss = 0;
            for (int ii : order) {
                double pred = intercept; for (int j = 0; j < d; j++) pred += coef[j] * X[ii][j];
                double err = pred - y[ii];
                double grad = err; // squared_error gradient
                totalLoss += err * err;
                for (int j = 0; j < d; j++) {
                    double reg = "none".equals(penalty) ? 0 : alpha * coef[j];
                    coef[j] -= eta0 * (grad * X[ii][j] + reg);
                }
                intercept -= eta0 * grad;
            }
            if (totalLoss / n < tol) break;
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            p[i] = intercept; for (int j = 0; j < coef.length; j++) p[i] += coef[j] * X[i][j];
        }
        return p;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("loss", loss); p.put("penalty", penalty); p.put("alpha", alpha);
        p.put("max_iter", maxIter); p.put("eta0", eta0); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("alpha")) alpha = ((Number) params.get("alpha")).doubleValue();
        if (params.containsKey("max_iter")) maxIter = ((Number) params.get("max_iter")).intValue();
    }
}

