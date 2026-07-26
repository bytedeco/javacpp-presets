package org.bytedeco.pytorch.data.dataframe.ml.regression;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/**
 * 梯度提升回归器（GBDT，MSE 损失）
 */
public class GradientBoostingRegressor extends BaseRegressor {
    private int nEstimators; private double learningRate;
    private int maxDepth; private double subsample; private Long randomState;
    private List<DecisionTreeRegressor> trees = new ArrayList<>();
    private double initPred;

    public GradientBoostingRegressor() { this(100, 0.1, 3, 1.0, null); }
    public GradientBoostingRegressor(int n, double lr, int depth, double sub, Long rs) {
        nEstimators = n; learningRate = lr; maxDepth = depth; subsample = sub; randomState = rs;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length;
        double sum = 0; for (double v : y) sum += v; initPred = sum / n;
        double[] F = new double[n]; Arrays.fill(F, initPred);
        Random rng = randomState == null ? new Random() : new Random(randomState);
        int subN = Math.max(1, (int)(n * subsample));
        trees.clear();
        for (int t = 0; t < nEstimators; t++) {
            double[] residuals = new double[n];
            for (int i = 0; i < n; i++) residuals[i] = y[i] - F[i]; // neg gradient of MSE
            int[] idx = subsampleIdx(n, subN, rng);
            double[][] subX = new double[subN][X[0].length]; double[] subR = new double[subN];
            for (int i = 0; i < subN; i++) { subX[i] = X[idx[i]]; subR[i] = residuals[idx[i]]; }
            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, 2, "mse", null);
            tree.fit(subX, subR); trees.add(tree);
            for (int i = 0; i < n; i++) F[i] += learningRate * tree.predict(new double[][]{X[i]})[0];
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] F = new double[X.length]; Arrays.fill(F, initPred);
        for (DecisionTreeRegressor t : trees) {
            double[] d = t.predict(X); for (int i = 0; i < X.length; i++) F[i] += learningRate * d[i];
        }
        return F;
    }

    private int[] subsampleIdx(int n, int k, Random rng) {
        List<Integer> all = new ArrayList<>(); for (int i = 0; i < n; i++) all.add(i);
        Collections.shuffle(all, rng); int[] idx = new int[k]; for (int i = 0; i < k; i++) idx[i] = all.get(i); return idx;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("learning_rate", learningRate);
        p.put("max_depth", maxDepth); p.put("subsample", subsample); p.put("random_state", randomState); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) nEstimators = ((Number) params.get("n_estimators")).intValue();
        if (params.containsKey("learning_rate")) learningRate = ((Number) params.get("learning_rate")).doubleValue();
    }
}

