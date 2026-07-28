package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.dataframe.ml.regression.DecisionTreeRegressor;
import java.util.*;

/**
 * 梯度提升分类器（GBDT，使用对数损失）
 * 实现标准 Friedman GBDT with logistic loss (binary)
 */
public class GradientBoostingClassifier extends BaseClassifier {
    private int nEstimators;
    private double learningRate;
    private int maxDepth;
    private double subsample;
    private Long randomState;

    private List<DecisionTreeRegressor> trees = new ArrayList<>();
    private double initPred;  // log-odds prior
    private double[] classes;

    public GradientBoostingClassifier() { this(100, 0.1, 3, 1.0, null); }
    public GradientBoostingClassifier(int nEstimators, double lr, int maxDepth, double subsample, Long rs) {
        this.nEstimators = nEstimators; this.learningRate = lr; this.maxDepth = maxDepth;
        this.subsample = subsample; this.randomState = rs;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length;
        TreeSet<Double> cs = new TreeSet<>(); for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        // binary only for now
        double pos = 0; for (double v : y) if (v == classes[classes.length-1]) pos++;
        initPred = Math.log((pos + 1e-10) / (n - pos + 1e-10));

        double[] F = new double[n]; Arrays.fill(F, initPred);
        Random rng = randomState == null ? new Random() : new Random(randomState);
        int subN = (int)(n * subsample);

        for (int t = 0; t < nEstimators; t++) {
            // compute residuals (negative gradient of log-loss)
            double[] residuals = new double[n];
            for (int i = 0; i < n; i++) {
                double prob = sigmoid(F[i]);
                double yBin = (y[i] == classes[classes.length-1]) ? 1.0 : 0.0;
                residuals[i] = yBin - prob;
            }

            // subsample
            int[] idx = subsampleIdx(n, subN, rng);
            double[][] subX = new double[subN][X[0].length];
            double[]   subR = new double[subN];
            for (int i = 0; i < subN; i++) { subX[i] = X[idx[i]]; subR[i] = residuals[idx[i]]; }

            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, 2, "mse", null);
            tree.fit(subX, subR);
            trees.add(tree);

            // update F
            for (int i = 0; i < n; i++) F[i] += learningRate * tree.predict(new double[][]{X[i]})[0];
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] proba = predictProba1D(X);
        double threshold = classes.length > 1 ? 0.5 : 0.5;
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++)
            result[i] = proba[i] >= threshold ? classes[classes.length-1] : classes[0];
        return result;
    }

    @Override
    public double[][] predictProba(double[][] X) {
        double[] p1 = predictProba1D(X);
        double[][] proba = new double[X.length][2];
        for (int i = 0; i < X.length; i++) { proba[i][0] = 1 - p1[i]; proba[i][1] = p1[i]; }
        return proba;
    }

    private double[] predictProba1D(double[][] X) {
        double[] F = new double[X.length]; Arrays.fill(F, initPred);
        for (DecisionTreeRegressor t : trees) {
            double[] delta = t.predict(X);
            for (int i = 0; i < X.length; i++) F[i] += learningRate * delta[i];
        }
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) p[i] = sigmoid(F[i]);
        return p;
    }

    private double sigmoid(double z) { return 1.0 / (1 + Math.exp(-z)); }

    private int[] subsampleIdx(int n, int k, Random rng) {
        List<Integer> all = new ArrayList<>(); for (int i = 0; i < n; i++) all.add(i);
        Collections.shuffle(all, rng);
        int[] idx = new int[k]; for (int i = 0; i < k; i++) idx[i] = all.get(i);
        return idx;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("learning_rate", learningRate);
        p.put("max_depth", maxDepth); p.put("subsample", subsample);
        p.put("random_state", randomState); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) nEstimators = ((Number) params.get("n_estimators")).intValue();
        if (params.containsKey("learning_rate")) learningRate = ((Number) params.get("learning_rate")).doubleValue();
        if (params.containsKey("max_depth")) maxDepth = ((Number) params.get("max_depth")).intValue();
    }
}

