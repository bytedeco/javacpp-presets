package org.bytedeco.pytorch.data.dataframe.ml.regression;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** 随机森林回归器 */
public class RandomForestRegressor extends BaseRegressor {
    private int nEstimators; private Integer maxDepth; private int minSamplesSplit;
    private Integer maxFeatures; private Long randomState;
    private List<DecisionTreeRegressor> trees = new ArrayList<>();
    private List<int[]> treeFeatures = new ArrayList<>();

    public RandomForestRegressor() { this(100, null, 2, null, null); }
    public RandomForestRegressor(int nEstimators) { this(nEstimators, null, 2, null, null); }
    public RandomForestRegressor(int nEstimators, Integer maxDepth, int minSamplesSplit,
                                  Integer maxFeatures, Long randomState) {
        this.nEstimators = nEstimators; this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit; this.maxFeatures = maxFeatures; this.randomState = randomState;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        Random rng = randomState == null ? new Random() : new Random(randomState);
        int mf = maxFeatures == null ? Math.max(1, (int)(d / 3.0)) : Math.min(maxFeatures, d);
        trees.clear(); treeFeatures.clear();
        for (int t = 0; t < nEstimators; t++) {
            int[] bag = new int[n]; for (int i = 0; i < n; i++) bag[i] = rng.nextInt(n);
            double[][] bX = new double[n][d]; double[] bY = new double[n];
            for (int i = 0; i < n; i++) { bX[i] = X[bag[i]]; bY[i] = y[bag[i]]; }
            int[] fi = sampleFeatures(d, mf, rng);
            double[][] subX = selectFeatures(bX, fi);
            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, minSamplesSplit, "mse",
                randomState == null ? null : randomState + t);
            tree.fit(subX, bY); trees.add(tree); treeFeatures.add(fi);
        }
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] sum = new double[X.length];
        for (int t = 0; t < trees.size(); t++) {
            double[] p = trees.get(t).predict(selectFeatures(X, treeFeatures.get(t)));
            for (int i = 0; i < X.length; i++) sum[i] += p[i];
        }
        for (int i = 0; i < X.length; i++) sum[i] /= trees.size();
        return sum;
    }

    private int[] sampleFeatures(int d, int mf, Random rng) {
        List<Integer> all = new ArrayList<>(); for (int i = 0; i < d; i++) all.add(i);
        Collections.shuffle(all, rng);
        int[] fi = new int[mf]; for (int i = 0; i < mf; i++) fi[i] = all.get(i); return fi;
    }
    private double[][] selectFeatures(double[][] X, int[] fi) {
        double[][] r = new double[X.length][fi.length];
        for (int i = 0; i < X.length; i++) for (int j = 0; j < fi.length; j++) r[i][j] = X[i][fi[j]];
        return r;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("max_depth", maxDepth);
        p.put("random_state", randomState); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) nEstimators = ((Number) params.get("n_estimators")).intValue();
    }
}

