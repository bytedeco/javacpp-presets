package org.bytedeco.pytorch.dataframe.ml.classification;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;

import java.util.*;

/**
 * 随机森林分类器（Bagging + 随机特征子集）
 */
public class RandomForestClassifier extends BaseClassifier {
    private int nEstimators;
    private Integer maxDepth;
    private int minSamplesSplit;
    private Integer maxFeatures; // null = sqrt(n_features)
    private Long randomState;

    private List<DecisionTreeClassifier> trees = new ArrayList<>();
    private List<int[]> treeFeatures = new ArrayList<>();
    private double[] classes;
    private int nFeatures;

    public RandomForestClassifier() { this(100, null, 2, null, null); }
    public RandomForestClassifier(int nEstimators) { this(nEstimators, null, 2, null, null); }
    public RandomForestClassifier(int nEstimators, Integer maxDepth, int minSamplesSplit,
                                   Integer maxFeatures, Long randomState) {
        this.nEstimators = nEstimators; this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit; this.maxFeatures = maxFeatures;
        this.randomState = randomState;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        nFeatures = d;
        TreeSet<Double> cs = new TreeSet<>();
        for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();

        Random rng = randomState == null ? new Random() : new Random(randomState);
        int mf = maxFeatures == null ? Math.max(1, (int) Math.sqrt(d)) : Math.min(maxFeatures, d);

        trees.clear();
        treeFeatures.clear();
        for (int t = 0; t < nEstimators; t++) {
            // Bootstrap sample
            int[] bag = new int[n];
            for (int i = 0; i < n; i++) bag[i] = rng.nextInt(n);
            double[][] bX = new double[n][d];
            double[]   bY = new double[n];
            for (int i = 0; i < n; i++) { bX[i] = X[bag[i]]; bY[i] = y[bag[i]]; }

            // Random feature subset
            int[] featIdx = sampleFeatures(d, mf, rng);
            double[][] subX = selectFeatures(bX, featIdx);

            DecisionTreeClassifier tree = new DecisionTreeClassifier(maxDepth, minSamplesSplit, "gini",
                randomState == null ? null : randomState + t);
            tree.fit(subX, bY);
            trees.add(tree);
            treeFeatures.add(featIdx);
        }
        fitted = true;
        return this;
    }

    /**
     * Mean decrease in impurity across trees, mapped back to original feature indices.
     * Normalized to sum to 1 (sklearn RandomForestClassifier.feature_importances_).
     */
    public double[] getFeatureImportances() {
        double[] imp = new double[nFeatures];
        if (trees.isEmpty()) return imp;
        for (int t = 0; t < trees.size(); t++) {
            double[] local = trees.get(t).getFeatureImportances();
            int[] fi = treeFeatures.get(t);
            for (int j = 0; j < fi.length && j < local.length; j++) {
                imp[fi[j]] += local[j];
            }
        }
        double s = 0;
        for (double v : imp) s += v;
        if (s > 0) {
            for (int i = 0; i < imp.length; i++) imp[i] /= s;
        }
        return imp;
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] votes = new double[X.length][classes.length];
        for (int t = 0; t < trees.size(); t++) {
            int[] fi = treeFeatures.get(t);
            double[][] subX = selectFeatures(X, fi);
            double[] preds = trees.get(t).predict(subX);
            for (int i = 0; i < preds.length; i++) {
                for (int c = 0; c < classes.length; c++) {
                    if (preds[i] == classes[c]) { votes[i][c]++; break; }
                }
            }
        }
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            int best = 0;
            for (int c = 1; c < classes.length; c++) if (votes[i][c] > votes[i][best]) best = c;
            result[i] = classes[best];
        }
        return result;
    }

    @Override
    public double[][] predictProba(double[][] X) {
        double[][] proba = new double[X.length][classes.length];
        for (int t = 0; t < trees.size(); t++) {
            int[] fi = treeFeatures.get(t);
            double[][] subX = selectFeatures(X, fi);
            double[] preds = trees.get(t).predict(subX);
            for (int i = 0; i < preds.length; i++) {
                for (int c = 0; c < classes.length; c++) {
                    if (preds[i] == classes[c]) { proba[i][c]++; break; }
                }
            }
        }
        for (double[] row : proba) { for (int c = 0; c < row.length; c++) row[c] /= nEstimators; }
        return proba;
    }

    private int[] sampleFeatures(int d, int mf, Random rng) {
        List<Integer> all = new ArrayList<>();
        for (int i = 0; i < d; i++) all.add(i);
        Collections.shuffle(all, rng);
        int[] fi = new int[mf];
        for (int i = 0; i < mf; i++) fi[i] = all.get(i);
        return fi;
    }

    private double[][] selectFeatures(double[][] X, int[] fi) {
        double[][] r = new double[X.length][fi.length];
        for (int i = 0; i < X.length; i++)
            for (int j = 0; j < fi.length; j++) r[i][j] = X[i][fi[j]];
        return r;
    }

    public int getNEstimators() { return nEstimators; }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("max_depth", maxDepth);
        p.put("min_samples_split", minSamplesSplit); p.put("max_features", maxFeatures);
        p.put("random_state", randomState); return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_estimators")) nEstimators = ((Number) params.get("n_estimators")).intValue();
        if (params.containsKey("max_depth")) { Object v = params.get("max_depth"); maxDepth = v == null ? null : ((Number)v).intValue(); }
        if (params.containsKey("random_state")) { Object v = params.get("random_state"); randomState = v == null ? null : ((Number)v).longValue(); }
    }
}

