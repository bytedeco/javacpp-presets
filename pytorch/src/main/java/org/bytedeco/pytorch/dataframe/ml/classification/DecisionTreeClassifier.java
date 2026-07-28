package org.bytedeco.pytorch.dataframe.ml.classification;
import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;

import java.util.*;

/**
 * 决策树分类器（CART，支持 gini / entropy 准则）
 */
public class DecisionTreeClassifier extends BaseClassifier {
    private int maxDepth;
    private int minSamplesSplit;
    private String criterion; // "gini" | "entropy"
    private Long randomState;

    private Node root;
    private double[] classes;
    private int nFeatures;

    public DecisionTreeClassifier() { this(null, 2, "gini", null); }
    public DecisionTreeClassifier(Integer maxDepth, int minSamplesSplit, String criterion, Long randomState) {
        this.maxDepth = maxDepth == null ? Integer.MAX_VALUE : maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.criterion = criterion;
        this.randomState = randomState;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        TreeSet<Double> cs = new TreeSet<>();
        for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(d -> d).toArray();
        nFeatures = X[0].length;
        root = buildTree(X, y, 0);
        fitted = true;
        return this;
    }

    /**
     * Mean decrease in impurity (MDI) feature importances, normalized to sum to 1.
     * sklearn DecisionTreeClassifier.feature_importances_ compatible core.
     */
    public double[] getFeatureImportances() {
        double[] imp = new double[nFeatures];
        if (root == null) return imp;
        accumulateImportance(root, imp);
        double s = 0;
        for (double v : imp) s += v;
        if (s > 0) {
            for (int i = 0; i < imp.length; i++) imp[i] /= s;
        }
        return imp;
    }

    private void accumulateImportance(Node node, double[] imp) {
        if (node == null || node.isLeaf) return;
        double leftN = node.left == null ? 0 : node.left.nSamples;
        double rightN = node.right == null ? 0 : node.right.nSamples;
        double total = node.nSamples;
        if (total <= 0 || node.featureIndex < 0 || node.featureIndex >= imp.length) {
            if (node.left != null) accumulateImportance(node.left, imp);
            if (node.right != null) accumulateImportance(node.right, imp);
            return;
        }
        double leftImp = node.left == null ? 0 : node.left.impurity;
        double rightImp = node.right == null ? 0 : node.right.impurity;
        double decrease = node.impurity - (leftN / total) * leftImp - (rightN / total) * rightImp;
        if (decrease > 0) imp[node.featureIndex] += decrease * total;
        if (node.left != null) accumulateImportance(node.left, imp);
        if (node.right != null) accumulateImportance(node.right, imp);
    }

    @Override
    public double[] predict(double[][] X) {
        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) preds[i] = predict(root, X[i]);
        return preds;
    }

    private double predict(Node node, double[] x) {
        if (node.isLeaf) return node.classValue;
        if (x[node.featureIndex] <= node.threshold) return predict(node.left, x);
        return predict(node.right, x);
    }

    private Node buildTree(double[][] X, double[] y, int depth) {
        Node node = new Node();
        node.nSamples = y.length;
        node.impurity = impurity(y);
        if (y.length < minSamplesSplit || depth >= maxDepth || isPure(y)) {
            node.isLeaf = true;
            node.classValue = majorityClass(y);
            return node;
        }
        int[] best = bestSplit(X, y);
        if (best == null) {
            node.isLeaf = true;
            node.classValue = majorityClass(y);
            return node;
        }
        node.featureIndex = best[0];
        node.threshold = Double.longBitsToDouble(((long) best[1] << 32) | (best[2] & 0xFFFFFFFFL));
        List<Integer> leftIdx = new ArrayList<>(), rightIdx = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            if (X[i][node.featureIndex] <= node.threshold) leftIdx.add(i);
            else rightIdx.add(i);
        }
        node.left  = buildTree(subset(X, leftIdx),  subset(y, leftIdx),  depth + 1);
        node.right = buildTree(subset(X, rightIdx), subset(y, rightIdx), depth + 1);
        return node;
    }

    private int[] bestSplit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        double bestGain = -1; int bestFeat = -1; double bestThresh = 0;
        double parentImpurity = impurity(y);
        for (int j = 0; j < d; j++) {
            double[] vals = new double[n];
            for (int i = 0; i < n; i++) vals[i] = X[i][j];
            Arrays.sort(vals);
            Set<Double> tried = new HashSet<>();
            for (int i = 0; i < n - 1; i++) {
                double mid = (vals[i] + vals[i + 1]) / 2.0;
                if (!tried.add(mid)) continue;
                List<Integer> L = new ArrayList<>(), R = new ArrayList<>();
                for (int k = 0; k < n; k++) {
                    if (X[k][j] <= mid) L.add(k); else R.add(k);
                }
                if (L.isEmpty() || R.isEmpty()) continue;
                double gain = parentImpurity
                    - (L.size() * impurity(subset(y, L)) + R.size() * impurity(subset(y, R))) / n;
                if (gain > bestGain) { bestGain = gain; bestFeat = j; bestThresh = mid; }
            }
        }
        if (bestFeat < 0) return null;
        long bits = Double.doubleToLongBits(bestThresh);
        return new int[]{bestFeat, (int)(bits >>> 32), (int)(bits & 0xFFFFFFFFL)};
    }

    private double impurity(double[] y) {
        Map<Double, Integer> counts = new HashMap<>();
        for (double v : y) counts.merge(v, 1, Integer::sum);
        double imp = 0;
        for (int c : counts.values()) {
            double p = (double) c / y.length;
            if ("entropy".equals(criterion)) imp -= p * Math.log(p + 1e-15) / Math.log(2);
            else imp += p * (1 - p); // gini
        }
        return imp;
    }

    private boolean isPure(double[] y) {
        for (double v : y) if (v != y[0]) return false;
        return true;
    }

    private double majorityClass(double[] y) {
        Map<Double, Integer> counts = new HashMap<>();
        for (double v : y) counts.merge(v, 1, Integer::sum);
        return counts.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(0.0);
    }

    private double[][] subset(double[][] X, List<Integer> idx) {
        double[][] r = new double[idx.size()][X[0].length];
        for (int i = 0; i < idx.size(); i++) r[i] = X[idx.get(i)];
        return r;
    }

    private double[] subset(double[] y, List<Integer> idx) {
        double[] r = new double[idx.size()];
        for (int i = 0; i < idx.size(); i++) r[i] = y[idx.get(i)];
        return r;
    }

    static class Node implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        boolean isLeaf; double classValue;
        int featureIndex = -1; double threshold;
        int nSamples;
        double impurity;
        Node left, right;
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_depth", maxDepth == Integer.MAX_VALUE ? null : maxDepth);
        p.put("min_samples_split", minSamplesSplit);
        p.put("criterion", criterion);
        p.put("random_state", randomState);
        return p;
    }
    @Override
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("max_depth")) {
            Object v = params.get("max_depth");
            maxDepth = v == null ? Integer.MAX_VALUE : ((Number) v).intValue();
        }
        if (params.containsKey("min_samples_split")) minSamplesSplit = ((Number) params.get("min_samples_split")).intValue();
        if (params.containsKey("criterion")) criterion = (String) params.get("criterion");
    }
}

