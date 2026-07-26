package org.bytedeco.pytorch.data.dataframe.ml.anomaly;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 孤立森林（Isolation Forest）异常检测
 * 返回 predict: -1 (异常) 或 +1 (正常)
 */
public class IsolationForest extends BaseClassifier {
    private int nEstimators; private double contamination; private Long randomState;
    private int maxSamples;
    private List<ITree> trees = new ArrayList<>();
    private double threshold;
    // keep a reference to training data so scoreAll can be called with null
    private double[][] trainX = null;

    public IsolationForest() { this(100, 0.1, 256, null); }
    public IsolationForest(int n, double contamination, int maxSamples, Long rs) {
        this.nEstimators = n; this.contamination = contamination;
        this.maxSamples = maxSamples; this.randomState = rs;
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        if (X == null) throw new IllegalArgumentException("X (training data) must not be null");
        this.trainX = X; // remember training data
        int n = X.length;
        Random rng = randomState == null ? new Random() : new Random(randomState);
        int sampleSize = Math.min(maxSamples, n);
        trees.clear();
        for (int t = 0; t < nEstimators; t++) {
            int[] idx = sampleWithoutReplacement(n, sampleSize, rng);
            double[][] sample = new double[sampleSize][X[0].length];
            for (int i = 0; i < sampleSize; i++) sample[i] = X[idx[i]];
            trees.add(new ITree(sample, 0, (int)(Math.ceil(Math.log(sampleSize) / Math.log(2))), rng));
        }
        // Compute threshold from training scores
        double[] scores = scoreAll(X);
        Arrays.sort(scores);
        int cutIdx = (int)((1 - contamination) * n);
        threshold = scores[Math.max(0, Math.min(cutIdx, n-1))];
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] scores = scoreAll(X);
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) result[i] = scores[i] >= threshold ? 1.0 : -1.0;
        return result;
    }

    public double[] scoreAll(double[][] X) {
        // allow callers to pass null: fall back to training data if available
        if (X == null) {
            if (this.trainX == null) throw new IllegalArgumentException("Both X and internal training data are null");
            X = this.trainX;
        }
        double[] scores = new double[X.length];
        int sampleSize = Math.min(maxSamples, trees.isEmpty() ? 1 : (trees.get(0) == null ? 1 : trees.get(0).sampleSize));
        for (ITree tree : trees) {
            for (int i = 0; i < X.length; i++) scores[i] += tree.pathLength(X[i]);
        }
        double c = cFactor(sampleSize);
        for (int i = 0; i < X.length; i++) scores[i] = Math.pow(2, -scores[i] / (nEstimators * c));
        return scores;
    }

    private double cFactor(int n) {
        if (n <= 1) return 1;
        return 2 * (Math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n;
    }

    private int[] sampleWithoutReplacement(int n, int k, Random rng) {
        List<Integer> all = new ArrayList<>(); for (int i = 0; i < n; i++) all.add(i);
        Collections.shuffle(all, rng); int[] res = new int[k]; for (int i = 0; i < k; i++) res[i] = all.get(i); return res;
    }

    static class ITree {
        boolean isLeaf; int feat; double thresh; ITree left, right;
        int size; double[][] minMax; int sampleSize;
        ITree(double[][] X, int depth, int maxDepth, Random rng) {
            sampleSize = X.length;
            if (X.length <= 1 || depth >= maxDepth) { isLeaf = true; size = X.length; return; }
            int d = X[0].length;
            // Pick random feature
            feat = rng.nextInt(d);
            double min = X[0][feat], max = X[0][feat];
            for (double[] row : X) { min = Math.min(min, row[feat]); max = Math.max(max, row[feat]); }
            if (min == max) { isLeaf = true; size = X.length; return; }
            thresh = min + rng.nextDouble() * (max - min);
            List<double[]> L = new ArrayList<>(), R = new ArrayList<>();
            for (double[] row : X) (row[feat] <= thresh ? L : R).add(row);
            left  = new ITree(L.toArray(new double[0][]), depth+1, maxDepth, rng);
            right = new ITree(R.toArray(new double[0][]), depth+1, maxDepth, rng);
        }
        double pathLength(double[] x) {
            if (isLeaf) return size <= 1 ? 0 : cFactor(size);
            if (x[feat] <= thresh) return 1 + left.pathLength(x);
            return 1 + right.pathLength(x);
        }
        double cFactor(int n) { return n <= 1 ? 1 : 2*(Math.log(n-1)+0.5772156649) - 2.0*(n-1)/n; }
    }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("n_estimators", nEstimators); p.put("contamination", contamination); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("contamination")) contamination = ((Number) params.get("contamination")).doubleValue();
    }
}
