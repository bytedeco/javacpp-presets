package org.bytedeco.pytorch.dataframe.ml.classification;
import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 单棵极端随机化决策树（分裂点完全随机选取）
 */
public class ExtraTreeClassifier extends BaseClassifier {
    private int maxDepth;
    private int minSamplesSplit;
    private Long randomState;
    private Node root;
    private double[] classes;
    private Random rng;

    public ExtraTreeClassifier() { this(null, 2, null); }
    public ExtraTreeClassifier(Integer maxDepth, int minSamplesSplit, Long randomState) {
        this.maxDepth = maxDepth == null ? Integer.MAX_VALUE : maxDepth;
        this.minSamplesSplit = minSamplesSplit; this.randomState = randomState;
        this.rng = randomState == null ? new Random() : new Random(randomState);
    }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        TreeSet<Double> cs = new TreeSet<>(); for (double v : y) cs.add(v);
        classes = cs.stream().mapToDouble(v -> v).toArray();
        root = buildTree(X, y, 0);
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) p[i] = predictNode(root, X[i]);
        return p;
    }

    private double predictNode(Node n, double[] x) {
        if (n.isLeaf) return n.value;
        return x[n.feat] <= n.thresh ? predictNode(n.left, x) : predictNode(n.right, x);
    }

    private Node buildTree(double[][] X, double[] y, int depth) {
        Node n = new Node();
        if (y.length < minSamplesSplit || depth >= maxDepth || isPure(y)) {
            n.isLeaf = true; n.value = majority(y); return n;
        }
        int d = X[0].length, feat = -1; double thresh = 0; double bestGain = -1;
        // Try all features, random thresholds
        for (int j = 0; j < d; j++) {
            double minV = X[0][j], maxV = X[0][j];
            for (double[] row : X) { minV = Math.min(minV, row[j]); maxV = Math.max(maxV, row[j]); }
            if (minV == maxV) continue;
            double t = minV + rng.nextDouble() * (maxV - minV);
            List<Integer> L = new ArrayList<>(), R = new ArrayList<>();
            for (int i = 0; i < X.length; i++) (X[i][j] <= t ? L : R).add(i);
            if (L.isEmpty() || R.isEmpty()) continue;
            double gain = gini(y) - (L.size() * gini(sub(y,L)) + R.size() * gini(sub(y,R))) / y.length;
            if (gain > bestGain) { bestGain = gain; feat = j; thresh = t; }
        }
        if (feat < 0) { n.isLeaf = true; n.value = majority(y); return n; }
        n.feat = feat; n.thresh = thresh;
        List<Integer> L = new ArrayList<>(), R = new ArrayList<>();
        for (int i = 0; i < X.length; i++) (X[i][feat] <= thresh ? L : R).add(i);
        n.left  = buildTree(sub2D(X,L), sub(y,L), depth+1);
        n.right = buildTree(sub2D(X,R), sub(y,R), depth+1);
        return n;
    }

    private double gini(double[] y) {
        Map<Double,Integer> c = new HashMap<>(); for (double v : y) c.merge(v,1,Integer::sum);
        double g = 1; for (int cnt : c.values()) g -= Math.pow((double)cnt/y.length, 2); return g;
    }
    private boolean isPure(double[] y) { for (double v : y) if (v != y[0]) return false; return true; }
    private double majority(double[] y) {
        Map<Double,Integer> c = new HashMap<>(); for (double v : y) c.merge(v,1,Integer::sum);
        return c.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(0.0);
    }
    private double[] sub(double[] y, List<Integer> idx) { double[] r = new double[idx.size()]; for (int i=0;i<idx.size();i++) r[i]=y[idx.get(i)]; return r; }
    private double[][] sub2D(double[][] X, List<Integer> idx) { double[][] r = new double[idx.size()][X[0].length]; for (int i=0;i<idx.size();i++) r[i]=X[idx.get(i)]; return r; }

    static class Node implements java.io.Serializable { private static final long serialVersionUID = 1L; boolean isLeaf; double value; int feat; double thresh; Node left, right; }

    @Override
    public Map<String, Object> getParams() {
        return new LinkedHashMap<>(Map.of("max_depth", maxDepth, "random_state", randomState));
    }
    @Override public void setParams(Map<String, Object> params) {}
}

