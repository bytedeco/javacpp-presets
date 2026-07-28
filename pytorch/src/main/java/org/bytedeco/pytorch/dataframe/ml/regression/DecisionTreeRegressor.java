package org.bytedeco.pytorch.dataframe.ml.regression;
import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/**
 * 决策树回归器（CART，MSE / MAE 准则）
 */
public class DecisionTreeRegressor extends BaseRegressor {
    private int maxDepth;
    private int minSamplesSplit;
    private String criterion; // "mse" | "mae"
    private Long randomState;
    private Node root;

    public DecisionTreeRegressor() { this(null, 2, "mse", null); }
    public DecisionTreeRegressor(Integer maxDepth, int minSamplesSplit, String criterion, Long randomState) {
        this.maxDepth = maxDepth == null ? Integer.MAX_VALUE : maxDepth;
        this.minSamplesSplit = minSamplesSplit; this.criterion = criterion; this.randomState = randomState;
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        root = buildTree(X, y, 0); fitted = true; return this;
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
        if (y.length < minSamplesSplit || depth >= maxDepth) {
            n.isLeaf = true; n.value = mean(y); return n;
        }
        int[] best = bestSplit(X, y);
        if (best == null) { n.isLeaf = true; n.value = mean(y); return n; }
        n.feat = best[0];
        n.thresh = Double.longBitsToDouble(((long) best[1] << 32) | (best[2] & 0xFFFFFFFFL));
        List<Integer> L = new ArrayList<>(), R = new ArrayList<>();
        for (int i = 0; i < X.length; i++) (X[i][n.feat] <= n.thresh ? L : R).add(i);
        if (L.isEmpty() || R.isEmpty()) { n.isLeaf = true; n.value = mean(y); return n; }
        n.left  = buildTree(sub(X,L), sub(y,L), depth+1);
        n.right = buildTree(sub(X,R), sub(y,R), depth+1);
        return n;
    }

    private int[] bestSplit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        double bestGain = -1; int bestFeat = -1; double bestThresh = 0;
        double parentImp = impurity(y);
        for (int j = 0; j < d; j++) {
            double[] vals = new double[n]; for (int i = 0; i < n; i++) vals[i] = X[i][j];
            Arrays.sort(vals);
            Set<Double> tried = new HashSet<>();
            for (int i = 0; i < n-1; i++) {
                double mid = (vals[i] + vals[i+1]) / 2.0;
                if (!tried.add(mid)) continue;
                List<Integer> L = new ArrayList<>(), R = new ArrayList<>();
                for (int k = 0; k < n; k++) (X[k][j] <= mid ? L : R).add(k);
                if (L.isEmpty() || R.isEmpty()) continue;
                double gain = parentImp - (L.size() * impurity(sub(y,L)) + R.size() * impurity(sub(y,R))) / n;
                if (gain > bestGain) { bestGain = gain; bestFeat = j; bestThresh = mid; }
            }
        }
        if (bestFeat < 0) return null;
        long bits = Double.doubleToLongBits(bestThresh);
        return new int[]{bestFeat, (int)(bits >>> 32), (int)(bits & 0xFFFFFFFFL)};
    }

    private double impurity(double[] y) {
        double m = mean(y), s = 0;
        if ("mae".equals(criterion)) { for (double v : y) s += Math.abs(v - m); }
        else { for (double v : y) s += (v-m)*(v-m); }
        return s / y.length;
    }
    private double mean(double[] y) { double s = 0; for (double v : y) s += v; return s / y.length; }
    private double[] sub(double[] y, List<Integer> idx) { double[] r = new double[idx.size()]; for (int i=0;i<idx.size();i++) r[i]=y[idx.get(i)]; return r; }
    private double[][] sub(double[][] X, List<Integer> idx) { double[][] r = new double[idx.size()][X[0].length]; for (int i=0;i<idx.size();i++) r[i]=X[idx.get(i)]; return r; }

    static class Node implements java.io.Serializable { private static final long serialVersionUID = 1L; boolean isLeaf; double value; int feat; double thresh; Node left, right; }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_depth", maxDepth == Integer.MAX_VALUE ? null : maxDepth);
        p.put("min_samples_split", minSamplesSplit); p.put("criterion", criterion); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("max_depth")) { Object v = params.get("max_depth"); maxDepth = v==null ? Integer.MAX_VALUE : ((Number)v).intValue(); }
        if (params.containsKey("criterion")) criterion = (String) params.get("criterion");
    }
}

