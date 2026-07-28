package org.bytedeco.pytorch.dataframe.ml.anomaly;

import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import java.util.*;

/**
 * 局部异常因子（LOF - Local Outlier Factor）
 */
public class LocalOutlierFactor extends BaseClassifier {
    private int nNeighbors; private double contamination;
    private double[][] trainX; private double threshold;

    public LocalOutlierFactor() { this(20, 0.1); }
    public LocalOutlierFactor(int k, double contamination) { this.nNeighbors = k; this.contamination = contamination; }

    @Override
    public BaseClassifier fit(double[][] X, double[] y) {
        trainX = X;
        double[] lofs = computeLOF(X, X);
        Arrays.sort(lofs.clone()); // compute threshold
        double[] sorted = lofs.clone(); Arrays.sort(sorted);
        int tIdx = (int)((1 - contamination) * X.length); tIdx = Math.max(0, Math.min(tIdx, X.length-1));
        threshold = sorted[tIdx];
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] lofs = computeLOF(X, trainX);
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) result[i] = lofs[i] <= threshold ? 1.0 : -1.0;
        return result;
    }

    private double[] computeLOF(double[][] queries, double[][] refs) {
        int n = queries.length, m = refs.length;
        double[] lof = new double[n];
        int k = Math.min(nNeighbors, m);
        for (int i = 0; i < n; i++) {
            int[] knn = kNearest(queries[i], refs, k);
            double reachDist = 0;
            for (int nb : knn) {
                double dNb = dist(queries[i], refs[nb]);
                double kDistNb = kDistance(refs[nb], refs, k);
                reachDist += Math.max(dNb, kDistNb);
            }
            double lrd = k / Math.max(reachDist, 1e-10);
            double sumNbLrd = 0;
            for (int nb : knn) {
                int[] nbKnn = kNearest(refs[nb], refs, k);
                double nbReach = 0;
                for (int nb2 : nbKnn) {
                    nbReach += Math.max(dist(refs[nb], refs[nb2]), kDistance(refs[nb2], refs, k));
                }
                sumNbLrd += k / Math.max(nbReach, 1e-10);
            }
            lof[i] = sumNbLrd / (k * lrd + 1e-10);
        }
        return lof;
    }

    private int[] kNearest(double[] x, double[][] refs, int k) {
        double[] dists = new double[refs.length]; for (int i = 0; i < refs.length; i++) dists[i] = dist(x, refs[i]);
        Integer[] idx = new Integer[refs.length]; for (int i = 0; i < refs.length; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> dists[i]));
        int[] res = new int[k]; for (int i = 0; i < k; i++) res[i] = idx[Math.min(i, refs.length-1)]; return res;
    }

    private double kDistance(double[] x, double[][] refs, int k) {
        int[] knn = kNearest(x, refs, k);
        return dist(x, refs[knn[knn.length-1]]);
    }

    private double dist(double[] a, double[] b) {
        double s = 0; for (int i = 0; i < a.length; i++) s += (a[i]-b[i])*(a[i]-b[i]); return Math.sqrt(s);
    }

    @Override
    public Map<String, Object> getParams() {
        return new LinkedHashMap<>(Map.of("n_neighbors", nNeighbors, "contamination", contamination));
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_neighbors")) nNeighbors = ((Number) params.get("n_neighbors")).intValue();
    }
}

