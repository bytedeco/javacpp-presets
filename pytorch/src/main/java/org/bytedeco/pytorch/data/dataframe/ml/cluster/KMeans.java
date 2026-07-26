package org.bytedeco.pytorch.data.dataframe.ml.cluster;

import java.util.*;

/**
 * Simple KMeans implementation with KMeans++ initialization.
 */
public class KMeans implements Clusterer {
    private int nClusters = 8;
    private int maxIter = 300;
    private long randomState = 0L;
    private double tol = 1e-4;

    private double[][] centers = null;
    private int[] labels = null;

    public KMeans(int nClusters) { this.nClusters = nClusters; }
    public KMeans(int nClusters, int maxIter, long randomState) { this.nClusters = nClusters; this.maxIter = maxIter; this.randomState = randomState; }

    @Override
    public void fit(double[][] X) {
        int n = X.length; int d = X[0].length;
        centers = kmeansPlusPlusInit(X, nClusters, new Random(randomState));
        labels = new int[n];
        double prevInertia = Double.POSITIVE_INFINITY;
        for (int iter = 0; iter < maxIter; iter++) {
            // assign step
            double inertia = 0.0;
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY; int bi = -1;
                for (int k = 0; k < nClusters; k++) {
                    double dist = Distance.squaredEuclidean(X[i], centers[k]);
                    if (dist < best) { best = dist; bi = k; }
                }
                labels[i] = bi; inertia += best;
            }
            // update step
            double[][] newC = new double[nClusters][d];
            int[] counts = new int[nClusters];
            for (int i = 0; i < n; i++) {
                int c = labels[i]; counts[c]++;
                for (int j = 0; j < d; j++) newC[c][j] += X[i][j];
            }
            for (int k = 0; k < nClusters; k++) {
                if (counts[k] > 0) {
                    for (int j = 0; j < d; j++) newC[k][j] /= counts[k];
                } else {
                    // reinitialize empty cluster randomly
                    newC[k] = X[new Random(randomState + iter + k).nextInt(n)].clone();
                }
            }
            // check tol
            double maxMove = 0.0;
            for (int k = 0; k < nClusters; k++) {
                maxMove = Math.max(maxMove, Math.sqrt(Distance.squaredEuclidean(centers[k], newC[k])));
            }
            centers = newC; if (Math.abs(prevInertia - inertia) < tol || maxMove < tol) break; prevInertia = inertia;
        }
    }

    @Override
    public int[] predict(double[][] X) {
        int n = X.length; int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            double best = Double.POSITIVE_INFINITY; int bi = -1;
            for (int k = 0; k < centers.length; k++) {
                double dist = Distance.squaredEuclidean(X[i], centers[k]);
                if (dist < best) { best = dist; bi = k; }
            }
            out[i] = bi;
        }
        return out;
    }

    private double[][] kmeansPlusPlusInit(double[][] X, int k, Random rng) {
        int n = X.length; int d = X[0].length;
        double[][] c = new double[k][d];
        int first = rng.nextInt(n); c[0] = X[first].clone();
        double[] closest = new double[n]; Arrays.fill(closest, Double.POSITIVE_INFINITY);
        for (int i = 1; i < k; i++) {
            double total = 0.0;
            for (int j = 0; j < n; j++) {
                double dist = Distance.squaredEuclidean(X[j], c[i-1]);
                if (dist < closest[j]) closest[j] = dist;
                total += closest[j];
            }
            double r = rng.nextDouble() * total; double cum = 0.0; int idx = 0;
            for (int j = 0; j < n; j++) { cum += closest[j]; if (cum >= r) { idx = j; break; } }
            c[i] = X[idx].clone();
        }
        return c;
    }

    @Override public Map<String,Object> getParams() { Map<String,Object> m = new LinkedHashMap<>(); m.put("n_clusters", nClusters); m.put("max_iter", maxIter); return m; }
    @Override public void setParams(Map<String,Object> p) { if (p.containsKey("n_clusters")) nClusters = ((Number)p.get("n_clusters")).intValue(); }
    @Override public int[] getLabels() { return labels; }
    @Override public double[][] getClusterCenters() { return centers; }
    @Override public int getNClusters() { return centers == null ? nClusters : centers.length; }
}

