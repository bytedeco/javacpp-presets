package org.bytedeco.pytorch.data.dataframe.ml.cluster;

import org.bytedeco.pytorch.data.dataframe.ml.math.RandomSampler;
import org.bytedeco.pytorch.data.dataframe.ml.math.Distance;
import java.util.*;

public class MiniBatchKMeans implements Clusterer {
    private int nClusters = 8;
    private int maxIter = 1000;
    private int batchSize = 100;
    private long randomState = 0L;

    private double[][] centers = null;
    private int[] labels = null;

    public MiniBatchKMeans(int nClusters) { this.nClusters = nClusters; }
    public MiniBatchKMeans(int nClusters, int batchSize, long randomState) { this.nClusters = nClusters; this.batchSize = batchSize; this.randomState = randomState; }

    @Override
    public void fit(double[][] X) {
        int n = X.length, d = X[0].length;
        Random rng = new Random(randomState);
        // init with kmeans++ simple version
        centers = kmeansPlusPlusInit(X, nClusters, rng);
        double[] counts = new double[nClusters];

        for (int it = 0; it < maxIter; it++) {
            int b = Math.min(batchSize, n);
            int[] idx = RandomSampler.sampleIndices(n, b, rng);
            for (int id : idx) {
                double best = Double.POSITIVE_INFINITY; int bi = -1;
                for (int k = 0; k < nClusters; k++) {
                    double dist = Distance.squaredEuclidean(X[id], centers[k]);
                    if (dist < best) { best = dist; bi = k; }
                }
                counts[bi] += 1.0;
                double eta = 1.0 / counts[bi];
                for (int j = 0; j < d; j++) centers[bi][j] = centers[bi][j] + eta * (X[id][j] - centers[bi][j]);
            }
        }
        // set labels
        labels = new int[n];
        for (int i = 0; i < n; i++) {
            double best = Double.POSITIVE_INFINITY; int bi = -1;
            for (int k = 0; k < nClusters; k++) {
                double dist = Distance.squaredEuclidean(X[i], centers[k]);
                if (dist < best) { best = dist; bi = k; }
            }
            labels[i] = bi;
        }
    }

    @Override
    public int[] predict(double[][] X) { int n = X.length; int[] out = new int[n]; for (int i=0;i<n;i++){ double best=Double.POSITIVE_INFINITY; int bi=-1; for (int k=0;k<centers.length;k++){ double dist=Distance.squaredEuclidean(X[i], centers[k]); if(dist<best){best=dist;bi=k;} } out[i]=bi; } return out; }

    @Override public Map<String,Object> getParams() { Map<String,Object> m = new LinkedHashMap<>(); m.put("n_clusters", nClusters); m.put("batch_size", batchSize); return m; }
    @Override public void setParams(Map<String,Object> p) { if (p.containsKey("n_clusters")) nClusters = ((Number)p.get("n_clusters")).intValue(); }
    @Override public int[] getLabels() { return labels; }
    @Override public double[][] getClusterCenters() { return centers; }
    @Override public int getNClusters() { return centers == null ? nClusters : centers.length; }

    // kmeans++ initializer (copied locally)
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

}

