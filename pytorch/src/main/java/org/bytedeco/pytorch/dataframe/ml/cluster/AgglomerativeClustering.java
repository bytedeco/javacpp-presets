package org.bytedeco.pytorch.dataframe.ml.cluster;

import org.bytedeco.pytorch.dataframe.ml.math.Distance;
import java.util.*;

/**
 * Simple agglomerative clustering (single linkage) producing k clusters.
 */
public class AgglomerativeClustering implements Clusterer {
    private int nClusters = 2;
    private int[] labels = null;

    public AgglomerativeClustering(int nClusters) { this.nClusters = nClusters; }

    @Override
    public void fit(double[][] X) {
        int n = X.length;
        // start with each point its own cluster
        List<Set<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < n; i++) { Set<Integer> s = new HashSet<>(); s.add(i); clusters.add(s); }

        while (clusters.size() > nClusters) {
            double best = Double.POSITIVE_INFINITY; int a=-1, b=-1;
            for (int i = 0; i < clusters.size(); i++) for (int j = i+1; j < clusters.size(); j++) {
                double dist = clusterDistance(clusters.get(i), clusters.get(j), X);
                if (dist < best) { best = dist; a = i; b = j; }
            }
            // merge b into a
            clusters.get(a).addAll(clusters.get(b)); clusters.remove(b);
        }

        labels = new int[n];
        for (int cid = 0; cid < clusters.size(); cid++) for (int idx : clusters.get(cid)) labels[idx] = cid;
    }

    private double clusterDistance(Set<Integer> A, Set<Integer> B, double[][] X) {
        double best = Double.POSITIVE_INFINITY;
        for (int i : A) for (int j : B) best = Math.min(best, Distance.euclidean(X[i], X[j]));
        return best;
    }

    @Override public int[] predict(double[][] X) { throw new UnsupportedOperationException("Agglomerative predict not supported"); }
    @Override public Map<String,Object> getParams() { Map<String,Object> m = new LinkedHashMap<>(); m.put("n_clusters", nClusters); return m; }
    @Override public void setParams(Map<String,Object> p) { if (p.containsKey("n_clusters")) nClusters = ((Number)p.get("n_clusters")).intValue(); }
    @Override public int[] getLabels() { return labels; }
    @Override public double[][] getClusterCenters() { return null; }
    @Override public int getNClusters() { return labels == null ? nClusters : Arrays.stream(labels).max().orElse(nClusters-1)+1; }
}

