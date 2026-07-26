package org.bytedeco.pytorch.data.dataframe.ml.cluster;

import org.bytedeco.pytorch.data.dataframe.ml.math.Distance;
import java.util.*;

public class DBSCAN implements Clusterer {
    private double eps = 0.5;
    private int minSamples = 5;
    private int[] labels = null;

    public DBSCAN() {}
    public DBSCAN(double eps, int minSamples) { this.eps = eps; this.minSamples = minSamples; }

    @Override
    public void fit(double[][] X) {
        int n = X.length; labels = new int[n]; Arrays.fill(labels, -2); // -2 = unvisited
        int clusterId = 0;
        for (int i = 0; i < n; i++) {
            if (labels[i] != -2) continue;
            List<Integer> neigh = regionQuery(X, i);
            if (neigh.size() < minSamples) { labels[i] = -1; continue; } // noise
            // expand
            Queue<Integer> q = new ArrayDeque<>(neigh);
            labels[i] = clusterId;
            while (!q.isEmpty()) {
                int j = q.poll();
                if (labels[j] == -1) labels[j] = clusterId;
                if (labels[j] != -2) continue;
                labels[j] = clusterId;
                List<Integer> neigh2 = regionQuery(X, j);
                if (neigh2.size() >= minSamples) q.addAll(neigh2);
            }
            clusterId++;
        }
    }

    private List<Integer> regionQuery(double[][] X, int idx) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < X.length; i++) if (Distance.euclidean(X[idx], X[i]) <= eps) out.add(i);
        return out;
    }

    @Override public int[] predict(double[][] X) { throw new UnsupportedOperationException("DBSCAN predict not supported (requires full dataset)"); }
    @Override public Map<String,Object> getParams() { Map<String,Object> m = new LinkedHashMap<>(); m.put("eps", eps); m.put("min_samples", minSamples); return m; }
    @Override public void setParams(Map<String,Object> p) { if (p.containsKey("eps")) eps = ((Number)p.get("eps")).doubleValue(); }
    @Override public int[] getLabels() { return labels; }
    @Override public double[][] getClusterCenters() { return null; }
    @Override public int getNClusters() { if (labels==null) return 0; int max=-1; for(int v:labels) if (v>max) max=v; return max+1; }
}

