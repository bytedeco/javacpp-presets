package org.bytedeco.pytorch.dataframe.ml.cluster;

import org.bytedeco.pytorch.dataframe.ml.math.Distance;
import java.util.*;

public class SpectralClustering implements Clusterer {
    private int nClusters = 2;
    private int[] labels = null;
    private double[][] centers = null;

    public SpectralClustering(int nClusters) { this.nClusters = nClusters; }

    @Override
    public void fit(double[][] X) {
        int n = X.length;
        // build affinity matrix (rbf)
        double sigma = 1.0;
        double[][] A = new double[n][n];
        for (int i=0;i<n;i++) for (int j=0;j<n;j++) { double d = Distance.euclidean(X[i], X[j]); A[i][j] = Math.exp(-d*d/(2*sigma*sigma)); }
        // degree and normalized Laplacian approx: return first k eigenvectors via power iteration on (D^-1 A)
        double[][] U = new double[n][n];
        for (int k = 0; k < nClusters; k++) {
            double[] v = new double[n]; Random rng = new Random(42 + k);
            for (int i=0;i<n;i++) v[i]=rng.nextDouble();
            for (int it=0; it<50; it++) {
                double[] nv = new double[n];
                for (int i=0;i<n;i++) for (int j=0;j<n;j++) nv[i] += A[i][j]*v[j];
                double norm=0; for (int i=0;i<n;i++) norm+=nv[i]*nv[i]; norm=Math.sqrt(norm); if(norm==0) break; for (int i=0;i<n;i++) v[i]=nv[i]/norm;
            }
            for (int i=0;i<n;i++) U[i][k]=v[i];
        }
        // run KMeans on rows of U
        double[][] rows = new double[n][nClusters]; for (int i=0;i<n;i++) for (int j=0;j<nClusters;j++) rows[i][j]=U[i][j];
        KMeans km = new KMeans(nClusters);
        km.fit(rows);
        labels = km.getLabels(); centers = km.getClusterCenters();
    }

    @Override public int[] predict(double[][] X) { throw new UnsupportedOperationException("SpectralClustering predict not supported"); }
    @Override public Map<String,Object> getParams() { Map<String,Object> m=new LinkedHashMap<>(); m.put("n_clusters", nClusters); return m; }
    @Override public void setParams(Map<String,Object> p) { if(p.containsKey("n_clusters")) nClusters=((Number)p.get("n_clusters")).intValue(); }
    @Override public int[] getLabels() { return labels; }
    @Override public double[][] getClusterCenters() { return centers; }
    @Override public int getNClusters() { return centers==null? nClusters : centers.length; }
}

