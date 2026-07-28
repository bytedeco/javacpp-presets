package org.bytedeco.pytorch.dataframe.ml.cluster;

public class Birch implements Clusterer {
    private int nClusters = 3;
    private int[] labels = null;
    private double[][] centers = null;

    public Birch() {}
    public Birch(int nClusters) { this.nClusters = nClusters; }

    @Override
    public void fit(double[][] X) {
        // simple fallback: run MiniBatchKMeans to produce a smaller set then KMeans to finalize
        MiniBatchKMeans mb = new MiniBatchKMeans(Math.min(50, X.length/10+1), 50, 42L);
        mb.fit(X);
        double[][] smallCenters = mb.getClusterCenters();
        KMeans km = new KMeans(nClusters);
        km.fit(smallCenters != null ? smallCenters : X);
        labels = km.predict(X);
        centers = km.getClusterCenters();
    }

    @Override public int[] predict(double[][] X) { throw new UnsupportedOperationException("Birch predict not implemented"); }
    @Override public java.util.Map<String,Object> getParams() { java.util.Map<String,Object> m=new java.util.LinkedHashMap<>(); m.put("n_clusters",nClusters); return m; }
    @Override public void setParams(java.util.Map<String,Object> p) { if(p.containsKey("n_clusters")) nClusters=((Number)p.get("n_clusters")).intValue(); }
    @Override public int[] getLabels() { return labels; }
    @Override public double[][] getClusterCenters() { return centers; }
    @Override public int getNClusters() { return centers==null? nClusters : centers.length; }
}

