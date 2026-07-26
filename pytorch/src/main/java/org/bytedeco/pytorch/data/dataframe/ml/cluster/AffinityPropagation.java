package org.bytedeco.pytorch.data.dataframe.ml.cluster;

/**
 * Placeholder AffinityPropagation: not a true AP implementation.
 * Delegates to KMeans with inferred k for now.
 */
public class AffinityPropagation implements Clusterer {
    private KMeans km;
    public AffinityPropagation(int k) { km = new KMeans(k); }
    @Override public void fit(double[][] X) { km.fit(X); }
    @Override public int[] predict(double[][] X) { return km.predict(X); }
    @Override public java.util.Map<String,Object> getParams() { return km.getParams(); }
    @Override public void setParams(java.util.Map<String,Object> p) { km.setParams(p); }
    @Override public int[] getLabels() { return km.getLabels(); }
    @Override public double[][] getClusterCenters() { return km.getClusterCenters(); }
    @Override public int getNClusters() { return km.getNClusters(); }
}

