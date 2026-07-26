package org.bytedeco.pytorch.data.dataframe.ml.cluster;

/**
 * Lightweight OPTICS placeholder: currently not a full-featured OPTICS.
 * For now this class delegates to DBSCAN with a heuristic eps.
 */
public class OPTICS implements Clusterer {
    private DBSCAN db = new DBSCAN(0.5,5);
    @Override public void fit(double[][] X) { db.fit(X); }
    @Override public int[] predict(double[][] X) { throw new UnsupportedOperationException("OPTICS predict not supported"); }
    @Override public java.util.Map<String,Object> getParams() { return db.getParams(); }
    @Override public void setParams(java.util.Map<String,Object> p) { db.setParams(p); }
    @Override public int[] getLabels() { return db.getLabels(); }
    @Override public double[][] getClusterCenters() { return db.getClusterCenters(); }
    @Override public int getNClusters() { return db.getNClusters(); }
}

