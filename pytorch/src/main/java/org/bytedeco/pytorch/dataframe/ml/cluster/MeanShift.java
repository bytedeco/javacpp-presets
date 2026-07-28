package org.bytedeco.pytorch.dataframe.ml.cluster;

import org.bytedeco.pytorch.dataframe.ml.math.Distance;
import java.util.*;

public class MeanShift implements Clusterer {
    private double bandwidth = 1.0;
    private int[] labels = null;
    private double[][] modes = null;

    public MeanShift() {}
    public MeanShift(double bandwidth) { this.bandwidth = bandwidth; }

    @Override
    public void fit(double[][] X) {
        int n = X.length, d = X[0].length;
        double[][] points = new double[n][d]; for (int i=0;i<n;i++) points[i]=X[i].clone();
        double bw2 = bandwidth * bandwidth;
        for (int i = 0; i < n; i++) {
            double[] cur = points[i];
            for (int it = 0; it < 100; it++) {
                double[] num = new double[d]; double den = 0.0;
                for (int j = 0; j < n; j++) {
                    double dist2 = Distance.squaredEuclidean(cur, X[j]);
                    double w = Math.exp(-dist2 / (2*bw2));
                    den += w; for (int k=0;k<d;k++) num[k] += w * X[j][k];
                }
                for (int k=0;k<d;k++) num[k] /= den;
                double move = Distance.euclidean(cur, num);
                cur = num;
                if (move < 1e-3) break;
            }
            points[i] = cur;
        }
        // collapse near-duplicates to modes
        List<double[]> uniq = new ArrayList<>();
        for (double[] p : points) {
            boolean found=false; for (double[] q:uniq) if (Distance.euclidean(p,q) < 1e-2) { found=true; break; }
            if(!found) uniq.add(p);
        }
        modes = uniq.toArray(new double[0][]);
        labels = new int[n]; for (int i=0;i<n;i++) { int bi=0; double best=Double.POSITIVE_INFINITY; for (int m=0;m<modes.length;m++){ double dist=Distance.euclidean(points[i], modes[m]); if(dist<best){best=dist;bi=m;} } labels[i]=bi; }
    }

    @Override public int[] predict(double[][] X) { throw new UnsupportedOperationException("MeanShift predict not supported"); }
    @Override public java.util.Map<String,Object> getParams() { java.util.Map<String,Object> m=new java.util.LinkedHashMap<>(); m.put("bandwidth",bandwidth); return m; }
    @Override public void setParams(java.util.Map<String,Object> p) { if(p.containsKey("bandwidth")) bandwidth=((Number)p.get("bandwidth")).doubleValue(); }
    @Override public int[] getLabels() { return labels; }
    @Override public double[][] getClusterCenters() { return modes; }
    @Override public int getNClusters() { return modes==null?0:modes.length; }
}

