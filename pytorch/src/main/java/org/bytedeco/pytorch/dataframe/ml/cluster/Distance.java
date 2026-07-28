package org.bytedeco.pytorch.dataframe.ml.cluster;

public class Distance {
    public static double euclidean(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i]; s += d * d;
        }
        return Math.sqrt(s);
    }

    public static double squaredEuclidean(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i]; s += d * d;
        }
        return s;
    }
}

