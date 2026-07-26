package org.bytedeco.pytorch.data.dataframe.ml.math;

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

    public static double manhattan(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }

    public static double cosine(double[] a, double[] b) {
        double da = 0.0, db = 0.0, dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i]; da += a[i] * a[i]; db += b[i] * b[i];
        }
        if (da == 0 || db == 0) return 1.0; // treat zero-vector as maximally distant
        return 1.0 - (dot / (Math.sqrt(da) * Math.sqrt(db)));
    }
}

