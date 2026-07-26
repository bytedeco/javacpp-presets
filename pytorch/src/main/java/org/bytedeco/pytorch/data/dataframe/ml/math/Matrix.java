package org.bytedeco.pytorch.data.dataframe.ml.math;

public class Matrix {
    public static double[][] copy(double[][] A) {
        double[][] B = new double[A.length][];
        for (int i = 0; i < A.length; i++) B[i] = A[i].clone();
        return B;
    }

    public static double[] centroid(double[][] X) {
        int n = X.length, d = X[0].length;
        double[] c = new double[d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) c[j] += X[i][j];
        for (int j = 0; j < d; j++) c[j] /= n;
        return c;
    }
}

