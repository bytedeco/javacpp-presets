package org.bytedeco.pytorch.data.dataframe.ml.model_selection;

import java.util.*;

/**
 * K 折交叉验证（对应 sklearn.model_selection.KFold）
 *
 * <pre>
 * KFold kf = new KFold(5, true, 42L);
 * for (KFold.Split s : kf.split(X, y)) {
 *     double[][] Xtr = s.trainX(X); double[] ytr = s.trainY(y);
 *     double[][] Xte = s.testX(X);  double[] yte = s.testY(y);
 * }
 * </pre>
 */
public class KFold {
    private final int nSplits;
    private final boolean shuffle;
    private final Long randomState;

    public KFold(int nSplits, boolean shuffle, Long randomState) {
        this.nSplits = nSplits; this.shuffle = shuffle; this.randomState = randomState;
    }
    public KFold(int nSplits) { this(nSplits, false, null); }

    public List<Split> split(double[][] X, double[] y) {
        int n = X.length;
        int[] idx = new int[n]; for (int i = 0; i < n; i++) idx[i] = i;
        if (shuffle) {
            Random rng = randomState == null ? new Random() : new Random(randomState);
            for (int i = n-1; i > 0; i--) { int j = rng.nextInt(i+1); int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; }
        }
        List<Split> splits = new ArrayList<>();
        int foldSize = n / nSplits;
        for (int k = 0; k < nSplits; k++) {
            int start = k * foldSize;
            int end = (k == nSplits - 1) ? n : start + foldSize;
            int[] test = Arrays.copyOfRange(idx, start, end);
            Set<Integer> testSet = new HashSet<>(); for (int i : test) testSet.add(i);
            int[] train = new int[n - test.length]; int t = 0;
            for (int i : idx) if (!testSet.contains(i)) train[t++] = i;
            splits.add(new Split(train, test));
        }
        return splits;
    }

    public int getNSplits() { return nSplits; }

    public static class Split {
        public final int[] trainIndices, testIndices;
        public Split(int[] train, int[] test) { this.trainIndices = train; this.testIndices = test; }

        public double[][] trainX(double[][] X) { return gather(X, trainIndices); }
        public double[]   trainY(double[] y)    { return gather(y, trainIndices); }
        public double[][] testX(double[][] X)   { return gather(X, testIndices); }
        public double[]   testY(double[] y)     { return gather(y, testIndices); }

        private double[][] gather(double[][] X, int[] idx) {
            double[][] r = new double[idx.length][X[0].length];
            for (int i = 0; i < idx.length; i++) r[i] = X[idx[i]]; return r;
        }
        private double[] gather(double[] y, int[] idx) {
            double[] r = new double[idx.length];
            for (int i = 0; i < idx.length; i++) r[i] = y[idx[i]]; return r;
        }
    }
}

