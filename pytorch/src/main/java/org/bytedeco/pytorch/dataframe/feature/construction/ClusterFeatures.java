package org.bytedeco.pytorch.dataframe.feature.construction;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;
import org.bytedeco.pytorch.dataframe.ml.cluster.KMeans;

import java.util.ArrayList;
import java.util.Locale;

/**
 * Derive cluster features via KMeans (sklearn-style unsupervised feature construction).
 * Modes: {@code LABEL} (cluster id), {@code DISTANCE} (distance to each center), {@code BOTH}.
 */
public class ClusterFeatures extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    public enum Mode { LABEL, DISTANCE, BOTH }

    private int nClusters = 8;
    private Mode mode = Mode.BOTH;
    private long randomState = 42L;
    private int maxIter = 100;
    private transient KMeans kmeans;
    private double[][] centers;

    public ClusterFeatures(int nClusters, String... columns) {
        super(columns);
        this.nClusters = Math.max(2, nClusters);
    }

    public ClusterFeatures setMode(Mode mode) {
        this.mode = mode == null ? Mode.BOTH : mode;
        return this;
    }

    public ClusterFeatures setMode(String mode) {
        if (mode == null) return this;
        String m = mode.toLowerCase(Locale.ROOT);
        this.mode = switch (m) {
            case "label", "labels", "predict" -> Mode.LABEL;
            case "distance", "distances", "transform" -> Mode.DISTANCE;
            default -> Mode.BOTH;
        };
        return this;
    }

    public ClusterFeatures setRandomState(long seed) {
        this.randomState = seed;
        return this;
    }

    public ClusterFeatures setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        double[][] mat = FeatureMatrices.fromDf(X, columns.toArray(new String[0]));
        // replace NaN with 0 for clustering
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                if (Double.isNaN(mat[i][j])) mat[i][j] = 0.0;
            }
        }
        kmeans = new KMeans(nClusters, maxIter, randomState);
        kmeans.fit(mat);
        centers = kmeans.getClusterCenters();
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        double[][] mat = FeatureMatrices.fromDf(X, columns.toArray(new String[0]));
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                if (Double.isNaN(mat[i][j])) mat[i][j] = 0.0;
            }
        }
        DataFrame result = X.copy();
        int n = result.rowCount();

        if (mode == Mode.LABEL || mode == Mode.BOTH) {
            int[] labels = kmeans.predict(mat);
            String name = unique(result, "cluster_label");
            result.addColumn(name, Column.DType.INT32);
            Column c = result.column(name);
            while (c.size() < n) c.add(null);
            for (int i = 0; i < n; i++) c.set(i, labels[i]);
        }

        if (mode == Mode.DISTANCE || mode == Mode.BOTH) {
            int k = centers.length;
            double[][] dists = new double[n][k];
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < k; c++) {
                    dists[i][c] = Math.sqrt(sqEuclidean(mat[i], centers[c]));
                }
            }
            String[] names = new String[k];
            for (int c = 0; c < k; c++) names[c] = "cluster_dist_" + c;
            result = FeatureMatrices.appendColumns(result, names, dists);
        }
        return result;
    }

    private static double sqEuclidean(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    private static String unique(DataFrame df, String base) {
        if (!df.hasColumn(base)) return base;
        int k = 1;
        String n = base + "_" + k;
        while (df.hasColumn(n)) n = base + "_" + (++k);
        return n;
    }

    public double[][] getClusterCenters() { return centers; }
    public int getNClusters() { return nClusters; }
    public Mode getMode() { return mode; }
}
