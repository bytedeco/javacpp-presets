package org.bytedeco.pytorch.dataframe.feature.imputation;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * KNN imputer (sklearn KNNImputer-style).
 * Fills missing numeric values in-place using K nearest complete(ish) neighbors.
 * Supports {@code weights=uniform|distance}.
 */
public class KNNImputer extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int nNeighbors = 5;
    /** "uniform" | "distance" */
    private String weights = "uniform";
    /** Fallback column means from fit. */
    private final Map<String, Double> columnMeans = new HashMap<>();
    /** Training matrix snapshot for neighbor search [n_train][d]. */
    private double[][] trainMatrix;
    private boolean[][] trainMissing;

    public KNNImputer(String... columns) {
        this(5, columns);
    }

    public KNNImputer(int nNeighbors, String... columns) {
        super(columns);
        this.nNeighbors = Math.max(1, nNeighbors);
    }

    public KNNImputer setWeights(String weights) {
        this.weights = weights == null ? "uniform" : weights.toLowerCase(Locale.ROOT);
        return this;
    }

    public KNNImputer setNNeighbors(int nNeighbors) {
        this.nNeighbors = Math.max(1, nNeighbors);
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<String> cols = resolveColumns(X);
        this.columns = new ArrayList<>(cols);
        columnMeans.clear();

        int n = X.rowCount();
        int d = cols.size();
        trainMatrix = new double[n][d];
        trainMissing = new boolean[n][d];

        for (int j = 0; j < d; j++) {
            String col = cols.get(j);
            Column c = X.column(col);
            double sum = 0;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                Object v = c.get(i);
                boolean miss = FeatureMatrices.isMissing(v)
                    || (v != null && Double.isNaN(DataValues.asDouble(v)));
                // also treat pure null / NaN
                if (v == null) miss = true;
                else {
                    double dv = DataValues.asDouble(v);
                    if (Double.isNaN(dv)) miss = true;
                }
                trainMissing[i][j] = miss;
                if (miss) {
                    trainMatrix[i][j] = Double.NaN;
                } else {
                    double dv = DataValues.asDouble(v);
                    trainMatrix[i][j] = dv;
                    sum += dv;
                    cnt++;
                }
            }
            columnMeans.put(col, cnt == 0 ? 0.0 : sum / cnt);
        }

        // fill train matrix NaNs with column means so distance is defined
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                if (Double.isNaN(trainMatrix[i][j])) {
                    trainMatrix[i][j] = columnMeans.get(cols.get(j));
                }
            }
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        int n = result.rowCount();
        int d = columns.size();

        double[][] query = new double[n][d];
        boolean[][] qMiss = new boolean[n][d];
        for (int j = 0; j < d; j++) {
            Column c = X.column(columns.get(j));
            for (int i = 0; i < n; i++) {
                Object v = c.get(i);
                boolean miss = v == null || Double.isNaN(DataValues.asDouble(v));
                qMiss[i][j] = miss;
                if (miss) {
                    query[i][j] = columnMeans.getOrDefault(columns.get(j), 0.0);
                } else {
                    query[i][j] = DataValues.asDouble(v);
                }
            }
        }

        for (int i = 0; i < n; i++) {
            boolean anyMiss = false;
            for (int j = 0; j < d; j++) if (qMiss[i][j]) { anyMiss = true; break; }
            if (!anyMiss) continue;

            NeighborResult nr = findNeighbors(query[i], nNeighbors);
            for (int j = 0; j < d; j++) {
                if (!qMiss[i][j]) continue;
                double fill = imputeDistance(nr.indices, nr.distances, j);
                result.column(columns.get(j)).set(i, fill);
            }
        }
        return result;
    }

    private static final class NeighborResult {
        final int[] indices;
        final double[] distances;
        NeighborResult(int[] indices, double[] distances) {
            this.indices = indices;
            this.distances = distances;
        }
    }

    private NeighborResult findNeighbors(double[] row, int k) {
        int nTrain = trainMatrix.length;
        PriorityQueue<double[]> heap = new PriorityQueue<>((a, b) -> Double.compare(b[0], a[0]));
        for (int t = 0; t < nTrain; t++) {
            double dist = euclidean(row, trainMatrix[t]);
            if (heap.size() < k) {
                heap.offer(new double[]{dist, t});
            } else if (dist < heap.peek()[0]) {
                heap.poll();
                heap.offer(new double[]{dist, t});
            }
        }
        List<double[]> pairs = new ArrayList<>(heap);
        pairs.sort((a, b) -> Double.compare(a[0], b[0]));
        int[] idx = new int[pairs.size()];
        double[] dists = new double[pairs.size()];
        for (int p = 0; p < pairs.size(); p++) {
            dists[p] = pairs.get(p)[0];
            idx[p] = (int) pairs.get(p)[1];
        }
        return new NeighborResult(idx, dists);
    }

    /** Uniform or distance-weighted impute with explicit distances. */
    private double imputeDistance(int[] nn, double[] dists, int featureIdx) {
        if (nn.length == 0) {
            return columnMeans.getOrDefault(columns.get(featureIdx), 0.0);
        }
        if (!"distance".equals(weights)) {
            double sum = 0;
            for (int t : nn) sum += trainMatrix[t][featureIdx];
            return sum / nn.length;
        }
        double num = 0, den = 0;
        for (int r = 0; r < nn.length; r++) {
            double dist = dists[r];
            double w = dist < 1e-12 ? 1e12 : 1.0 / dist;
            num += w * trainMatrix[nn[r]][featureIdx];
            den += w;
        }
        return den == 0 ? columnMeans.getOrDefault(columns.get(featureIdx), 0.0) : num / den;
    }

    @Override
    public DataFrame fitTransform(DataFrame X) throws Exception {
        fit(X);
        // For fit_transform use refined neighbor search with distances
        DataFrame result = X.copy();
        int n = result.rowCount();
        int d = columns.size();
        double[][] query = FeatureMatrices.copyMatrix(trainMatrix);
        // restore original missing mask for query
        for (int i = 0; i < n; i++) {
            boolean any = false;
            for (int j = 0; j < d; j++) if (trainMissing[i][j]) { any = true; break; }
            if (!any) continue;
            double[] dists = new double[n];
            Integer[] order = new Integer[n];
            for (int t = 0; t < n; t++) {
                order[t] = t;
                if (t == i) {
                    dists[t] = Double.POSITIVE_INFINITY;
                } else {
                    dists[t] = euclidean(query[i], trainMatrix[t]);
                }
            }
            Arrays.sort(order, (a, b) -> Double.compare(dists[a], dists[b]));
            int k = Math.min(nNeighbors, n - 1);
            int[] nn = new int[k];
            double[] nnDist = new double[k];
            for (int r = 0; r < k; r++) {
                nn[r] = order[r];
                nnDist[r] = dists[order[r]];
            }
            for (int j = 0; j < d; j++) {
                if (!trainMissing[i][j]) continue;
                double fill;
                if ("distance".equals(weights)) {
                    fill = imputeDistance(nn, nnDist, j);
                } else {
                    double sum = 0;
                    for (int t : nn) sum += trainMatrix[t][j];
                    fill = sum / k;
                }
                result.column(columns.get(j)).set(i, fill);
            }
        }
        return result;
    }

    private List<String> resolveColumns(DataFrame X) {
        if (columns != null && !columns.isEmpty()) return new ArrayList<>(columns);
        return FeatureMatrices.numericColumnNames(X);
    }

    private static double euclidean(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    public int getNNeighbors() { return nNeighbors; }
    public String getWeights() { return weights; }
    public Map<String, Double> getColumnMeans() { return columnMeans; }
}
