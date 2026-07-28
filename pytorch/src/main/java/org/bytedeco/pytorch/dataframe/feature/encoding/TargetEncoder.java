package org.bytedeco.pytorch.dataframe.feature.encoding;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Target / mean encoding with optional CV anti-leakage (sklearn TargetEncoder-style).
 *
 * <p>Supports:
 * <ul>
 *   <li>{@code fit(DataFrame)} when target lives in {@code targetColumn}</li>
 *   <li>{@code fit(DataFrame X, double[] y)} when y is external</li>
 *   <li>{@code cv > 1}: out-of-fold encoding via {@link #fitTransform(DataFrame, double[])};
 *       global map used for pure {@code transform}</li>
 * </ul>
 */
public class TargetEncoder extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private final Map<String, Map<Object, Double>> encodings = new HashMap<>();
    private String targetColumn;
    private double smoothing = 1.0;
    private int cv = 1;
    private double globalMean = 0.0;
    private long randomState = 42L;
    /** When true, replace original columns; otherwise append {@code col_te}. */
    private boolean replace = true;

    public TargetEncoder(String targetColumn, String... columns) {
        super(columns);
        this.targetColumn = targetColumn;
    }

    public TargetEncoder(String targetColumn, double smoothing, String... columns) {
        super(columns);
        this.targetColumn = targetColumn;
        this.smoothing = smoothing;
    }

    public TargetEncoder setCv(int cv) {
        this.cv = Math.max(1, cv);
        return this;
    }

    public TargetEncoder setSmoothing(double smoothing) {
        this.smoothing = smoothing;
        return this;
    }

    public TargetEncoder setRandomState(long seed) {
        this.randomState = seed;
        return this;
    }

    public TargetEncoder setReplace(boolean replace) {
        this.replace = replace;
        return this;
    }

    public TargetEncoder setTargetColumn(String targetColumn) {
        this.targetColumn = targetColumn;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (targetColumn == null || !X.hasColumn(targetColumn)) {
            throw new IllegalStateException("TargetEncoder requires targetColumn in DataFrame for fit(X)");
        }
        double[] y = new double[X.rowCount()];
        Column tc = X.column(targetColumn);
        for (int i = 0; i < y.length; i++) {
            y[i] = DataValues.asDouble(tc.get(i));
            if (Double.isNaN(y[i])) y[i] = 0.0;
        }
        fitInternal(X, y);
        return this;
    }

    /** sklearn-style fit with external y. */
    public TargetEncoder fit(DataFrame X, double[] y) {
        if (y == null || y.length != X.rowCount()) {
            throw new IllegalArgumentException("y length must equal rowCount");
        }
        fitInternal(X, y);
        return this;
    }

    private void fitInternal(DataFrame X, double[] y) {
        encodings.clear();
        double sum = 0;
        for (double v : y) sum += v;
        globalMean = y.length == 0 ? 0.0 : sum / y.length;

        for (String col : columns) {
            Column c = X.column(col);
            Map<Object, List<Integer>> groups = new LinkedHashMap<>();
            for (int i = 0; i < c.size(); i++) {
                Object key = normalizeKey(DataValues.unwrap(c.get(i)));
                groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
            }
            Map<Object, Double> encoding = new HashMap<>();
            for (Map.Entry<Object, List<Integer>> e : groups.entrySet()) {
                List<Integer> idx = e.getValue();
                double catSum = 0;
                for (int i : idx) catSum += y[i];
                double catMean = catSum / idx.size();
                int count = idx.size();
                double smoothed = (count * catMean + smoothing * globalMean) / (count + smoothing);
                encoding.put(e.getKey(), smoothed);
            }
            encodings.put(col, encoding);
        }
        fitted = true;
    }

    /**
     * Fit + optional OOF transform for training set when {@code cv > 1}.
     */
    public DataFrame fitTransform(DataFrame X, double[] y) throws Exception {
        fit(X, y);
        if (cv <= 1) return transform(X);

        int n = X.rowCount();
        int[] fold = new int[n];
        Random rng = new Random(randomState);
        for (int i = 0; i < n; i++) fold[i] = rng.nextInt(cv);

        DataFrame result = X.copy();
        for (String col : columns) {
            Column src = X.column(col);
            double[] oof = new double[n];
            for (int f = 0; f < cv; f++) {
                Map<Object, double[]> stats = new HashMap<>();
                for (int i = 0; i < n; i++) {
                    if (fold[i] == f) continue;
                    Object key = normalizeKey(DataValues.unwrap(src.get(i)));
                    double[] sc = stats.computeIfAbsent(key, k -> new double[2]);
                    sc[0] += y[i];
                    sc[1] += 1;
                }
                for (int i = 0; i < n; i++) {
                    if (fold[i] != f) continue;
                    Object key = normalizeKey(DataValues.unwrap(src.get(i)));
                    double[] sc = stats.get(key);
                    if (sc == null || sc[1] == 0) {
                        oof[i] = globalMean;
                    } else {
                        double catMean = sc[0] / sc[1];
                        oof[i] = (sc[1] * catMean + smoothing * globalMean) / (sc[1] + smoothing);
                    }
                }
            }
            String outName = replace ? col : col + "_te";
            if (!replace) {
                if (result.hasColumn(outName)) result.removeColumn(outName);
                result.addColumn(outName, Column.DType.FLOAT64);
                Column oc = result.column(outName);
                while (oc.size() < n) oc.add(null);
            }
            Column dst = result.column(outName);
            for (int i = 0; i < n; i++) dst.set(i, oof[i]);
        }
        return result;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (String col : columns) {
            Map<Object, Double> encoding = encodings.get(col);
            if (encoding == null) continue;
            String outName = replace ? col : col + "_te";
            if (!replace) {
                if (result.hasColumn(outName)) result.removeColumn(outName);
                result.addColumn(outName, Column.DType.FLOAT64);
                Column oc = result.column(outName);
                while (oc.size() < result.rowCount()) oc.add(null);
            }
            Column src = X.column(col);
            Column dst = result.column(outName);
            for (int i = 0; i < result.rowCount(); i++) {
                Object key = normalizeKey(DataValues.unwrap(src.get(i)));
                Double val = encoding.get(key);
                dst.set(i, val != null ? val : globalMean);
            }
        }
        return result;
    }

    private static Object normalizeKey(Object v) {
        return v == null ? "null" : v;
    }

    public Map<Object, Double> getEncoding(String column) {
        return encodings.get(column);
    }

    public double getGlobalMean() { return globalMean; }
    public int getCv() { return cv; }
}
