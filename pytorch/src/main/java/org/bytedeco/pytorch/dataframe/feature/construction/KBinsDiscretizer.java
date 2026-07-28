package org.bytedeco.pytorch.dataframe.feature.construction;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Discretize continuous features into K bins (sklearn KBinsDiscretizer).
 * Strategies: {@code uniform}, {@code quantile}, {@code kmeans}.
 * Encode: {@code ordinal} (default) or {@code onehot}/{@code onehot-dense}.
 */
public class KBinsDiscretizer extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int nBins;
    private String strategy = "quantile";
    private String encode = "ordinal";
    private final Map<String, double[]> binEdges = new HashMap<>();
    private boolean replace = false;
    private long randomState = 42L;

    /** Columns only; strategy defaults to quantile, encode to ordinal. */
    public KBinsDiscretizer(int nBins, String... columns) {
        super(columns);
        this.nBins = Math.max(2, nBins);
        this.strategy = "quantile";
        this.encode = "ordinal";
    }

    /** strategy + explicit column array (avoids varargs overload ambiguity). */
    public KBinsDiscretizer(int nBins, String strategy, String[] columns) {
        super(columns);
        this.nBins = Math.max(2, nBins);
        this.strategy = strategy == null ? "quantile" : strategy.toLowerCase(Locale.ROOT);
        this.encode = "ordinal";
    }

    public static KBinsDiscretizer withStrategy(int nBins, String strategy, String... columns) {
        return new KBinsDiscretizer(nBins, strategy, columns);
    }

    public static KBinsDiscretizer of(int nBins, String strategy, String encode, String... columns) {
        KBinsDiscretizer b = new KBinsDiscretizer(nBins, strategy, columns);
        b.setEncode(encode);
        return b;
    }

    public KBinsDiscretizer setStrategy(String strategy) {
        this.strategy = strategy == null ? "quantile" : strategy.toLowerCase(Locale.ROOT);
        return this;
    }

    public KBinsDiscretizer setEncode(String encode) {
        this.encode = encode == null ? "ordinal" : encode.toLowerCase(Locale.ROOT);
        return this;
    }

    public KBinsDiscretizer setReplace(boolean replace) {
        this.replace = replace;
        return this;
    }

    public KBinsDiscretizer setRandomState(long seed) {
        this.randomState = seed;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        binEdges.clear();
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        for (String col : columns) {
            List<Double> values = new ArrayList<>();
            Column c = X.column(col);
            for (int i = 0; i < c.size(); i++) {
                Object v = c.get(i);
                if (v == null) continue;
                double d = DataValues.asDouble(v);
                if (!Double.isNaN(d)) values.add(d);
            }
            Collections.sort(values);
            double[] edges;
            if (values.isEmpty()) {
                edges = new double[]{0, 1};
            } else if ("uniform".equals(strategy)) {
                edges = computeUniformEdges(values);
            } else if ("kmeans".equals(strategy)) {
                edges = computeKMeansEdges(values);
            } else {
                edges = computeQuantileEdges(values);
            }
            binEdges.put(col, edges);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        boolean onehot = encode.startsWith("onehot");
        DataFrame result = X.copy();

        for (String col : columns) {
            double[] edges = binEdges.get(col);
            if (edges == null) continue;
            int nBinsEff = edges.length - 1;
            Column src = X.column(col);

            if (!onehot) {
                String outName = replace ? col : col + "_bin";
                if (!replace) {
                    if (result.hasColumn(outName)) result.removeColumn(outName);
                    result.addColumn(outName, Column.DType.INT32);
                    Column oc = result.column(outName);
                    while (oc.size() < result.rowCount()) oc.add(null);
                }
                Column dst = result.column(outName);
                for (int i = 0; i < result.rowCount(); i++) {
                    Object value = src.get(i);
                    if (value == null || Double.isNaN(DataValues.asDouble(value))) {
                        dst.set(i, -1);
                    } else {
                        dst.set(i, findBin(DataValues.asDouble(value), edges));
                    }
                }
                if (replace && !outName.equals(col) && result.hasColumn(col)) {
                    result.removeColumn(col);
                }
            } else {
                // one-hot dense columns
                List<String> dummyNames = new ArrayList<>();
                for (int b = 0; b < nBinsEff; b++) {
                    String name = col + "_bin_" + b;
                    String base = name;
                    int n = 1;
                    while (result.hasColumn(name)) name = base + "_" + (n++);
                    result.addColumn(name, Column.DType.INT32);
                    Column dcol = result.column(name);
                    while (dcol.size() < result.rowCount()) dcol.add(0);
                    for (int i = 0; i < result.rowCount(); i++) dcol.set(i, 0);
                    dummyNames.add(name);
                }
                for (int i = 0; i < result.rowCount(); i++) {
                    Object value = src.get(i);
                    if (value == null || Double.isNaN(DataValues.asDouble(value))) continue;
                    int bin = findBin(DataValues.asDouble(value), edges);
                    if (bin >= 0 && bin < dummyNames.size()) {
                        result.set(i, dummyNames.get(bin), 1);
                    }
                }
                if (replace && result.hasColumn(col)) result.removeColumn(col);
            }
        }
        return result;
    }

    private double[] computeUniformEdges(List<Double> values) {
        double min = values.get(0);
        double max = values.get(values.size() - 1);
        if (max == min) max = min + 1e-9;
        double[] edges = new double[nBins + 1];
        for (int i = 0; i <= nBins; i++) {
            edges[i] = min + (max - min) * i / nBins;
        }
        return edges;
    }

    private double[] computeQuantileEdges(List<Double> values) {
        double[] edges = new double[nBins + 1];
        for (int i = 0; i <= nBins; i++) {
            edges[i] = FeatureMatrices.percentileSorted(values, (double) i / nBins);
        }
        // ensure strictly non-decreasing
        for (int i = 1; i < edges.length; i++) {
            if (edges[i] < edges[i - 1]) edges[i] = edges[i - 1];
        }
        if (edges[edges.length - 1] == edges[0]) edges[edges.length - 1] = edges[0] + 1e-9;
        return edges;
    }

    /** 1D k-means on values → sort centers → midpoints as edges. */
    private double[] computeKMeansEdges(List<Double> values) {
        int n = values.size();
        int k = Math.min(nBins, n);
        if (k < 2) return computeUniformEdges(values);
        double[] data = new double[n];
        for (int i = 0; i < n; i++) data[i] = values.get(i);
        Random rng = new Random(randomState);
        double[] centers = new double[k];
        // init: quantiles
        for (int i = 0; i < k; i++) {
            centers[i] = FeatureMatrices.percentileSorted(values, (i + 0.5) / k);
        }
        int[] assign = new int[n];
        for (int iter = 0; iter < 30; iter++) {
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY;
                int bi = 0;
                for (int c = 0; c < k; c++) {
                    double d = Math.abs(data[i] - centers[c]);
                    if (d < best) { best = d; bi = c; }
                }
                assign[i] = bi;
            }
            double[] sum = new double[k];
            int[] cnt = new int[k];
            for (int i = 0; i < n; i++) {
                sum[assign[i]] += data[i];
                cnt[assign[i]]++;
            }
            for (int c = 0; c < k; c++) {
                if (cnt[c] > 0) centers[c] = sum[c] / cnt[c];
                else centers[c] = data[rng.nextInt(n)];
            }
        }
        java.util.Arrays.sort(centers);
        double[] edges = new double[k + 1];
        edges[0] = values.get(0);
        edges[k] = values.get(n - 1);
        for (int i = 1; i < k; i++) {
            edges[i] = 0.5 * (centers[i - 1] + centers[i]);
        }
        return edges;
    }

    private int findBin(double value, double[] edges) {
        // right-inclusive style: last bin catches max
        for (int i = 0; i < edges.length - 1; i++) {
            if (i == edges.length - 2) return i;
            if (value < edges[i + 1]) return i;
        }
        return edges.length - 2;
    }

    public Map<String, double[]> getBinEdges() { return binEdges; }
    public String getStrategy() { return strategy; }
    public String getEncode() { return encode; }
}
