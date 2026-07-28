package org.bytedeco.pytorch.dataframe.feature.scaling;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Robust scaler using median and inter-quantile range (sklearn RobustScaler).
 * Supports custom {@code quantile_range} e.g. (2.5, 97.5).
 */
public class RobustScaler extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private double qLow = 25.0;
    private double qHigh = 75.0;
    private final Map<String, Double> centers = new HashMap<>();
    private final Map<String, Double> scales = new HashMap<>();
    private boolean replace = true;

    public RobustScaler(String... columns) {
        super(columns);
    }

    public RobustScaler setQuantileRange(double lowPercent, double highPercent) {
        if (lowPercent < 0 || highPercent > 100 || lowPercent >= highPercent) {
            throw new IllegalArgumentException("quantile_range must satisfy 0 <= low < high <= 100");
        }
        this.qLow = lowPercent;
        this.qHigh = highPercent;
        return this;
    }

    public RobustScaler setReplace(boolean replace) {
        this.replace = replace;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        centers.clear();
        scales.clear();
        List<String> cols = columns.isEmpty() ? FeatureMatrices.numericColumnNames(X) : columns;
        if (columns.isEmpty()) this.columns = new ArrayList<>(cols);

        for (String col : cols) {
            List<Double> values = new ArrayList<>();
            Column c = X.column(col);
            for (int i = 0; i < c.size(); i++) {
                Object v = c.get(i);
                if (v == null) continue;
                double d = DataValues.asDouble(v);
                if (!Double.isNaN(d)) values.add(d);
            }
            if (values.isEmpty()) {
                centers.put(col, 0.0);
                scales.put(col, 1.0);
                continue;
            }
            Collections.sort(values);
            double median = FeatureMatrices.percentileSorted(values, 0.5);
            double lo = FeatureMatrices.percentileSorted(values, qLow / 100.0);
            double hi = FeatureMatrices.percentileSorted(values, qHigh / 100.0);
            double scale = hi - lo;
            if (scale == 0.0 || Double.isNaN(scale)) scale = 1.0;
            centers.put(col, median);
            scales.put(col, scale);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (String col : columns) {
            double center = centers.getOrDefault(col, 0.0);
            double scale = scales.getOrDefault(col, 1.0);
            String outName = replace ? col : col + "_robust";
            if (!replace) {
                if (result.hasColumn(outName)) result.removeColumn(outName);
                result.addColumn(outName, Column.DType.FLOAT64);
                Column oc = result.column(outName);
                while (oc.size() < result.rowCount()) oc.add(null);
            }
            Column src = X.column(col);
            Column dst = result.column(outName);
            for (int i = 0; i < result.rowCount(); i++) {
                double v = DataValues.asDouble(src.get(i));
                dst.set(i, Double.isNaN(v) ? null : (v - center) / scale);
            }
        }
        return result;
    }

    public Map<String, Double> getMedians() { return centers; }
    public Map<String, Double> getIQRs() { return scales; }
    public double[] getQuantileRange() { return new double[]{qLow, qHigh}; }
}
