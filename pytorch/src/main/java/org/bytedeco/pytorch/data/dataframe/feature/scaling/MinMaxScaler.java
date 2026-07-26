package org.bytedeco.pytorch.data.dataframe.feature.scaling;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.DataValues;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.HashMap;
import java.util.Map;

/** Scale features to a given range, default [0, 1]. */
public class MinMaxScaler extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    private final Map<String, Double> mins = new HashMap<>();
    private final Map<String, Double> maxs = new HashMap<>();
    private final double featureMin;
    private final double featureMax;
    private final boolean replace;

    public MinMaxScaler(String... columns) {
        this(0.0, 1.0, true, columns);
    }

    public MinMaxScaler(double featureMin, double featureMax, boolean replace, String... columns) {
        super(columns);
        this.featureMin = featureMin;
        this.featureMax = featureMax;
        this.replace = replace;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        mins.clear();
        maxs.clear();
        for (String col : columns) {
            Column c = X.column(col);
            double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < c.size(); i++) {
                double v = DataValues.asDouble(c.get(i));
                if (Double.isNaN(v)) continue;
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
            if (min == Double.POSITIVE_INFINITY) { min = 0; max = 1; }
            mins.put(col, min);
            maxs.put(col, max);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (String col : columns) {
            double min = mins.getOrDefault(col, 0.0);
            double max = maxs.getOrDefault(col, 1.0);
            double range = max - min;
            if (range == 0) range = 1.0;
            String outName = replace ? col : col + "_scaled";
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
                if (Double.isNaN(v)) dst.set(i, null);
                else {
                    double t = (v - min) / range;
                    dst.set(i, t * (featureMax - featureMin) + featureMin);
                }
            }
        }
        return result;
    }
}
