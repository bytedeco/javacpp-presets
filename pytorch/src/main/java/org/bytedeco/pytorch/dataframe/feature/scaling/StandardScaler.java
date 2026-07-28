package org.bytedeco.pytorch.dataframe.feature.scaling;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.HashMap;
import java.util.Map;

/** Standardize features by removing the mean and scaling to unit variance. */
public class StandardScaler extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    private final Map<String, Double> means = new HashMap<>();
    private final Map<String, Double> stds = new HashMap<>();
    private final boolean replace;

    public StandardScaler(String... columns) {
        this(true, columns);
    }

    public StandardScaler(boolean replace, String... columns) {
        super(columns);
        this.replace = replace;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        means.clear();
        stds.clear();
        for (String col : columns) {
            Column c = X.column(col);
            double sum = 0, sumSq = 0;
            int n = 0;
            for (int i = 0; i < c.size(); i++) {
                double v = DataValues.asDouble(c.get(i));
                if (Double.isNaN(v)) continue;
                sum += v;
                sumSq += v * v;
                n++;
            }
            double mean = n == 0 ? 0.0 : sum / n;
            double var = n < 2 ? 0.0 : (sumSq - n * mean * mean) / n; // population for scaler
            double std = Math.sqrt(Math.max(0, var));
            if (std == 0.0) std = 1.0; // avoid div-by-zero
            means.put(col, mean);
            stds.put(col, std);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (String col : columns) {
            double mean = means.getOrDefault(col, 0.0);
            double std = stds.getOrDefault(col, 1.0);
            String outName = replace ? col : col + "_scaled";
            if (!replace) {
                if (result.hasColumn(outName)) result.removeColumn(outName);
                result.addColumn(outName, Column.DType.FLOAT64);
                Column oc = result.column(outName);
                while (oc.size() < result.rowCount()) oc.add(null);
            } else if (result.column(col).dtype() != Column.DType.FLOAT64) {
                // replace in place with float values via set (dtype may stay original)
            }
            Column src = X.column(col);
            Column dst = result.column(outName);
            for (int i = 0; i < result.rowCount(); i++) {
                double v = DataValues.asDouble(src.get(i));
                dst.set(i, Double.isNaN(v) ? null : (v - mean) / std);
            }
        }
        return result;
    }

    public Map<String, Double> getMeans() { return means; }
    public Map<String, Double> getStds() { return stds; }
}
