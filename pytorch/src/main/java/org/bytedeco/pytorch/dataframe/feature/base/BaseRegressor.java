package org.bytedeco.pytorch.dataframe.feature.base;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

/** Abstract regressor base: dual API for {@code double[][]} and {@link DataFrame}. */
public abstract class BaseRegressor implements BaseEstimator, Serializable {
    private static final long serialVersionUID = 1L;
    protected boolean fitted = false;

    public abstract BaseRegressor fit(double[][] X, double[] y);

    public abstract double[] predict(double[][] X);

    public BaseRegressor fit(DataFrame X, String[] featureCols, String labelCol) {
        return fit(extractMatrix(X, featureCols), extractLabels(X, labelCol));
    }

    public DataFrame predict(DataFrame X, String[] featureCols, String outputCol) {
        double[] preds = predict(extractMatrix(X, featureCols));
        DataFrame out = X.copy();
        if (out.hasColumn(outputCol)) out.removeColumn(outputCol);
        out.addColumn(outputCol, Column.DType.FLOAT64);
        Column c = out.column(outputCol);
        while (c.size() < out.rowCount()) c.add(null);
        for (int i = 0; i < preds.length; i++) c.set(i, preds[i]);
        return out;
    }

    /** Extract numeric feature matrix (shared with classifiers). */
    public double[][] extractMatrix(DataFrame df, String[] cols) {
        return BaseClassifier.extractMatrix(df, cols);
    }

    /** R² score. */
    public double score(double[][] X, double[] y) {
        double[] preds = predict(X);
        double yMean = 0;
        for (double v : y) yMean += v;
        yMean /= Math.max(1, y.length);
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < y.length; i++) {
            ssTot += (y[i] - yMean) * (y[i] - yMean);
            ssRes += (y[i] - preds[i]) * (y[i] - preds[i]);
        }
        return ssTot == 0 ? 0.0 : 1.0 - ssRes / ssTot;
    }

    protected double[] extractLabels(DataFrame df, String col) {
        int n = df.rowCount();
        double[] y = new double[n];
        Column c = df.column(col);
        for (int i = 0; i < n; i++) {
            double v = DataValues.asDouble(c.get(i));
            y[i] = Double.isNaN(v) ? 0.0 : v;
        }
        return y;
    }

    public boolean isFitted() { return fitted; }

    @Override public Map<String, Object> getParams() { return new LinkedHashMap<>(); }
    @Override public void setParams(Map<String, Object> params) {}
}
