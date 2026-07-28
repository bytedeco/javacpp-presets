package org.bytedeco.pytorch.dataframe.feature.construction;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.function.Function;

/**
 * Apply a custom function to features (sklearn FunctionTransformer).
 *
 * <p>Two modes:
 * <ul>
 *   <li>Element-wise: {@code Function<Double, Double>} per cell</li>
 *   <li>Matrix: {@code Function<double[][], double[][]>} for bulk logic
 *       (log-mean fill, business bins, ratio features, …)</li>
 * </ul>
 */
public class FunctionTransformer extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private Function<Double, Double> elementFn;
    private Function<double[][], double[][]> matrixFn;
    private String functionName = "fn";
    private boolean replace = true;
    private String[] outputNames;

    public FunctionTransformer(Function<Double, Double> function, String functionName, String... columns) {
        super(columns);
        this.elementFn = function;
        this.functionName = functionName == null ? "fn" : functionName;
    }

    /** Preferred factory for matrix callables. */
    public static FunctionTransformer ofMatrix(Function<double[][], double[][]> matrixFn,
                                               String name, String... columns) {
        FunctionTransformer t = new FunctionTransformer((Function<Double, Double>) null, name, columns);
        t.matrixFn = matrixFn;
        t.elementFn = null;
        return t;
    }

    public FunctionTransformer setReplace(boolean replace) {
        this.replace = replace;
        return this;
    }

    public FunctionTransformer setOutputNames(String... outputNames) {
        this.outputNames = outputNames;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        if (matrixFn != null) {
            return transformMatrix(X);
        }
        if (elementFn == null) {
            throw new IllegalStateException("FunctionTransformer has no function configured");
        }
        DataFrame result = X.copy();
        for (String col : columns) {
            String outName = replace ? col : col + "_" + functionName;
            if (!replace) {
                if (result.hasColumn(outName)) result.removeColumn(outName);
                result.addColumn(outName, Column.DType.FLOAT64);
                Column oc = result.column(outName);
                while (oc.size() < result.rowCount()) oc.add(null);
            }
            Column src = X.column(col);
            Column dst = result.column(outName);
            for (int i = 0; i < result.rowCount(); i++) {
                Object value = src.get(i);
                if (value == null || Double.isNaN(DataValues.asDouble(value))) {
                    dst.set(i, null);
                } else {
                    double v = DataValues.asDouble(value);
                    dst.set(i, elementFn.apply(v));
                }
            }
        }
        return result;
    }

    private DataFrame transformMatrix(DataFrame X) {
        String[] cols = columns.toArray(new String[0]);
        double[][] src = FeatureMatrices.fromDf(X, cols);
        double[][] out = matrixFn.apply(src);
        if (out == null) throw new IllegalStateException("matrix function returned null");
        int dOut = out.length == 0 ? 0 : out[0].length;
        if (dOut == cols.length && replace) {
            return FeatureMatrices.replaceColumns(X, cols, out);
        }
        String[] names;
        if (outputNames != null && outputNames.length == dOut) {
            names = outputNames;
        } else {
            names = new String[dOut];
            for (int j = 0; j < dOut; j++) names[j] = functionName + "_" + j;
        }
        DataFrame base = X.copy();
        if (replace) {
            for (String c : cols) {
                if (base.hasColumn(c)) base.removeColumn(c);
            }
        }
        return FeatureMatrices.appendColumns(base, names, out);
    }

    public String getFunctionName() { return functionName; }
}
