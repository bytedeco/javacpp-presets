package org.bytedeco.pytorch.dataframe.feature.selection;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;
import org.bytedeco.pytorch.dataframe.ml.classification.LogisticRegression;
import org.bytedeco.pytorch.dataframe.ml.classification.RandomForestClassifier;
import org.bytedeco.pytorch.dataframe.ml.regression.Lasso;
import org.bytedeco.pytorch.dataframe.ml.regression.LinearRegression;
import org.bytedeco.pytorch.dataframe.ml.regression.Ridge;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Model-based feature selection (sklearn SelectFromModel).
 * Threshold may be a numeric value or {@code "mean"} / {@code "median"}.
 */
public class SelectFromModel extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private final Object estimator;
    private Double thresholdValue;   // numeric threshold; null if string mode
    private String thresholdMode;    // "mean" | "median" | null
    private final String labelCol;
    private List<String> selectedCols = new ArrayList<>();
    private double[] importances;

    public SelectFromModel(BaseClassifier estimator, Double threshold, String[] featureCols, String labelCol) {
        super(featureCols);
        this.estimator = estimator;
        this.thresholdValue = threshold;
        this.labelCol = labelCol;
    }

    public SelectFromModel(BaseRegressor estimator, Double threshold, String[] featureCols, String labelCol) {
        super(featureCols);
        this.estimator = estimator;
        this.thresholdValue = threshold;
        this.labelCol = labelCol;
    }

    public SelectFromModel(Object estimator, String threshold, String[] featureCols, String labelCol) {
        super(featureCols);
        this.estimator = estimator;
        this.labelCol = labelCol;
        setThreshold(threshold);
    }

    public SelectFromModel setThreshold(String threshold) {
        if (threshold == null) {
            this.thresholdMode = "mean";
            this.thresholdValue = null;
            return this;
        }
        String t = threshold.toLowerCase(Locale.ROOT).trim();
        if ("mean".equals(t) || "median".equals(t)) {
            this.thresholdMode = t;
            this.thresholdValue = null;
        } else {
            this.thresholdMode = null;
            this.thresholdValue = Double.parseDouble(t);
        }
        return this;
    }

    public SelectFromModel setThreshold(double threshold) {
        this.thresholdValue = threshold;
        this.thresholdMode = null;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
            if (labelCol != null) columns.remove(labelCol);
        }
        String[] cols = columns.toArray(new String[0]);
        double[][] mat = FeatureMatrices.fromDf(X, cols);
        if (labelCol == null || !X.hasColumn(labelCol)) {
            throw new IllegalStateException("SelectFromModel requires labelCol in DataFrame");
        }
        double[] y = FeatureMatrices.columnAsDoubles(X, labelCol);

        if (estimator instanceof BaseClassifier clf) {
            clf.fit(mat, y);
        } else if (estimator instanceof BaseRegressor reg) {
            reg.fit(mat, y);
        } else {
            throw new IllegalStateException("Unsupported estimator: " + estimator.getClass().getName());
        }

        importances = extractImportances(cols.length);
        double thresh = resolveThreshold(importances);

        selectedCols = new ArrayList<>();
        for (int j = 0; j < cols.length; j++) {
            if (importances[j] >= thresh) selectedCols.add(cols[j]);
        }
        if (selectedCols.isEmpty()) selectedCols.add(cols[0]);
        fitted = true;
        return this;
    }

    private double[] extractImportances(int d) {
        // LogisticRegression
        if (estimator instanceof LogisticRegression lr) {
            double[] coef = tryGetCoef(lr, d);
            if (coef != null) return abs(coef);
        }
        // Linear models
        if (estimator instanceof LinearRegression lin) {
            try {
                double[] c = lin.getCoef();
                if (c != null) return abs(pad(c, d));
            } catch (Throwable ignored) {}
        }
        if (estimator instanceof Ridge ridge) {
            try {
                double[] c = ridge.getCoef();
                if (c != null) return abs(pad(c, d));
            } catch (Throwable ignored) {}
        }
        if (estimator instanceof Lasso lasso) {
            try {
                double[] c = lasso.getCoef();
                if (c != null) return abs(pad(c, d));
            } catch (Throwable ignored) {}
        }
        // RandomForest via reflection (featureImportances / getFeatureImportances)
        if (estimator instanceof RandomForestClassifier || estimator != null) {
            double[] fi = tryInvokeDoubleArray(estimator, "getFeatureImportances", d);
            if (fi == null) fi = tryInvokeDoubleArray(estimator, "featureImportances", d);
            if (fi == null) fi = tryGetCoef(estimator, d);
            if (fi != null) return abs(fi);
        }
        // fallback: all ones
        double[] ones = new double[d];
        Arrays.fill(ones, 1.0);
        return ones;
    }

    private static double[] tryGetCoef(Object est, int d) {
        double[] c = tryInvokeDoubleArray(est, "getCoef", d);
        if (c != null) return c;
        // LogisticRegression multi-class: getWeights
        double[][] w = tryInvokeDoubleMatrix(est, "getWeights");
        if (w != null && w.length > 0) {
            int dim = w[0].length;
            double[] out = new double[dim];
            for (double[] row : w) {
                for (int j = 0; j < dim; j++) out[j] += Math.abs(row[j]);
            }
            for (int j = 0; j < dim; j++) out[j] /= w.length;
            return pad(out, d);
        }
        return null;
    }

    private static double[] tryInvokeDoubleArray(Object est, String method, int d) {
        try {
            Method m = est.getClass().getMethod(method);
            Object r = m.invoke(est);
            if (r instanceof double[] arr) return pad(arr, d);
        } catch (Throwable ignored) {}
        return null;
    }

    private static double[][] tryInvokeDoubleMatrix(Object est, String method) {
        try {
            Method m = est.getClass().getMethod(method);
            Object r = m.invoke(est);
            if (r instanceof double[][] arr) return arr;
        } catch (Throwable ignored) {}
        return null;
    }

    private static double[] abs(double[] a) {
        double[] o = new double[a.length];
        for (int i = 0; i < a.length; i++) o[i] = Math.abs(a[i]);
        return o;
    }

    private static double[] pad(double[] a, int d) {
        if (a.length == d) return a;
        double[] o = new double[d];
        System.arraycopy(a, 0, o, 0, Math.min(a.length, d));
        return o;
    }

    private double resolveThreshold(double[] imp) {
        if (thresholdValue != null) return thresholdValue;
        String mode = thresholdMode == null ? "mean" : thresholdMode;
        if ("median".equals(mode)) {
            double[] sorted = imp.clone();
            Arrays.sort(sorted);
            int n = sorted.length;
            return n % 2 == 1 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
        }
        double sum = 0;
        for (double v : imp) sum += v;
        return imp.length == 0 ? 0 : sum / imp.length;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        return X.select(selectedCols.toArray(new String[0]));
    }

    public List<String> getSelectedColumns() { return selectedCols; }
    public double[] getImportances() { return importances; }
}
