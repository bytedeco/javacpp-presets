package org.bytedeco.pytorch.dataframe.feature.selection;

import org.bytedeco.pytorch.dataframe.DataValues;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import org.bytedeco.pytorch.dataframe.ml.classification.LogisticRegression;
import org.bytedeco.pytorch.dataframe.ml.regression.Lasso;
import org.bytedeco.pytorch.dataframe.ml.regression.LinearRegression;
import org.bytedeco.pytorch.dataframe.ml.regression.Ridge;

import java.util.*;

/**
 * 递归特征消除（RFE – Recursive Feature Elimination）
 * 对应 sklearn RFE
 *
 * <pre>
 * RFE rfe = new RFE(new LogisticRegression(), 5, featureCols, "label");
 * rfe.fit(df);
 * DataFrame selected = rfe.transform(df);
 * </pre>
 */
public class RFE extends BaseTransformer {
    private final Object estimator;   // BaseClassifier | BaseRegressor
    private final int nFeaturesToSelect;
    private final String labelCol;
    private List<String> selectedCols;

    public RFE(BaseClassifier estimator, int nFeaturesToSelect, String[] featureCols, String labelCol) {
        super(featureCols);
        this.estimator = estimator;
        this.nFeaturesToSelect = nFeaturesToSelect;
        this.labelCol = labelCol;
    }

    public RFE(BaseRegressor estimator, int nFeaturesToSelect, String[] featureCols, String labelCol) {
        super(featureCols);
        this.estimator = estimator;
        this.nFeaturesToSelect = nFeaturesToSelect;
        this.labelCol = labelCol;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<String> remaining = new ArrayList<>(columns);
        while (remaining.size() > nFeaturesToSelect) {
            // Extract matrix and labels
            double[][] mat = extractMatrix(X, remaining);
            double[] y = extractLabels(X, labelCol);

            // Fit estimator
            double[] importances;
            if (estimator instanceof BaseClassifier clf) {
                clf.fit(mat, y);
                importances = getImportances(clf, mat[0].length);
            } else {
                BaseRegressor reg = (BaseRegressor) estimator;
                reg.fit(mat, y);
                importances = getImportances(reg, mat[0].length);
            }

            // Remove the least important feature
            int worstIdx = 0;
            for (int j = 1; j < importances.length; j++)
                if (importances[j] < importances[worstIdx]) worstIdx = j;
            remaining.remove(worstIdx);
        }
        selectedCols = remaining;
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("RFE not fitted");
        return X.select(selectedCols.toArray(new String[0]));
    }

    private double[] getImportances(Object model, int d) {
        // Prefer |coef| / feature_importances_; never silent uniform for LR.
        double[] imp = new double[d];
        if (model instanceof LogisticRegression lr) {
            double[] coef = lr.getCoef(); // mean |weight| across classes
            if (coef != null && coef.length > 0) {
                for (int j = 0; j < Math.min(d, coef.length); j++) imp[j] = Math.abs(coef[j]);
                return imp;
            }
            double[][] w = lr.getWeights();
            if (w != null && w.length > 0) {
                int dim = w[0].length;
                for (double[] row : w)
                    for (int j = 0; j < Math.min(d, dim); j++) imp[j] += Math.abs(row[j]);
                for (int j = 0; j < d; j++) imp[j] /= w.length;
                return imp;
            }
        } else if (model instanceof LinearRegression linreg) {
            double[] coef = linreg.getCoef();
            for (int j = 0; j < Math.min(d, coef.length); j++) imp[j] = Math.abs(coef[j]);
            return imp;
        } else if (model instanceof Ridge ridge) {
            double[] coef = ridge.getCoef();
            for (int j = 0; j < Math.min(d, coef.length); j++) imp[j] = Math.abs(coef[j]);
            return imp;
        } else if (model instanceof Lasso lasso) {
            double[] coef = lasso.getCoef();
            for (int j = 0; j < Math.min(d, coef.length); j++) imp[j] = Math.abs(coef[j]);
            return imp;
        } else {
            // reflection: getFeatureImportances / getCoef
            try {
                var m = model.getClass().getMethod("getFeatureImportances");
                Object r = m.invoke(model);
                if (r instanceof double[] arr) {
                    for (int j = 0; j < Math.min(d, arr.length); j++) imp[j] = Math.abs(arr[j]);
                    return imp;
                }
            } catch (Throwable ignored) {}
            try {
                var m = model.getClass().getMethod("getCoef");
                Object r = m.invoke(model);
                if (r instanceof double[] arr) {
                    for (int j = 0; j < Math.min(d, arr.length); j++) imp[j] = Math.abs(arr[j]);
                    return imp;
                }
            } catch (Throwable ignored) {}
            Arrays.fill(imp, 1.0); // last-resort fallback
        }
        return imp;
    }

    private double[][] extractMatrix(DataFrame df, List<String> cols) {
        int n = df.rowCount();
        double[][] mat = new double[n][cols.size()];
        for (int j = 0; j < cols.size(); j++) {
            List<?> data = df.column(cols.get(j)).data();
            for (int i = 0; i < n; i++) {
                Object v = data.get(i);
                mat[i][j] = v == null ? 0.0 :
                    DataValues.asDouble(v);
            }
        }
        return mat;
    }

    private double[] extractLabels(DataFrame df, String col) {
        int n = df.rowCount();
        double[] y = new double[n];
        List<?> data = df.column(col).data();
        for (int i = 0; i < n; i++) {
            Object v = data.get(i);
            y[i] = v == null ? 0.0 : DataValues.asDouble(v);
        }
        return y;
    }

    public List<String> getSelectedColumns() { return selectedCols; }
}

