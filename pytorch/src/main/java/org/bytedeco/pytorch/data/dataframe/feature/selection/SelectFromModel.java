package org.bytedeco.pytorch.data.dataframe.feature.selection;

import org.bytedeco.pytorch.data.dataframe.DataValues;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 基于模型的特征选择（对应 sklearn SelectFromModel）
 * 使用模型权重（|coef|）筛选特征
 */
public class SelectFromModel extends BaseTransformer {
    private final Object estimator;
    private Double threshold;   // null → mean
    private final String labelCol;
    private List<String> selectedCols;

    public SelectFromModel(BaseClassifier estimator, Double threshold, String[] featureCols, String labelCol) {
        super(featureCols); this.estimator = estimator; this.threshold = threshold; this.labelCol = labelCol;
    }

    public SelectFromModel(BaseRegressor estimator, Double threshold, String[] featureCols, String labelCol) {
        super(featureCols); this.estimator = estimator; this.threshold = threshold; this.labelCol = labelCol;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        int n = X.rowCount();
        double[][] mat = new double[n][columns.size()];
        for (int j = 0; j < columns.size(); j++) {
            List<Object> data = X.column(columns.get(j)).data();
            for (int i = 0; i < n; i++) mat[i][j] = data.get(i) == null ? 0.0 : DataValues.asDouble(data.get(i));
        }
        List<Object> labelData = X.column(labelCol).data();
        double[] y = new double[n];
        for (int i = 0; i < n; i++) y[i] = labelData.get(i) == null ? 0.0 : DataValues.asDouble(labelData.get(i));

        double[] importances;
        if (estimator instanceof BaseClassifier clf) {
            clf.fit(mat, y); importances = new double[columns.size()]; Arrays.fill(importances, 1.0);
        } else {
            BaseRegressor reg = (BaseRegressor) estimator;
            reg.fit(mat, y);
            if (reg instanceof org.bytedeco.pytorch.data.dataframe.ml.regression.LinearRegression lr) {
                double[] c = lr.getCoef();
                importances = Arrays.stream(c).map(Math::abs).toArray();
            } else if (reg instanceof org.bytedeco.pytorch.data.dataframe.ml.regression.Ridge r) {
                double[] c = r.getCoef();
                importances = Arrays.stream(c).map(Math::abs).toArray();
            } else if (reg instanceof org.bytedeco.pytorch.data.dataframe.ml.regression.Lasso r) {
                double[] c = r.getCoef();
                importances = Arrays.stream(c).map(Math::abs).toArray();
            } else {
                importances = new double[columns.size()]; Arrays.fill(importances, 1.0);
            }
        }

        double thresh = threshold == null
            ? Arrays.stream(importances).average().orElse(0)
            : threshold;

        selectedCols = new ArrayList<>();
        for (int j = 0; j < columns.size(); j++)
            if (importances[j] >= thresh) selectedCols.add(columns.get(j));

        if (selectedCols.isEmpty()) selectedCols.add(columns.get(0)); // at least one
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("SelectFromModel not fitted");
        return X.select(selectedCols.toArray(new String[0]));
    }

    public List<String> getSelectedColumns() { return selectedCols; }
}

