package org.bytedeco.pytorch.data.dataframe.feature.selection;

import org.bytedeco.pytorch.data.dataframe.DataValues;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseClassifier;
import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import org.bytedeco.pytorch.data.dataframe.ml.model_selection.CrossValidation;
import org.bytedeco.pytorch.data.dataframe.ml.model_selection.KFold;

import java.util.*;

/**
 * 带交叉验证的递归特征消除（RFECV）
 * 自动确定最优特征数量
 */
public class RFECV extends BaseTransformer {
    private final Object estimator;
    private final int minFeaturesToSelect;
    private final int cv;
    private final String scoring;
    private final String labelCol;

    private List<String> selectedCols;
    private int optimalNFeatures;
    private double[] gridScores;

    public RFECV(BaseClassifier estimator, int minFeaturesToSelect, int cv,
                  String scoring, String[] featureCols, String labelCol) {
        super(featureCols);
        this.estimator = estimator; this.minFeaturesToSelect = minFeaturesToSelect;
        this.cv = cv; this.scoring = scoring; this.labelCol = labelCol;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<String> remaining = new ArrayList<>(columns);
        List<String> eliminatedOrder = new ArrayList<>();
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestSize = remaining.size();

        while (remaining.size() > minFeaturesToSelect) {
            double[][] mat = extractMatrix(X, remaining);
            double[] y = extractLabels(X, labelCol);

            // CV score with current features
            double score = cvScore(mat, y);
            if (score >= bestScore) { bestScore = score; bestSize = remaining.size(); }

            // Fit and get importances
            double[] imp = getImportances(mat, y, remaining.size());
            int worst = 0; for (int j = 1; j < imp.length; j++) if (imp[j] < imp[worst]) worst = j;
            eliminatedOrder.add(0, remaining.remove(worst));
        }
        optimalNFeatures = bestSize;
        // Rebuild selected: all columns minus the last (n - bestSize) eliminated
        selectedCols = new ArrayList<>(columns);
        for (String col : eliminatedOrder.subList(0, columns.size() - bestSize)) selectedCols.remove(col);

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("RFECV not fitted");
        return X.select(selectedCols.toArray(new String[0]));
    }

    private double cvScore(double[][] X, double[] y) {
        if (estimator instanceof BaseClassifier clf) {
            double[] scores = CrossValidation.crossValScore(clf, X, y, cv, scoring);
            return Arrays.stream(scores).average().orElse(0);
        } else {
            BaseRegressor reg = (BaseRegressor) estimator;
            double[] scores = CrossValidation.crossValScore(reg, X, y, cv, scoring);
            return Arrays.stream(scores).average().orElse(0);
        }
    }

    private double[] getImportances(double[][] X, double[] y, int d) {
        double[] imp = new double[d];
        if (estimator instanceof BaseClassifier clf) { clf.fit(X, y); Arrays.fill(imp, 1.0); }
        else { ((BaseRegressor) estimator).fit(X, y); Arrays.fill(imp, 1.0); }
        return imp;
    }

    private double[][] extractMatrix(DataFrame df, List<String> cols) {
        int n = df.rowCount(); double[][] mat = new double[n][cols.size()];
        for (int j = 0; j < cols.size(); j++) {
            List<?> data = df.column(cols.get(j)).data();
            for (int i = 0; i < n; i++) {
                Object v = data.get(i);
                mat[i][j] = v == null ? 0.0 : DataValues.asDouble(v);
            }
        }
        return mat;
    }

    private double[] extractLabels(DataFrame df, String col) {
        int n = df.rowCount(); double[] y = new double[n];
        List<?> data = df.column(col).data();
        for (int i = 0; i < n; i++) {
            Object v = data.get(i);
            y[i] = v == null ? 0.0 : DataValues.asDouble(v);
        }
        return y;
    }

    public int getOptimalNFeatures()     { return optimalNFeatures; }
    public List<String> getSelectedCols(){ return selectedCols; }
}

