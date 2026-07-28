package org.bytedeco.pytorch.dataframe.feature.selection;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.List;
import java.util.Map;

/**
 * Keep top percentile of features by score (sklearn SelectPercentile).
 * Delegates scoring to {@link SelectKBest}.
 */
public class SelectPercentile extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private final int percentile;
    private final String scoreFunc;
    private String labelCol;
    private double[] externalY;
    private SelectKBest delegate;

    public SelectPercentile(int percentile, String... columns) {
        this(percentile, "f_classif", columns);
    }

    public SelectPercentile(int percentile, String scoreFunc, String... columns) {
        super(columns);
        this.percentile = Math.max(1, Math.min(100, percentile));
        this.scoreFunc = scoreFunc;
    }

    public SelectPercentile setLabelCol(String labelCol) {
        this.labelCol = labelCol;
        return this;
    }

    public SelectPercentile fit(DataFrame X, double[] y) {
        this.externalY = y;
        fit(X);
        return this;
    }

    public SelectPercentile fit(DataFrame X, String labelColumn) {
        this.labelCol = labelColumn;
        fit(X);
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        int nFeat = columns.isEmpty()
            ? (int) X.columns().stream().filter(c -> {
                var d = c.dtype();
                return d == org.bytedeco.pytorch.dataframe.Column.DType.FLOAT64
                    || d == org.bytedeco.pytorch.dataframe.Column.DType.FLOAT32
                    || d == org.bytedeco.pytorch.dataframe.Column.DType.INT32
                    || d == org.bytedeco.pytorch.dataframe.Column.DType.INT64;
            }).count()
            : columns.size();
        int k = Math.max(1, (int) Math.ceil(nFeat * percentile / 100.0));
        delegate = new SelectKBest(k, scoreFunc, columns.toArray(new String[0]));
        if (labelCol != null) delegate.setLabelCol(labelCol);
        if (externalY != null) delegate.fit(X, externalY);
        else if (labelCol != null) delegate.fit(X, labelCol);
        else delegate.fit(X);
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        return delegate.transform(X);
    }

    public List<String> getSelectedColumns() {
        return delegate == null ? List.of() : delegate.getSelectedColumns();
    }

    public Map<String, Double> getScores() {
        return delegate == null ? Map.of() : delegate.getScores();
    }
}
