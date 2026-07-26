package org.bytedeco.pytorch.data.dataframe.feature.preprocessing;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * 缺失值指示器（对应 sklearn MissingIndicator）
 * 为每个含缺失值的列添加 _missing 二值列
 */
public class MissingIndicator extends BaseTransformer {
    private List<String> missingCols;

    public MissingIndicator(String... columns) { super(columns); }
    public MissingIndicator() { super(); }

    @Override
    public BaseTransformer fit(DataFrame X) {
        missingCols = new ArrayList<>();
        List<String> cols = columns.isEmpty() ? new ArrayList<>(X.getColumnNames()) : columns;
        for (String col : cols) {
            List<Object> data = X.column(col).data();
            boolean hasMissing = data.stream().anyMatch(v -> v == null);
            if (hasMissing) missingCols.add(col);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("MissingIndicator not fitted");
        DataFrame result = X.copy();
        for (String col : missingCols) {
            List<Object> data = X.column(col).data();
            List<Double> indicator = new ArrayList<>();
            for (Object v : data) indicator.add(v == null ? 1.0 : 0.0);
            result = result.withColumnForDouble(col + "_missing", indicator);
        }
        return result;
    }

    public List<String> getMissingColumns() { return missingCols; }
}

