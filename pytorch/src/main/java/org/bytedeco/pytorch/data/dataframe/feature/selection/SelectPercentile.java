package org.bytedeco.pytorch.data.dataframe.feature.selection;

import org.bytedeco.pytorch.data.dataframe.DataValues;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 按百分比选择最佳特征（对应 sklearn SelectPercentile）
 */
public class SelectPercentile extends BaseTransformer {
    private final int percentile;       // 保留前 percentile% 的特征
    private final String scoreFunc;     // "f_score" | "chi2" | "mutual_info"
    private List<String> selectedCols;
    private Map<String, Double> scores;

    public SelectPercentile(int percentile, String scoreFunc, String... columns) {
        super(columns);
        this.percentile = percentile;
        this.scoreFunc = scoreFunc;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        scores = new LinkedHashMap<>();
        for (String col : columns) {
            List<Object> vals = X.column(col).data();
            scores.put(col, computeScore(vals));
        }
        int k = Math.max(1, (int) Math.ceil(columns.size() * percentile / 100.0));
        selectedCols = scores.entrySet().stream()
            .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
            .limit(k)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("SelectPercentile not fitted");
        return X.select(selectedCols.toArray(new String[0]));
    }

    private double computeScore(List<Object> vals) {
        double mean = vals.stream().filter(v -> v != null)
            .mapToDouble(v -> DataValues.asDouble(v)).average().orElse(0);
        return vals.stream().filter(v -> v != null)
            .mapToDouble(v -> Math.pow(DataValues.asDouble(v) - mean, 2))
            .average().orElse(0);
    }

    public List<String> getSelectedColumns() { return selectedCols; }
    public Map<String, Double> getScores()   { return scores; }
}

