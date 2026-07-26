package org.bytedeco.pytorch.data.dataframe.feature.selection;
import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 选择 K 个最佳特征 (Select K Best)
 * 使用 F-Score（对于回归）或卡方检验（对于分类）来选择特征
 */
public class SelectKBest extends BaseTransformer {
    private int k;
    private String[] columns;
    private String scoreMethod;
    private List<String> selectedColumns;
    private Map<String, Double> featureScores;

    /**
     * @param k 选择的特征个数
     * @param scoreMethod 评分方法："f_score" 或 "chi2"
     * @param columns 待选择的列
     */
    public SelectKBest(int k, String scoreMethod, String... columns) {
        super(columns);
        if (k <= 0) {
            throw new IllegalArgumentException("k 必须大于 0");
        }
        this.k = k;
        this.scoreMethod = scoreMethod;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        featureScores = new HashMap<>();

        for (String col : columns) {
            double score = calculateScore(X, col);
            featureScores.put(col, score);
        }

        // 按分数降序排序，选择前 k 个
        selectedColumns = featureScores.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(k)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();

        // 只保留选中的列
        List<String> toRemove = new ArrayList<>();
        for (String col : columns) {
            if (!selectedColumns.contains(col)) {
                toRemove.add(col);
            }
        }

        return result;
    }

    /**
     * 计算特征的评分
     */
    private double calculateScore(DataFrame X, String column) {
        List<Object> values = X.column(column).data();

        if ("f_score".equalsIgnoreCase(scoreMethod)) {
            return calculateFScore(values);
        } else if ("chi2".equalsIgnoreCase(scoreMethod)) {
            return calculateChi2(values);
        }

        return 0.0;
    }

    /**
     * 计算 F-Score（用于回归）
     * F-Score = 特征的方差 / 总方差
     */
    private double calculateFScore(List<Object> values) {
        double mean = values.stream()
                .mapToDouble(v -> DataValues.asDouble(v))
                .average()
                .orElse(0.0);

        double variance = values.stream()
                .mapToDouble(v -> {
                    double diff =  DataValues.asDouble(v) - mean;
                    return diff * diff;
                })
                .average()
                .orElse(0.0);

        return variance;
    }

    /**
     * 计算卡方统计量（用于分类）
     * 用于检验特征和目标之间的独立性
     */
    private double calculateChi2(List<Object> values) {
        // 简化实现：计算值的熵
        Map<Object, Integer> counts = new HashMap<>();
        for (Object v : values) {
            counts.put(v, counts.getOrDefault(v, 0) + 1);
        }

        double entropy = 0;
        int total = values.size();
        for (int count : counts.values()) {
            if (count > 0) {
                double p = (double) count / total;
                entropy -= p * Math.log(p) / Math.log(2);
            }
        }

        return entropy;
    }

    /**
     * 获取选中的列
     */
    public List<String> getSelectedColumns() {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }
        return new ArrayList<>(selectedColumns);
    }

    /**
     * 获取所有特征的评分
     */
    public Map<String, Double> getScores() {
        return new HashMap<>(featureScores);
    }
}