
package org.bytedeco.pytorch.data.dataframe.feature.scaling;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * 鲁棒标准化器 (Robust Scaler)
 * 使用中位数和四分位数范围来标准化特征
 * 对离群值不敏感，比 StandardScaler 更加鲁棒
 */
public class RobustScaler extends BaseTransformer {
    private String[] columns;
    private Map<String, Double> medians = new HashMap<>();
    private Map<String, Double> q1Values = new HashMap<>();
    private Map<String, Double> q3Values = new HashMap<>();

    public RobustScaler(String... columns) {

        super(columns);
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            try {
                List<Object> rawValues = X.column(col).data();

                // ✅ 完全避免 stream，手动循环过滤 null
                List<Double> values = new ArrayList<>();
                for (Object v : rawValues) {
                    // 安全检查
                    if (v != null) {
                        try {
                            values.add(((Number) v).doubleValue());
                        } catch (ClassCastException e) {
                            // 跳过无法转换为 Number 的值
                            continue;
                        }
                    }
                }

                // 排序
                if (!values.isEmpty()) {
                    Collections.sort(values);

                    // 计算中位数
                    medians.put(col, calculateMedian(values));

                    // 计算 Q1 (25 百分位)
                    q1Values.put(col, calculatePercentile(values, 0.25));

                    // 计算 Q3 (75 百分位)
                    q3Values.put(col, calculatePercentile(values, 0.75));
                } else {
                    // 如果列全是 null，使用默认值
                    medians.put(col, 0.0);
                    q1Values.put(col, 0.0);
                    q3Values.put(col, 0.0);
                }
            } catch (Exception e) {
                System.err.printf("警告: 处理列 %s 时出错: %s%n", col, e.getMessage());
                medians.put(col, 0.0);
                q1Values.put(col, 0.0);
                q3Values.put(col, 0.0);
            }
        }

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();

        for (String col : columns) {
            double median = medians.getOrDefault(col, 0.0);
            double q1 = q1Values.getOrDefault(col, 0.0);
            double q3 = q3Values.getOrDefault(col, 0.0);
            double iqr = q3 - q1;

            // 避免 IQR 为 0
            if (iqr == 0 || iqr < 1e-10) {
                iqr = 1.0;
            }

            List<Double> scaled = new ArrayList<>();
            List<Object> colData = X.column(col).data();

            for (Object value : colData) {
                // ✅ 安全处理 null 值
                if (value == null) {
                    scaled.add(null);
                } else {
                    try {
                        double v = ((Number) value).doubleValue();
                        scaled.add((v - median) / iqr);
                    } catch (ClassCastException e) {
                        scaled.add(null);
                    }
                }
            }

            result = result.withColumn(col + "_robust_scaled", scaled);
        }

        return result;
    }

    /**
     * 计算中位数
     */
    private double calculateMedian(List<Double> sortedValues) {
        if (sortedValues.isEmpty()) return 0.0;

        int size = sortedValues.size();
        if (size % 2 == 0) {
            return (sortedValues.get(size / 2 - 1) + sortedValues.get(size / 2)) / 2.0;
        } else {
            return sortedValues.get(size / 2);
        }
    }

    /**
     * 计算百分位数
     */
    private double calculatePercentile(List<Double> sortedValues, double percentile) {
        if (sortedValues.isEmpty()) return 0.0;

        int size = sortedValues.size();
        double index = percentile * (size - 1);
        int lower = (int) Math.floor(index);
        int upper = (int) Math.ceil(index);

        lower = Math.max(0, Math.min(lower, size - 1));
        upper = Math.max(0, Math.min(upper, size - 1));

        if (lower == upper) {
            return sortedValues.get(lower);
        }

        double lowerValue = sortedValues.get(lower);
        double upperValue = sortedValues.get(upper);
        double weight = index - lower;

        return lowerValue + weight * (upperValue - lowerValue);
    }

    /**
     * 获取中位数
     */
    public Map<String, Double> getMedians() {
        return new HashMap<>(medians);
    }

    /**
     * 获取四分位数范围
     */
    public Map<String, Double> getIQRs() {
        Map<String, Double> iqrs = new HashMap<>();
        for (String col : columns) {
            double q1 = q1Values.getOrDefault(col, 0.0);
            double q3 = q3Values.getOrDefault(col, 0.0);
            iqrs.put(col, q3 - q1);
        }
        return iqrs;
    }
}
