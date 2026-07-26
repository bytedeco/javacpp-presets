package org.bytedeco.pytorch.data.dataframe.feature.encoding;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
 import org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue;
 import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 目标编码器 (Target Encoding)
 * 使用目标变量的统计特性来编码类别特征
 * 适用于监督学习任务
 */
public class TargetEncoder extends BaseTransformer {
    private Map<String, Map<Object, Double>> encodings = new HashMap<>();
    private String[] columns;
    private String targetColumn;
    private double smoothing = 1.0;

    public TargetEncoder(String targetColumn, String... columns) {
        super(columns);
        this.targetColumn = targetColumn;
        this.columns = columns;
    }

    public TargetEncoder(String targetColumn, double smoothing, String... columns) {
        this.targetColumn = targetColumn;
        this.smoothing = smoothing;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        List<Object> target = X.column(targetColumn).data();

        // 计算目标变量的全局均值
        double globalMean = target.stream()
//                .mapToDouble(v -> ((Number) v).doubleValue())
                .mapToDouble(v -> DataValues.asDouble(v)) //AbstractDataValue::getNumericValue
                .average()
                .orElse(0.0);

        for (String col : columns) {
            List<Object> column = X.column(col).data();
            Map<Object, Double> encoding = new HashMap<>();

            // 按类别分组计算目标均值
            Map<Object, List<Integer>> groups = new HashMap<>();
            for (int i = 0; i < column.size(); i++) {
                Object key = column.get(i);
                groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
            }

            // 使用 Smoothing 公式计算编码值
            for (Map.Entry<Object, List<Integer>> entry : groups.entrySet()) {
                List<Integer> indices = entry.getValue();
                double categoryMean = indices.stream()
                        .mapToDouble(i -> DataValues.asDouble(target.get(i)))
                        .average()
                        .orElse(globalMean);

                int count = indices.size();
                double smoothedValue = (count * categoryMean + smoothing * globalMean) / (count + smoothing);

                encoding.put(entry.getKey(), smoothedValue);
            }

            encodings.put(col, encoding);
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
            Map<Object, Double> encoding = encodings.get(col);
            List<Double> encoded = new ArrayList<>();

            for (Object value : X.column(col).data()) {
                encoded.add(encoding.getOrDefault(value, 0.0));
            }

            result = result.withColumn(col + "_target_encoded", encoded);
        }

        return result;
    }

    /**
     * 获取编码映射
     */
    public Map<Object, Double> getEncoding(String column) {
        return encodings.get(column);
    }
}
