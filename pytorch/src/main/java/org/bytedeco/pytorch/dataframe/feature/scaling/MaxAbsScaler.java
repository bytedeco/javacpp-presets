package org.bytedeco.pytorch.dataframe.feature.scaling;

import org.bytedeco.pytorch.dataframe.DataValues;

 import org.bytedeco.pytorch.dataframe.DataFrame;
  import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 最大绝对值标准化器 (Max Abs Scaler)
 * 通过除以特征的最大绝对值来缩放数据
 * 适合稀疏数据，保留零值
 */
public class MaxAbsScaler extends BaseTransformer implements Serializable {
    private static final long serialVersionUID = 1L; // 必须声明，确保序列化

    private String[] columns;
    private Map<String, Double> maxAbsValues = new HashMap<>();

    public MaxAbsScaler(String... columns) {
        super(columns);
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Object> values = X.column(col).data();
            // ✅ 过滤 null 值
            double maxAbs = values.stream()
                    .filter(v -> v != null)  // 添加 null 检查
                    .mapToDouble(v -> Math.abs(DataValues.asDouble(v)))
                    .max()
                    .orElse(1.0);

            // 避免除以零
            if (maxAbs == 0) {
                maxAbs = 1.0;
            }
            maxAbsValues.put(col, maxAbs);
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
            double maxAbs = maxAbsValues.get(col);
            List<Double> scaled = new ArrayList<>();

            for (Object value : X.column(col).data()) {
                // ✅ 处理 null 值
                if (value == null) {
                    scaled.add(null);
                } else {
                    double v = DataValues.asDouble(value);
                    scaled.add(v / maxAbs);
                }
            }

            result = result.withColumn(col + "_maxabs_scaled", scaled);
        }

        return result;
    }

    /**
     * 获取最大绝对值
     */
    public Map<String, Double> getMaxAbsValues() {
        return new HashMap<>(maxAbsValues);
    }
}
