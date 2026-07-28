package org.bytedeco.pytorch.dataframe.feature.scaling;

 import org.bytedeco.pytorch.dataframe.DataFrame;
 import org.bytedeco.pytorch.dataframe.enums.NormType;
 import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * 归一化器 (Normalizer)
 * 对每个样本（行）进行单位长度归一化
 * 常用范数：L1, L2, Max
 */
public class Normalizer extends BaseTransformer implements Serializable {
    private static final long serialVersionUID = 1L; // 必须声明，确保序列化

    private String[] columns;

//    public enum Norm {
//        L1,      // 曼哈顿距离
//        L2,      // 欧几里得距离
//        MAX      // 最大值
//    }

    private NormType norm;

    public Normalizer(String... columns) {

        super(columns);
        this.columns = columns;
        this.norm = NormType.L2;
//        this(NormType.L2, columns);
    }

    public Normalizer(NormType norm, String... columns) {
        this.norm = norm;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 归一化器不需要拟合参数
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();

        // 实现方式：一次性计算所有归一化值
        for (String col : columns) {
            List<Double> normalizedCol = new ArrayList<>();

            for (int row = 0; row < X.rowCount(); row++) {
                List<Double> rowValues = new ArrayList<>();

                // ✅ 安全地提取行值，过滤 null
                for (String c : columns) {
                    try {
                        Object val = X.column(c).get(row);
                        if (val != null) {
                            rowValues.add(((Number) val).doubleValue());
                        } else {
                            rowValues.add(0.0);
                        }
                    } catch (Exception e) {
                        // 如果列不存在或其他异常，使用 0
                        rowValues.add(0.0);
                    }
                }

                double normValue = computeNorm(rowValues);

                // ✅ 安全地获取原始值
                double normalized;
                try {
                    Object originalVal = X.column(col).get(row);
                    if (originalVal == null) {
                        normalized = 0.0;
                    } else if (normValue > 1e-10) {
                        double v = ((Number) originalVal).doubleValue();
                        normalized = v / normValue;
                    } else {
                        normalized = 0.0;
                    }
                } catch (Exception e) {
                    normalized = 0.0;
                }

                normalizedCol.add(normalized);
            }

            result = result.withColumn(col + "_normalized", normalizedCol);
        }

        return result;
    }

    /**
     * 计算范数
     */
    private double computeNorm(List<Double> values) {
        if (values.isEmpty()) {
            return 1.0;
        }

        switch (norm) {
            case L1:
                return values.stream()
                        .mapToDouble(Math::abs)
                        .sum();

            case L2:
                double l2Norm = Math.sqrt(values.stream()
                        .mapToDouble(v -> v * v)
                        .sum());
                return l2Norm > 0 ? l2Norm : 1.0;

            case MAX:
                double maxNorm = values.stream()
                        .mapToDouble(Math::abs)
                        .max()
                        .orElse(1.0);
                return maxNorm > 0 ? maxNorm : 1.0;

            default:
                return 1.0;
        }
    }
}

