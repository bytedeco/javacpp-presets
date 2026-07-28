package org.bytedeco.pytorch.dataframe.feature.construction;

import org.bytedeco.pytorch.dataframe.DataValues;

 import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.Int32Data;
 import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * 分位数离散化 (Quantile Discretizer)
 * 基于分位数将连续特征离散化为等频箱
 */
public class QuantileDiscretizer extends BaseTransformer {
    private String[] columns;
    private int nQuantiles;
    private Map<String, double[]> quantileValues = new HashMap<>();
    private Map<String, List<Double>> quantileBoundaries = new HashMap<>();

    public QuantileDiscretizer(int nQuantiles, String... columns) {
        super(columns);
        this.nQuantiles = nQuantiles;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Object> values = X.column(col).data();
            List<Double> doubleValues = new ArrayList<>();

            for (Object v : values) {
                if (v != null) {
                    doubleValues.add(DataValues.asDouble(v));
                }
            }

            if (doubleValues.isEmpty()) {
                continue;
            }

            Collections.sort(doubleValues);

            // 计算分位数
            List<Double> boundaries = new ArrayList<>();
            for (int i = 0; i <= nQuantiles; i++) {
                double percentile = (double) i / nQuantiles;
                int idx = (int) (percentile * (doubleValues.size() - 1));
                boundaries.add(doubleValues.get(Math.min(idx, doubleValues.size() - 1)));
            }

            quantileBoundaries.put(col, boundaries);

            // 计算分位数值
            double[] quantiles = new double[nQuantiles];
            for (int i = 0; i < nQuantiles; i++) {
                quantiles[i] = boundaries.get(i);
            }
            quantileValues.put(col, quantiles);
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

//        for (String col : columns) {
//            List<Double> boundaries = quantileBoundaries.get(col);
//            if (boundaries == null) {
//                continue;
//            }
//
//            List<Integer> discretized = new ArrayList<>();
//
//            for (Object value : X.column(col).data()) {
//                if (value == null) {
//                    discretized.add(-1);
//                } else {
//                    double v = ((Number) value).doubleValue();
//                    int bin = findQuantileBin(v, boundaries);
//                    discretized.add(bin);
//                }
//            }
//
//            result = result.withColumn(col + "_quantile", discretized);
//        }

        for (String col : columns) {
            List<Double> boundaries = quantileBoundaries.get(col);
            if (boundaries == null) {
                continue;
            }

            // 1. 修正：初始化 AbstractDataValue 列表（替代原 List<Integer>）
            List<Object> discretized = new ArrayList<>();

            // 2. 遍历 AbstractDataValue 类型的列数据，分箱后包装为 Int32Data
            for (Object value : X.column(col).data()) {
                Integer binValue;
                if (value == null) {
                    binValue = -1; // 空值标记为-1（保持原有逻辑）
                } else {
                    // 核心优化：通过 toArrowCompatible() 获取原始值（符合 AbstractDataValue 规范）
                    double rawValue = DataValues.asDouble(value);
                    if (Double.isNaN(rawValue)) {
                        throw new IllegalArgumentException(
                                String.format("列 %s 包含非数值类型数据：%s（类型：%s）",
                                        col, rawValue, "double"));
                    }
                    double v = rawValue;
                    binValue = findQuantileBin(v, boundaries);
                }
                // 核心：将 Integer 分箱值包装为 AbstractDataValue 子类
                discretized.add(new Int32Data(binValue));
            }

            // 3. 传入包装后的 AbstractDataValue 列表
            result = result.withColumn(col + "_quantile", discretized);
        }

        return result;
    }

    /**
     * 找到值对应的分位数箱
     */
    private int findQuantileBin(double value, List<Double> boundaries) {
        for (int i = 0; i < boundaries.size() - 1; i++) {
            if (value <= boundaries.get(i + 1)) {
                return i;
            }
        }
        return boundaries.size() - 2;
    }

    /**
     * 获取分位数边界
     */
    public Map<String, List<Double>> getQuantileBoundaries() {
        return new HashMap<>(quantileBoundaries);
    }
}