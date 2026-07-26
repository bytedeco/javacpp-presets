package org.bytedeco.pytorch.data.dataframe.feature.construction;

import org.bytedeco.pytorch.data.dataframe.DataValues;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue;
import org.bytedeco.pytorch.data.dataframe.dtype.Int32Data;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * K-Bins 离散化 (KBins Discretizer)
 * 将连续特征离散化为 K 个等宽或等频的箱
 */
public class KBinsDiscretizer extends BaseTransformer {
    private String[] columns;
    private int nBins;
    private String strategy = "quantile";  // quantile 或 uniform
    private Map<String, double[]> binEdges = new HashMap<>();

//    public KBinsDiscretizer(int nBins, String... columns) {
//        this.nBins = nBins;
//        this.columns = columns;
//    }

    public KBinsDiscretizer(int nBins, String strategy, String... columns) {
        super(columns);
        this.nBins = nBins;
        this.strategy = strategy;
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

            Collections.sort(doubleValues);

            double[] edges;
            if ("uniform".equals(strategy)) {
                edges = computeUniformEdges(doubleValues);
            } else {
                edges = computeQuantileEdges(doubleValues);
            }

            binEdges.put(col, edges);
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
            double[] edges = binEdges.get(col);
            // 1. 修正：初始化 AbstractDataValue 列表（替代原 List<Integer>）
            List<Object> discretized = new ArrayList<>();

            // 2. 遍历列数据，分箱后包装为 Int32Data
            for (Object value : X.column(col).data()) {
                Integer binValue;
                if (value == null) {
                    binValue = -1; // 空值标记为-1（保持原有逻辑）
                } else {
                    // 原有分箱逻辑：Object -> Double -> 计算分箱值
                    double v = DataValues.asDouble(value);
                    binValue = findBin(v, edges);
                }
                // 核心：将 Integer 分箱值包装为 AbstractDataValue 子类
                discretized.add(new Int32Data(binValue));
            }

            // 3. 传入包装后的 AbstractDataValue 列表
            result = result.withColumn(col + "_bin", discretized);
        }
//        for (String col : columns) {
//            double[] edges = binEdges.get(col);
//            List<Integer> discretized = new ArrayList<>();
//
//            for (Object value : X.column(col).data()) {
//                if (value == null) {
//                    discretized.add(-1);
//                } else {
//                    double v = ((Number) value).doubleValue();
//                    int bin = findBin(v, edges);
//                    discretized.add(bin);
//                }
//            }
//
//            result = result.withColumn(col + "_bin", discretized);
//        }

        return result;
    }

    private double[] computeUniformEdges(List<Double> values) {
        double min = values.get(0);
        double max = values.get(values.size() - 1);
        double[] edges = new double[nBins + 1];

        for (int i = 0; i <= nBins; i++) {
            edges[i] = min + (max - min) * i / nBins;
        }

        return edges;
    }

    private double[] computeQuantileEdges(List<Double> values) {
        double[] edges = new double[nBins + 1];
        edges[0] = values.get(0);
        edges[nBins] = values.get(values.size() - 1);

        int step = values.size() / nBins;
        for (int i = 1; i < nBins; i++) {
            edges[i] = values.get(i * step);
        }

        return edges;
    }

    private int findBin(double value, double[] edges) {
        for (int i = 0; i < edges.length - 1; i++) {
            if (value <= edges[i + 1]) {
                return i;
            }
        }
        return edges.length - 2;
    }
}
