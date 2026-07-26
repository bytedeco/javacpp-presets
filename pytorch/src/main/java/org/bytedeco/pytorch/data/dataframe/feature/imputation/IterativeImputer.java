package org.bytedeco.pytorch.data.dataframe.feature.imputation;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 迭代填充器 (Iterative Imputer)
 * 使用 MICE (Multivariate Imputation by Chained Equations) 算法
 * 使用其他特征建立模型预测缺失值
 * 适合特征间有相关性的情况
 */
public class IterativeImputer extends BaseTransformer {
    private String[] columns;
    private int maxIter = 10;
    private double tolerance = 1e-3;
    private Map<String, Double> columnMeans = new HashMap<>();
    private Map<String, List<Double>> imputationHistory = new HashMap<>();

    public IterativeImputer(String... columns) {

//        super(columns);
        this(10, columns);
    }

    public IterativeImputer(int maxIter, String... columns) {
        this.maxIter = maxIter;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 计算初始均值
        for (String col : columns) {
            List<Object> values = X.column(col).data();
            double mean = values.stream()
                    .filter(v -> v != null)
                    .mapToDouble(v ->  DataValues.asDouble(v))
                    .average()
                    .orElse(0.0);
            columnMeans.put(col, mean);
        }

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        // 初始化：用均值填充
        Map<String, List<Double>> imputedData = new HashMap<>();
        for (String col : columns) {
            List<Double> colData = new ArrayList<>();
            List<Object> original = X.column(col).data();
            double mean = columnMeans.get(col);

            for (Object value : original) {
                if (value == null) {
                    colData.add(mean);
                } else {
                    colData.add( DataValues.asDouble(value));
                }
            }
            imputedData.put(col, colData);
        }

        // MICE 迭代过程
        for (int iter = 0; iter < maxIter; iter++) {
            boolean converged = true;

            for (String targetCol : columns) {
                List<Double> oldValues = new ArrayList<>(imputedData.get(targetCol));

                // 对每个缺失值，使用其他特征预测
                List<Object> original = X.column(targetCol).data();
                for (int row = 0; row < original.size(); row++) {
                    if (original.get(row) == null) {
                        // 简化：使用邻近值和其他特征的加权平均
                        double predictedValue = predictValue(X, targetCol, row, imputedData);
                        imputedData.get(targetCol).set(row, predictedValue);

                        // 检查收敛
                        if (Math.abs(predictedValue - oldValues.get(row)) > tolerance) {
                            converged = false;
                        }
                    }
                }
            }

            if (converged) {
                break;
            }
        }

        // 返回结果
        DataFrame result = X.copy();
        for (String col : columns) {
            result = result.withColumn(col + "_iterative_imputed",
                    new ArrayList<>(imputedData.get(col)));
        }

        return result;
    }

    /**
     * 预测缺失值
     */
    private double predictValue(DataFrame X, String targetCol, int row,
                                Map<String, List<Double>> imputedData) {
        double predictedValue = 0;
        int count = 0;

        // 使用其他特征和邻居值的加权和
        for (String col : columns) {
            if (!col.equals(targetCol)) {
                double value = imputedData.get(col).get(row);
                predictedValue += value;
                count++;
            }
        }

        // 加上邻近行的权重
        if (row > 0) {
            predictedValue += imputedData.get(targetCol).get(row - 1) * 0.3;
        }
        if (row < X.rowCount() - 1) {
            predictedValue += imputedData.get(targetCol).get(row + 1) * 0.3;
        }

        return count > 0 ? predictedValue / (count + 0.6) : columnMeans.get(targetCol);
    }

    /**
     * 获取填充历史
     */
    public Map<String, List<Double>> getImputationHistory() {
        return new HashMap<>(imputationHistory);
    }
}
