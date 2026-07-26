package org.bytedeco.pytorch.data.dataframe.feature.imputation;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * KNN 填充器 (KNN Imputer)
 * 使用 K 个最近邻的平均值填充缺失值
 * 保留数据的结构，适合保持样本间的相似性
 */
public class KNNImputer extends BaseTransformer {
    private String[] columns;
    private int nNeighbors = 5;
    private Map<String, List<Double>> columnMeans = new HashMap<>();

    public KNNImputer(String... columns) {
        super(columns);
        this.nNeighbors = 5;
        this.columns = columns;
//        this(5, columns);
    }

    public KNNImputer(int nNeighbors, String... columns) {
        this.nNeighbors = nNeighbors;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 计算每列的均值作为后备
        for (String col : columns) {
            List<Object> values = X.column(col).data();
            double mean = values.stream()
                    .filter(v -> v != null)
                    .mapToDouble(v ->  DataValues.asDouble(v))
                    .average()
                    .orElse(0.0);
            columnMeans.put(col, Arrays.asList(mean));
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
            List<Object> originalCol = X.column(col).data();
            List<Double> imputed = new ArrayList<>();

            for (int i = 0; i < originalCol.size(); i++) {
                Object value = originalCol.get(i);

                if (value == null) {
                    // 找到最近的 K 个邻居（有值的行）
                    double imputedValue = findKNNMean(X, col, i);
                    imputed.add(imputedValue);
                } else {
                    imputed.add(DataValues.asDouble(value));
                }
            }

            result = result.withColumn(col + "_knn_imputed", imputed);
        }

        return result;
    }

    /**
     * 找到 K 个最近邻的均值
     */
    private double findKNNMean(DataFrame X, String targetCol, int rowIdx) {
        // 计算与其他行的距离
        List<Integer> distances = new ArrayList<>();
        List<Object> targetCol_data = X.column(targetCol).data();

        for (int i = 0; i < X.rowCount(); i++) {
            if (i == rowIdx) continue;
            if (targetCol_data.get(i) == null) continue;

            // 计算欧氏距离
            double distance = 0;
            for (String col : columns) {
                Object val1 = X.column(col).get(rowIdx);
                Object val2 = X.column(col).get(i);

                if (val1 != null && val2 != null) {
                    double diff = DataValues.asDouble(val1) - DataValues.asDouble(val2);
                    distance += diff * diff;
                }
            }

            distances.add(i);
        }

        // 排序距离
        distances.sort((a, b) -> {
            double distA = computeDistance(X, rowIdx, a);
            double distB = computeDistance(X, rowIdx, b);
            return Double.compare(distA, distB);
        });

        // 取最近的 K 个邻居的平均值
        int k = Math.min(nNeighbors, distances.size());
        double sum = 0;
        for (int i = 0; i < k; i++) {
            int neighborIdx = distances.get(i);
            Object value = X.column(targetCol).get(neighborIdx);
            if (value != null) {
                sum += DataValues.asDouble(value);
            }
        }

        return k > 0 ? sum / k : 0.0;
    }

    /**
     * 计算两行之间的欧氏距离
     */
    private double computeDistance(DataFrame X, int row1, int row2) {
        double distance = 0;
        for (String col : columns) {
            Object val1 = X.column(col).get(row1);
            Object val2 = X.column(col).get(row2);

            if (val1 != null && val2 != null) {
                double diff = DataValues.asDouble(val1) - DataValues.asDouble(val2);
                distance += diff * diff;
            }
        }
        return Math.sqrt(distance);
    }
}
