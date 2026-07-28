package org.bytedeco.pytorch.dataframe.feature.text;

import org.bytedeco.pytorch.dataframe.DataValues;

 import org.bytedeco.pytorch.dataframe.DataFrame;
  import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 特征哈希器 (Feature Hasher)
 * 将特征字典转换为固定维度的哈希向量
 * 适合处理高维稀疏数据
 */
public class FeatureHasher extends BaseTransformer {
    private int nFeatures = 256;
    private String[] columns;

    public FeatureHasher(int nFeatures, String... columns) {
        super(columns);
        this.nFeatures = nFeatures;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();
        int nRows = X.rowCount();

        // 初始化哈希向量列
        Map<Integer, List<Double>> hashColumns = new HashMap<>();
        for (int i = 0; i < nFeatures; i++) {
            hashColumns.put(i, new ArrayList<>());
            for (int j = 0; j < nRows; j++) {
                hashColumns.get(i).add(0.0);
            }
        }

        // 对每一行进行哈希
        for (int row = 0; row < nRows; row++) {
            Map<Integer, Double> hashValues = new HashMap<>();

            // 从所有列中收集特征
            for (String col : columns) {
                List<Object> colData = X.column(col).data();
                Object value = colData.get(row);

                if (value != null) {
                    // ✅ 安全类型检查
                    int hash = Math.abs(value.hashCode()) % nFeatures;

                    // 计算特征值
                    double val;
                    if ((value != null && !Double.isNaN(DataValues.asDouble(value)))) {
                        val = DataValues.asDouble(value);
                    } else {
                        val = 1.0;  // 非数值特征默认贡献 1
                    }

                    hashValues.put(hash, hashValues.getOrDefault(hash, 0.0) + val);
                }
            }

            // 更新哈希向量
            for (Map.Entry<Integer, Double> entry : hashValues.entrySet()) {
                int hashIdx = entry.getKey();
                double hashVal = entry.getValue();

                List<Double> colData = hashColumns.get(hashIdx);
                colData.set(row, hashVal);
            }
        }

        // 将所有哈希列添加到结果
        for (int i = 0; i < nFeatures; i++) {
            result = result.withColumn("hash_" + i, hashColumns.get(i));
        }

        return result;
    }

    /**
     * 设置特征数
     */
    public void setNFeatures(int nFeatures) {
        this.nFeatures = nFeatures;
    }
}

