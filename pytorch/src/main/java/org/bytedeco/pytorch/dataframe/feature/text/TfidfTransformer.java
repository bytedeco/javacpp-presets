package org.bytedeco.pytorch.dataframe.feature.text;

import org.bytedeco.pytorch.dataframe.DataValues;

 import org.bytedeco.pytorch.dataframe.DataFrame;
  import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TF-IDF 转换器 (TfidfTransformer)
 * 将词频矩阵转换为 TF-IDF 加权矩阵
 * 在 TfidfVectorizer 之后使用
 */
public class TfidfTransformer extends BaseTransformer {
    private String[] columns;
    private Map<String, Double> idfValues = new HashMap<>();
    private boolean sublinearTf = false;

    public TfidfTransformer(String... columns) {

        super(columns);
        this.columns = columns;
    }

    public TfidfTransformer(boolean sublinearTf, String... columns) {
        this.sublinearTf = sublinearTf;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 计算 IDF 值
        for (String col : columns) {
            List<Object> values = X.column(col).data();
            int docCount = 0;
            int nonZeroCount = 0;

            for (Object v : values) {
                if (v != null) {
                    docCount++;
                    // ✅ 安全类型检查
                    if (v instanceof Number) {
                        double val = ((Number) v).doubleValue();
                        if (val > 0) {
                            nonZeroCount++;
                        }
                    }
                }
            }

            // IDF = log(N / df) + 1
            double idf = Math.log((double) docCount / Math.max(nonZeroCount, 1)) + 1.0;
            idfValues.put(col, idf);
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
            double idf = idfValues.get(col);
            List<Double> transformed = new ArrayList<>();

            List<Object> colData = X.column(col).data();
            for (Object value : colData) {
                if (value == null) {
                    transformed.add(null);
                } else if ((value != null && !Double.isNaN(DataValues.asDouble(value)))) {
                    double tf =  DataValues.asDouble(value);

                    // 应用子线性 TF 缩放（可选）
                    if (sublinearTf) {
                        tf = 1.0 + Math.log(Math.max(tf, 1e-10));
                    }

                    // TF-IDF = TF × IDF
                    double tfidf = tf * idf;
                    transformed.add(tfidf);
                } else {
                    transformed.add(0.0);
                }
            }

            result = result.withColumn(col + "_tfidf", transformed);
        }

        // L2 归一化
        return normalizeL2(result);
    }

    /**
     * L2 归一化
     */
    private DataFrame normalizeL2(DataFrame X) throws Exception {
        DataFrame result = X.copy();
        int nRows = X.rowCount();

        // 对每一行进行 L2 归一化
        for (int row = 0; row < nRows; row++) {
            // 计算该行的 L2 范数
            double norm = 0;
            for (String col : columns) {
                Object val = X.column(col + "_tfidf").get(row);
                if (val != null && val instanceof Number) {
                    double v = ((Number) val).doubleValue();
                    norm += v * v;
                }
            }
            norm = Math.sqrt(norm);

            // 如果范数为 0，跳过
            if (norm < 1e-10) {
                continue;
            }

            // 归一化该行的所有特征
            final double finalNorm = norm;
            for (String col : columns) {
                List<Double> colData = new ArrayList<>();
                List<Object> originalCol = X.column(col + "_tfidf").data();

                for (int i = 0; i < nRows; i++) {
                    if (i == row) {
                        // 当前行，进行归一化
                        Object val = originalCol.get(i);
                        if (val != null && val instanceof Number) {
                            double v = ((Number) val).doubleValue();
                            colData.add(v / finalNorm);
                        } else {
                            colData.add(0.0);
                        }
                    } else {
                        // 其他行，保持不变
                        Object val = originalCol.get(i);
                        if (val instanceof Double) {
                            colData.add((Double) val);
                        } else if (val instanceof Number) {
                            colData.add(((Number) val).doubleValue());
                        } else {
                            colData.add(0.0);
                        }
                    }
                }

                result = result.withColumn(col + "_tfidf", colData);
            }
        }

        return result;
    }

    /**
     * 获取 IDF 值
     */
    public Map<String, Double> getIdfValues() {
        return new HashMap<>(idfValues);
    }
}


