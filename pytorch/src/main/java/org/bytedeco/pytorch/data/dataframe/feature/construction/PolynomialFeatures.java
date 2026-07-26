package org.bytedeco.pytorch.data.dataframe.feature.construction;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
 import org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue;
 import org.bytedeco.pytorch.data.dataframe.dtype.Float64Data;
 import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * 多项式特征 (Polynomial Features)
 * 生成多项式特征和交互特征
 * 例如：[a, b] -> [1, a, b, a², ab, b²]
 */
public class PolynomialFeatures extends BaseTransformer {
    private int degree = 2;
    private String[] columns;
    private boolean includesBias = true;
    private List<String> featureNames;

    public PolynomialFeatures(int degree, String... columns) {
        super(columns);
        this.degree = degree;
        this.columns = columns;
    }

    public PolynomialFeatures(int degree, boolean includesBias, String... columns) {
        this.degree = degree;
        this.includesBias = includesBias;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 生成特征名称
        featureNames = new ArrayList<>();

        if (includesBias) {
            featureNames.add("bias");
        }

        // 生成所有多项式特征组合
        generatePolynomialNames(columns, 0, "", 0);

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) {
            throw new IllegalStateException("转换器未拟合");
        }

        DataFrame result = X.copy();

        // 添加偏置项
        if (includesBias) {
            List<Double> bias = new ArrayList<>();
            for (int i = 0; i < X.rowCount(); i++) {
                bias.add(1.0);
            }
            // 2. 核心修正：将 Double 列表转换为 AbstractDataValue 列表
            List<Object> transformedData = new ArrayList<>();
            for (Double d : bias) {
                transformedData.add(new Float64Data(d)); // 包装为 Float64Data
            }
            result = result.withColumn("bias", transformedData);
//            result = result.withColumn("bias", bias);
        }

        // 原始特征
//        for (String col : columns) {
//            List<Double> features = new ArrayList<>();
//            for (Object val : X.column(col).data()) {
//                features.add(val != null ? ((Number) val).doubleValue() : 0);
//            }
//            result = result.withColumn(col, features);
//        }
        for (String col : columns) {
            // 1. 初始化 AbstractDataValue 列表（替代原 List<Double>）
            List<Object> features = new ArrayList<>();
            // 2. 遍历列数据，将每个值包装为 Float64Data
            for (Object val : X.column(col).data()) {
                if (val != null) {
                    // 将 AbstractDataValue 转为 Double 后，包装为 Float64Data
                    double value = DataValues.asDouble(val);
                    features.add(new Float64Data(value));
                } else {
                    // 空值包装为 Float64Data(null) 或 0.0（根据你的业务需求选择）
                    features.add(new Float64Data(0.0)); // 或 new Float64Data(null)
                }
            }
            // 3. 传入包装后的 AbstractDataValue 列表
            result = result.withColumn(col, features);
        }

        // 多项式特征
        generatePolynomialFeatures(X, result, columns, 0, new double[columns.length], "");

        return result;
    }

    private void generatePolynomialNames(String[] cols, int idx, String prefix, int currentDegree) {
        if (currentDegree > 0) {
            featureNames.add(prefix.trim());
        }

        if (currentDegree >= degree) {
            return;
        }

        for (int i = idx; i < cols.length; i++) {
            generatePolynomialNames(cols, i, prefix + cols[i] + " ", currentDegree + 1);
        }
    }

    private void generatePolynomialFeatures(DataFrame X, DataFrame result,
                                            String[] cols, int idx, double[] powers, String featureName) throws Exception {
        // 简化实现：只生成二次特征
//        if (degree >= 2 && idx < cols.length - 1) {
//            for (int i = idx + 1; i < cols.length; i++) {
//                List<Double> interaction = new ArrayList<>();
//                for (int row = 0; row < X.rowCount(); row++) {
//                    Object val1 = X.column(cols[idx]).get(row);
//                    Object val2 = X.column(cols[i]).get(row);
//                    double v1 = val1 != null ? ((Number) val1).doubleValue() : 0;
//                    double v2 = val2 != null ? ((Number) val2).doubleValue() : 0;
//                    interaction.add(v1 * v2);
//                }
//                result = result.withColumn(cols[idx] + "_x_" + cols[i], interaction);
//            }
//        }

        if (degree >= 2 && idx < cols.length - 1) {
            for (int i = idx + 1; i < cols.length; i++) {
                List<Object> interaction = new ArrayList<>();

                for (int row = 0; row < X.rowCount(); row++) {
                    Object val1 = X.column(cols[idx]).get(row);
                    Object val2 = X.column(cols[i]).get(row);

                    // 计算交互值（增加类型安全校验）
                    double v1 = 0.0;
                    if (val1 != null) {
//                        if (!((val1 != null && !Double.isNaN(DataValues.asDouble(val1))))) {
//                            throw new IllegalArgumentException(
//                                    String.format("列 %s 行 %d 包含非数值类型数据：%s（类型：%s）",
//                                            cols[idx], row, val1, val1.getClass().getSimpleName()));
//                        }
                        // 兼容 AbstractDataValue 类型的原始值
                        if (val1 instanceof AbstractDataValue) {
                            v1 =  DataValues.asDouble(val1);
                        } else {
                            v1 = DataValues.asDouble(val1);
                        }
                    }

                    double v2 = 0.0;
                    if (val2 != null) {
//                        if (!(val2 instanceof Number)) {
//                            throw new IllegalArgumentException(
//                                    String.format("列 %s 行 %d 包含非数值类型数据：%s（类型：%s）",
//                                            cols[i], row, val2, val2.getClass().getSimpleName()));
//                        }
//                        // 兼容 AbstractDataValue 类型的原始值
                        if (val2 instanceof AbstractDataValue) {
                            v2 = ( DataValues.asDouble(val2));
                        } else {
                            v2 = DataValues.asDouble(val2);
                        }
                    }

                    double interactionValue = v1 * v2;
                    interaction.add(new Float64Data(interactionValue));
                }

                result = result.withColumn(cols[idx] + "_x_" + cols[i], interaction);
            }
        }
    }

    /**
     * 获取特征名称
     */
    public List<String> getFeatureNames() {
        return new ArrayList<>(featureNames);
    }
}