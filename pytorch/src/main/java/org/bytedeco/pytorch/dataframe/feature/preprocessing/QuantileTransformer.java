package org.bytedeco.pytorch.dataframe.feature.preprocessing;

import org.bytedeco.pytorch.dataframe.DataValues;

 import org.bytedeco.pytorch.dataframe.DataFrame;
  import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * 分位数变换 (Quantile Transformer)
 * 将特征映射到均匀分布或正态分布
 * 对异常值不敏感，保留相对大小关系
 */
public class QuantileTransformer extends BaseTransformer {
    private String[] columns;
    private Map<String, List<Double>> quantiles = new HashMap<>();
    private int nQuantiles = 1000;

    public enum Output {
        UNIFORM,
        NORMAL
    }

    private Output output;

    public QuantileTransformer(String... columns) {

        super(columns);
        this.columns = columns;
        this.output = Output.UNIFORM;
//        this(Output.UNIFORM, columns);
    }

    public QuantileTransformer(Output output, String... columns) {
        this.output = output;
        this.columns = columns;
    }

    public QuantileTransformer(Output output, int nQuantiles, String... columns) {
        this.output = output;
        this.nQuantiles = nQuantiles;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Object> rawValues = X.column(col).data();

            // ✅ 手动过滤 null 并转换为 Double 列表
            List<Double> values = new ArrayList<>();
            for (Object v : rawValues) {
                if (v != null) {
                    values.add(DataValues.asDouble(v));
                }
            }

            // 排序
            Collections.sort(values);

            if (values.isEmpty()) {
                quantiles.put(col, new ArrayList<>());
                continue;
            }

            // 计算分位数
            List<Double> colQuantiles = new ArrayList<>();
            for (int i = 0; i <= nQuantiles; i++) {
                double percentile = (double) i / nQuantiles;
                int idx = (int) (percentile * (values.size() - 1));
                colQuantiles.add(values.get(Math.min(idx, values.size() - 1)));
            }

            quantiles.put(col, colQuantiles);
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
            List<Double> colQuantiles = quantiles.get(col);
            List<Double> transformed = new ArrayList<>();

            for (Object value : X.column(col).data()) {
                // ✅ 处理 null 值
                if (value == null) {
                    transformed.add(null);
                } else {
                    double v = DataValues.asDouble(value);

                    // 找到最近的分位数索引
                    int quantileIdx = findQuantileIndex(v, colQuantiles);
                    double quantile = (double) quantileIdx / nQuantiles;

                    // 映射到输出分布
                    double transformedValue;
                    if (output == Output.UNIFORM) {
                        transformedValue = quantile;
                    } else {
                        // 映射到正态分布
                        transformedValue = inverseCumulativeNormal(quantile);
                    }

                    transformed.add(transformedValue);
                }
            }

            result = result.withColumn(col + "_quantile_transformed", transformed);
        }

        return result;
    }

    /**
     * 找到最近的分位数索引
     */
    private int findQuantileIndex(double value, List<Double> colQuantiles) {
        if (colQuantiles.isEmpty()) return 0;

        for (int i = 0; i < colQuantiles.size() - 1; i++) {
            if (value >= colQuantiles.get(i) && value <= colQuantiles.get(i + 1)) {
                return i;
            }
        }
        return Math.max(0, colQuantiles.size() - 1);
    }

    /**
     * 逆累积正态分布函数 - Acklam 方法
     */
    private double inverseCumulativeNormal(double p) {
        if (p <= 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (p >= 1.0) {
            return Double.POSITIVE_INFINITY;
        }

        double a1 = -3.969683028665376e+01;
        double a2 = 2.209460984245205e+02;
        double a3 = -2.759285104469687e+02;
        double a4 = 1.383577518672690e+02;
        double a5 = -3.066479806614716e+01;
        double a6 = 2.506628277459239e+00;

        double b1 = -5.447609879822406e+01;
        double b2 = 1.615858368580409e+02;
        double b3 = -1.556989798598866e+02;
        double b4 = 6.680131188771972e+01;
        double b5 = -1.328068155288572e+01;

        double c1 = -7.784894002430293e-03;
        double c2 = -3.223964580411365e-01;
        double c3 = -2.400758277161838e+00;
        double c4 = -2.549732539343734e+00;
        double c5 = 4.374664141464968e+00;
        double c6 = 2.938163982698783e+00;

        double d1 = 7.784695709041462e-03;
        double d2 = 3.224671290700398e-01;
        double d3 = 2.445134137142996e+00;
        double d4 = 3.754408661907416e+00;

        double pLow = 0.02425;
        double pHigh = 1.0 - pLow;

        double q, r;

        if (p < pLow) {
            q = Math.sqrt(-2.0 * Math.log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                    ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        if (p > pHigh) {
            q = Math.sqrt(-2.0 * Math.log(1.0 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                    ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        q = p - 0.5;
        r = q * q;

        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }

    /**
     * 获取分位数
     */
    public Map<String, List<Double>> getQuantiles() {
        return new HashMap<>(quantiles);
    }
}

