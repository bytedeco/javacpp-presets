package org.bytedeco.pytorch.data.dataframe.feature.preprocessing;

import org.bytedeco.pytorch.data.dataframe.DataValues;

 import org.bytedeco.pytorch.data.dataframe.DataFrame;
  import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 幂变换 (Power Transformer)
 * 使用 Yeo-Johnson 变换使数据更接近正态分布
 * 适合高度倾斜的数据
 */
public class PowerTransformer extends BaseTransformer {
    private String[] columns;
    private Map<String, Double> lambdas = new HashMap<>();
    private static final double LAMBDA_BOUNDS = 2.0;

    public enum Method {
        YEO_JOHNSON,
        BOX_COX
    }

    private Method method;

    public PowerTransformer(String... columns) {
        super(columns);
        this.method = Method.YEO_JOHNSON;
        this.columns = columns;

    }

    public PowerTransformer(Method method, String... columns) {
        this.method = method;
        this.columns = columns;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Object> values = X.column(col).data();
            double lambda = estimateLambda(values);
            lambdas.put(col, lambda);
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
            double lambda = lambdas.get(col);
            List<Double> transformed = new ArrayList<>();

            for (Object value : X.column(col).data()) {
                double v = DataValues.asDouble(value);
                double transformed_v;

                if (method == Method.YEO_JOHNSON) {
                    transformed_v = yeoJohnsonTransform(v, lambda);
                } else {
                    transformed_v = boxCoxTransform(v, lambda);
                }

                transformed.add(transformed_v);
            }

            result = result.withColumn(col + "_power_transformed", transformed);
        }

        return result;
    }

    /**
     * Yeo-Johnson 变换
     */
    private double yeoJohnsonTransform(double x, double lambda) {
        if (Math.abs(lambda) < 1e-6) {
            return Math.log(x + 1);
        }

        if (x >= 0) {
            return (Math.pow(x + 1, lambda) - 1) / lambda;
        } else {
            return -(Math.pow(-x + 1, 2 - lambda) - 1) / (2 - lambda);
        }
    }

    /**
     * Box-Cox 变换
     */
    private double boxCoxTransform(double x, double lambda) {
        if (x <= 0) {
            throw new IllegalArgumentException("Box-Cox 变换要求正数");
        }

        if (Math.abs(lambda) < 1e-6) {
            return Math.log(x);
        }

        return (Math.pow(x, lambda) - 1) / lambda;
    }

    /**
     * 估计最优的 lambda 参数
     * 使用简化版本，基于峰度和偏度
     */
    private double estimateLambda(List<Object> values) {
        List<Double> doubleValues = new ArrayList<>();
        for (Object v : values) {
            doubleValues.add(DataValues.asDouble(v));
        }

        // 计算偏度
        double skewness = calculateSkewness(doubleValues);

        // 简化：根据偏度估计 lambda
        // 正数偏度：lambda < 1，负数偏度：lambda > 1
        if (Math.abs(skewness) < 0.5) {
            return 1.0;
        } else if (skewness > 0.5) {
            return Math.max(-LAMBDA_BOUNDS, 1.0 - skewness);
        } else {
            return Math.min(LAMBDA_BOUNDS, 1.0 - skewness);
        }
    }

    /**
     * 计算偏度
     */
    private double calculateSkewness(List<Double> values) {
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double variance = values.stream()
                .mapToDouble(v -> (v - mean) * (v - mean))
                .average()
                .orElse(1);
        double std = Math.sqrt(variance);

        if (std == 0) return 0;

        double skewness = values.stream()
                .mapToDouble(v -> Math.pow((v - mean) / std, 3))
                .average()
                .orElse(0);

        return skewness;
    }

    /**
     * 获取 lambda 参数
     */
    public Map<String, Double> getLambdas() {
        return new HashMap<>(lambdas);
    }
}