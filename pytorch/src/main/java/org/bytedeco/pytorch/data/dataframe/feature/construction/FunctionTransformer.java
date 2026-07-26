package org.bytedeco.pytorch.data.dataframe.feature.construction;
import org.bytedeco.pytorch.data.dataframe.DataValues;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue;
import org.bytedeco.pytorch.data.dataframe.dtype.Float64Data;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * 函数转换器 (Function Transformer)
 * 对特征应用自定义函数
 */
public class FunctionTransformer extends BaseTransformer {
    private String[] columns;
    private Function<Double, Double> function;
    private String functionName;

    public FunctionTransformer(Function<Double, Double> function, String functionName, String... columns) {
        super(columns);
        this.function = function;
        this.functionName = functionName;
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

//        for (String col : columns) {
//            List<Double> transformed = new ArrayList<>();
//            for (Object value : X.column(col).data()) {
//                if (value == null) {
//                    transformed.add(null);
//                } else {
//                    double v = ((Number) value).doubleValue();
//                    transformed.add(function.apply(v));
//                }
//            }
//            result = result.withColumn(col + "_" + functionName, transformed);
//        }

        for (String col : columns) {
            List<Object> transformed = new ArrayList<>();
            for (Object value : X.column(col).data()) {
                Double transformedValue;
                if (value == null) {
                    transformedValue = null;
                } else {
                    // 增加类型校验，避免强转失败
                    if (!((value != null && !Double.isNaN(DataValues.asDouble(value))))) {
                        throw new IllegalArgumentException(
                                String.format("列 %s 包含非数值类型数据：%s（类型：%s）",
                                        col, value, value.getClass().getSimpleName()));
                    }
                    // 兼容 AbstractDataValue 类型的原始值（如果 getData() 返回的是 AbstractDataValue）
                    double v;
                    if (value instanceof AbstractDataValue) {
                        v = (DataValues.asDouble(value));
                    } else {
                        v = DataValues.asDouble(value);
                    }
                    transformedValue = function.apply(v);
                }
                transformed.add(new Float64Data(transformedValue));
            }
            result = result.withColumn(col + "_" + functionName, transformed);
        }

        return result;
    }
}