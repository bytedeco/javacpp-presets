package org.bytedeco.pytorch.dataframe.feature.construction;

import org.bytedeco.pytorch.dataframe.DataValues;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.Int32Data;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * 二值化器 (Binarizer)
 * 根据阈值将特征转换为二值特征
 */
public class Binarizer extends BaseTransformer {
    private String[] columns;
    private double threshold = 0.0;

    public Binarizer(double threshold, String... columns) {
        super(columns);
        this.threshold = threshold;
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
//            List<Integer> binarized = new ArrayList<>();
//            for (Object value : X.column(col).data()) {
//                if (value == null) {
//                    binarized.add(0);
//                } else {
//                    double v = ((Number) value).doubleValue();
//                    binarized.add(v > threshold ? 1 : 0);
//                }
//            }
//            result = result.withColumn(col + "_binary", binarized);
//        }

        for (String col : columns) {
            // 1. 修正：初始化 AbstractDataValue 列表（替代原 List<Integer>）
            List<Object> binarized = new ArrayList<>();
            // 2. 遍历列数据，将每个值转换后包装为 Int32Data
            for (Object value : X.column(col).data()) {
                Integer binaryValue;
                if (value == null) {
                    binaryValue = 0; // 空值默认转为0
                } else {
                    double v = DataValues.asDouble(value);
                    binaryValue = v > threshold ? 1 : 0; // 二值化逻辑
                }
                // 核心：将 Integer 包装为 AbstractDataValue 子类（Int32Data）
                binarized.add(new Int32Data(binaryValue));
            }
            // 3. 传入包装后的 AbstractDataValue 列表
            result = result.withColumn(col + "_binary", binarized);
        }

        return result;
    }

    /**
     * 设置阈值
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
}