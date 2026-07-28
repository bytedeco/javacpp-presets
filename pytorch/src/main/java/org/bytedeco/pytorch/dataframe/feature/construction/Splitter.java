package org.bytedeco.pytorch.dataframe.feature.construction;

 import org.bytedeco.pytorch.dataframe.DataFrame;
 import org.bytedeco.pytorch.dataframe.dtype.AbstractDataValue;
 import org.bytedeco.pytorch.dataframe.dtype.StringData;
 import org.bytedeco.pytorch.dataframe.dtype.DataValue;
 import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * 分割器 (Splitter)
 * 将字符串特征分割为多个特征
 * 例如："a,b,c" -> ["a", "b", "c"]
 */
public class Splitter extends BaseTransformer {
    private String column;
    private String delimiter;
    private int maxSplits = -1;  // -1 表示无限制
    private List<String> splitColumnNames;

    public Splitter(String column, String delimiter) {
//        super(columns);
        this.column = column;
        this.delimiter = delimiter;
    }

    public Splitter(String column, String delimiter, int maxSplits) {
        this.column = column;
        this.delimiter = delimiter;
        this.maxSplits = maxSplits;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        // 确定最大分割数
        int maxParts = 0;
        List<Object> values = X.column(column).data();

        for (Object value : values) {
            if (value != null) {
                String[] parts = value.toString().split(delimiter, maxSplits < 0 ? -1 : maxSplits + 1);
                maxParts = Math.max(maxParts, parts.length);
            }
        }

        // 生成分割列名
        splitColumnNames = new ArrayList<>();
        for (int i = 0; i < maxParts; i++) {
            splitColumnNames.add(column + "_split_" + i);
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
        List<Object> values = X.column(column).data();

        // 为每个分割部分创建列
//        for (int splitIdx = 0; splitIdx < splitColumnNames.size(); splitIdx++) {
//            List<String> splitValues = new ArrayList<>();
//
//            for (Object value : values) {
//                String str = value != null ? value.toString() : "";
//                String[] parts = str.split(delimiter, maxSplits < 0 ? -1 : maxSplits + 1);
//
//                if (splitIdx < parts.length) {
//                    splitValues.add(parts[splitIdx].trim());
//                } else {
//                    splitValues.add("");
//                }
//            }
//
//            result = result.withColumn(splitColumnNames.get(splitIdx), splitValues);
//        }

        for (int splitIdx = 0; splitIdx < splitColumnNames.size(); splitIdx++) {
            List<Object> splitValues = new ArrayList<>();

            for (Object value : values) {
                String str;
                if (value == null) {
                    str = "";
                } else {
                    // 兼容 AbstractDataValue 类型的原始值（符合统一类型体系）
                    if (value instanceof AbstractDataValue) {
                        Object rawValue = ( (value instanceof DataValue) ? ((DataValue)value).toArrowCompatible() : value );
                        str = rawValue == null ? "" : rawValue.toString();
                    } else {
                        // 增加非字符串类型容错（避免 toString() 异常）
                        try {
                            str = value.toString();
                        } catch (Exception e) {
                            throw new IllegalArgumentException(
                                    String.format("值 %s 无法转换为字符串：%s", value, e.getMessage()), e);
                        }
                    }
                }

                String[] parts = str.split(delimiter, maxSplits < 0 ? -1 : maxSplits + 1);
                String splitValue = splitIdx < parts.length ? parts[splitIdx].trim() : "";
                splitValues.add(new StringData(splitValue));
            }

            result = result.withColumn(splitColumnNames.get(splitIdx), splitValues);
        }

        return result;
    }

    /**
     * 获取分割列名称
     */
    public List<String> getSplitColumnNames() {
        return new ArrayList<>(splitColumnNames);
    }
}
