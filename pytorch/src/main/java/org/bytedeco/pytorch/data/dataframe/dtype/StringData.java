package org.bytedeco.pytorch.data.dataframe.dtype;

import org.bytedeco.pytorch.data.dataframe.enums.ColumnType;

/**
 * String 类型数据类，适配统一的 AbstractDataValue 体系
 */
public class StringData extends AbstractDataValue {
    private static final long serialVersionUID = 1L;
    private final String value;

    public StringData(String value) {
        this.value = value; // 空字符串直接保留，符合原有逻辑
    }
    @Override
    public Number getNumericValue(){
        return null;
    }
    @Override
    public String getDataType() {
        return ColumnType.STRING.name();
    }

    @Override
    public Object toArrowCompatible() {
        return value;
    }

    @Override
    public String getShortDesc() {
        return value == null ? "null" : value;
    }

    // 可选：获取原始 String 值
    public String getValue() {
        return value;
    }

    @Override
    public String toString() {
        return value != null ? value.toString() : "null";
    }
}