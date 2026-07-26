package org.bytedeco.pytorch.data.dataframe.dtype;

import org.bytedeco.pytorch.data.dataframe.enums.ColumnType;

/**
 * INT64类型数据类（原有）
 */
public class Int64Data extends AbstractDataValue {
    private final Long value;

    public Int64Data(Long value) {
        this.value = value;
    }

//    @Override
    public Object getValue() {
        return value;
    }

    @Override
    public String getDataType() {
        return ColumnType.INT64.name();
    }

    @Override
    public Object toArrowCompatible() {
        return null;
    }

    @Override
    public String getShortDesc() {
        return "INT64";
    }

    @Override
    public Number getNumericValue() {
        return this.value;
    }

    @Override
    public String toString() {
        return value != null ? value.toString() : "null";
    }

}