package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;

/**
 * INT32类型数据类（修正后必须存在）
 */
public class Int32Data extends AbstractDataValue {
    private final Integer value;

    public Int32Data(Integer value) {
        this.value = value;
    }


    @Override
    public String toString() {
        return value != null ? value.toString() : "null";
    }

    public Object getValue() {
        return value;
    }

    @Override
    public String getDataType() {
        return ColumnType.INT32.name();
    }

    @Override
    public Object toArrowCompatible() {
        return null;
    }

    @Override
    public String getShortDesc() {
        return "INT32";
    }

    @Override
    public Integer getNumericValue() {
        return this.value;
    }
}





