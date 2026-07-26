package org.bytedeco.pytorch.data.dataframe.dtype;

import org.bytedeco.pytorch.data.dataframe.enums.ColumnType;

/**
 * FLOAT64类型数据类（原有）
 */
public class Float64Data extends AbstractDataValue {
    private final Double value;

    public Float64Data(Double value) {
        this.value = value;
    }

    //    @Override
    public Object getValue() {
        return value;
    }

    @Override
    public String getDataType() {
        return ColumnType.FLOAT64.name();
    }

    @Override
    public Object toArrowCompatible() {
        return null;
    }

    @Override
    public String getShortDesc() {
        return "FLOAT64";
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