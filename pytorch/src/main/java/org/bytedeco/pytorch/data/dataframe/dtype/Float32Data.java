package org.bytedeco.pytorch.data.dataframe.dtype;

import org.bytedeco.pytorch.data.dataframe.enums.ColumnType;

/**
 * FLOAT32类型数据类（原有）
 */
public class Float32Data extends AbstractDataValue {
    private final Float value;

    public Float32Data(Float value) {
        this.value = value;
    }

//    @Override
    public Object getValue() {
        return value;
    }

    @Override
    public String getDataType() {
        return ColumnType.FLOAT32.name();
    }

    @Override
    public Object toArrowCompatible() {
        return null;
    }

    @Override
    public String getShortDesc() {
        return "FLOAT32";
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