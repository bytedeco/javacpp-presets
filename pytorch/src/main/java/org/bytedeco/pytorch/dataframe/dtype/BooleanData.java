package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;

/**
 * BOOLEAN类型数据类（补充）
 */
public class BooleanData extends AbstractDataValue {
    private final Boolean value;

    public BooleanData(Boolean value) {
        this.value = value;
    }

//    @Override
    public Object getValue() {
        return value;
    }

    @Override
    public String getDataType() {
        return ColumnType.BOOLEAN.name();
    }

    @Override
    public Object toArrowCompatible() {
        return null;
    }

    @Override
    public String getShortDesc() {
        return "BOOLEAN";
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    @Override
    public String toString() {
        return value != null ? value.toString() : "null";
    }
}