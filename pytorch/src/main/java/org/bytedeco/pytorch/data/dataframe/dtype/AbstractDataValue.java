package org.bytedeco.pytorch.data.dataframe.dtype;

/**
 * Base for multimodal cell values with a numeric projection used by ML/feature code.
 */
public abstract class AbstractDataValue implements DataValue {
    private static final long serialVersionUID = 1L;

    /** Numeric projection; non-numeric types may return {@code null} or {@code 0}. */
    public abstract Number getNumericValue();

    @Override
    public String toString() {
        return String.format("[%s] %s", getDataType(), getShortDesc());
    }

    @Override
    public boolean isValid() {
        String t = getDataType();
        String d = getShortDesc();
        return t != null && !t.isEmpty() && d != null && !d.isEmpty();
    }

    @Override
    public abstract String getDataType();

    @Override
    public abstract Object toArrowCompatible();

    @Override
    public abstract String getShortDesc();
}
