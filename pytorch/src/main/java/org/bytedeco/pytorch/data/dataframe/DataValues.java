package org.bytedeco.pytorch.data.dataframe;

/**
 * Cell-value helpers shared by DataFrame ops, feature transformers, and ML estimators.
 * Accepts plain {@link Number}/{@link String}/… cells and multimodal {@code DataValue} wrappers.
 */
public final class DataValues {
    private DataValues() {}

    /** Unwrap a multimodal wrapper to its payload; pass-through for plain cells. */
    public static Object unwrap(Object v) {
        if (v == null) return null;
        if (v instanceof org.bytedeco.pytorch.data.dataframe.dtype.DataValue) {
            Object raw = ((org.bytedeco.pytorch.data.dataframe.dtype.DataValue) v).toArrowCompatible();
            return raw;
        }
        return v;
    }

    /** Numeric view of a cell. Null → NaN. Non-numeric wrappers try {@code getNumericValue()}. */
    public static double asDouble(Object v) {
        if (v == null) return Double.NaN;
        if (v instanceof Number) return ((Number) v).doubleValue();
        if (v instanceof Boolean) return ((Boolean) v) ? 1.0 : 0.0;
        if (v instanceof org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue) {
            Number n = ((org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue) v).getNumericValue();
            return n == null ? Double.NaN : n.doubleValue();
        }
        Object u = unwrap(v);
        if (u instanceof Number) return ((Number) u).doubleValue();
        if (u instanceof Boolean) return ((Boolean) u) ? 1.0 : 0.0;
        try {
            return Double.parseDouble(String.valueOf(u));
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    public static String asString(Object v) {
        if (v == null) return null;
        Object u = unwrap(v);
        return u == null ? null : u.toString();
    }

    public static boolean isNull(Object v) {
        return v == null;
    }

    public static boolean isNumericCell(Object v) {
        if (v instanceof Number) return true;
        if (v instanceof org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue) {
            try {
                Number n = ((org.bytedeco.pytorch.data.dataframe.dtype.AbstractDataValue) v).getNumericValue();
                return n != null && !Double.isNaN(n.doubleValue());
            } catch (Exception e) {
                return false;
            }
        }
        return false;
    }
}
