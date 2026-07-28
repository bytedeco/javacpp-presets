package org.bytedeco.pytorch.dataframe.feature.encoding;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Ordinal encode categorical columns (sklearn OrdinalEncoder-style).
 * Supports {@code handle_unknown=use_encoded_value} with {@code unknown_value}.
 */
public class OrdinalEncoder extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private final Map<String, List<Object>> categories = new LinkedHashMap<>();
    /** "error" | "use_encoded_value" */
    private String handleUnknown = "use_encoded_value";
    private Object unknownValue = -1;
    /** When true, replace original columns; otherwise append {@code col_ordinal}. */
    private boolean replace = true;

    public OrdinalEncoder(String... columns) {
        super(columns);
    }

    public OrdinalEncoder setHandleUnknown(String handleUnknown) {
        this.handleUnknown = handleUnknown == null
            ? "use_encoded_value"
            : handleUnknown.toLowerCase(Locale.ROOT);
        return this;
    }

    public OrdinalEncoder setUnknownValue(Object unknownValue) {
        this.unknownValue = unknownValue;
        return this;
    }

    public OrdinalEncoder setReplace(boolean replace) {
        this.replace = replace;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        categories.clear();
        for (String col : columns) {
            LinkedHashSet<Object> uniques = new LinkedHashSet<>();
            Column c = X.column(col);
            for (int i = 0; i < c.size(); i++) {
                Object v = DataValues.unwrap(c.get(i));
                if (v != null) uniques.add(v);
            }
            categories.put(col, new ArrayList<>(uniques));
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) {
        requireFitted();
        try {
            DataFrame result = X.copy();
            for (String col : columns) {
                List<Object> categoryList = categories.get(col);
                if (categoryList == null) continue;
                String outName = replace ? col : col + "_ordinal";
                if (!replace) {
                    if (result.hasColumn(outName)) result.removeColumn(outName);
                    result.addColumn(outName, Column.DType.FLOAT64);
                    Column oc = result.column(outName);
                    while (oc.size() < result.rowCount()) oc.add(null);
                }
                Column src = X.column(col);
                Column dst = result.column(outName);
                for (int i = 0; i < result.rowCount(); i++) {
                    Object value = DataValues.unwrap(src.get(i));
                    if (value == null) {
                        dst.set(i, unknownValue);
                        continue;
                    }
                    int index = categoryList.indexOf(value);
                    if (index >= 0) {
                        dst.set(i, index);
                    } else if ("error".equals(handleUnknown)) {
                        throw new IllegalArgumentException(
                            "OrdinalEncoder: unknown category '" + value + "' in column " + col);
                    } else {
                        dst.set(i, unknownValue);
                    }
                }
            }
            return result;
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("OrdinalEncoder transform failed", e);
        }
    }

    public Map<String, List<Object>> getCategories() {
        return categories;
    }

    public String getHandleUnknown() { return handleUnknown; }
    public Object getUnknownValue() { return unknownValue; }
}
