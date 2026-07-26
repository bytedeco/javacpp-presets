package org.bytedeco.pytorch.data.dataframe.feature.encoding;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.DataValues;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * One-hot encode categorical columns into INT32 0/1 indicator columns.
 * By default the original column is dropped.
 */
public class OneHotEncoder extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    private final boolean dropOriginal;
    private final String prefix;
    /** Fitted categories per input column. */
    private final java.util.Map<String, List<String>> categories = new java.util.LinkedHashMap<>();

    public OneHotEncoder(String... columns) {
        this(true, null, columns);
    }

    public OneHotEncoder(boolean dropOriginal, String prefix, String... columns) {
        super(columns);
        this.dropOriginal = dropOriginal;
        this.prefix = prefix;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        categories.clear();
        for (String col : columns) {
            LinkedHashSet<String> uniques = new LinkedHashSet<>();
            Column c = X.column(col);
            for (int i = 0; i < c.size(); i++) {
                Object v = DataValues.unwrap(c.get(i));
                uniques.add(v == null ? "null" : v.toString());
            }
            categories.put(col, new ArrayList<>(uniques));
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (String col : columns) {
            List<String> cats = categories.get(col);
            if (cats == null) continue;
            String pfx = prefix == null ? col : prefix;
            List<String> dummyNames = new ArrayList<>();
            for (String u : cats) {
                String name = pfx + "_" + u;
                String base = name;
                int n = 1;
                while (result.hasColumn(name)) name = base + "_" + (n++);
                result.addColumn(name, Column.DType.INT32);
                Column dcol = result.column(name);
                while (dcol.size() < result.rowCount()) dcol.add(0);
                for (int i = 0; i < result.rowCount(); i++) dcol.set(i, 0);
                dummyNames.add(name);
            }
            Column src = X.column(col);
            for (int i = 0; i < result.rowCount(); i++) {
                Object raw = DataValues.unwrap(src.get(i));
                String v = raw == null ? "null" : raw.toString();
                String target = pfx + "_" + v;
                for (String dn : dummyNames) {
                    // handle uniquified names: exact match preferred
                    boolean hit = dn.equals(target);
                    if (!hit && dn.startsWith(pfx + "_")) {
                        // if uniquified with _N suffix for collisions only
                        hit = false;
                    }
                    result.set(i, dn, hit ? 1 : 0);
                }
                // also try matching against category order
                int catIdx = cats.indexOf(v);
                if (catIdx >= 0 && catIdx < dummyNames.size()) {
                    // clear and set correct
                    for (String dn : dummyNames) result.set(i, dn, 0);
                    result.set(i, dummyNames.get(catIdx), 1);
                }
            }
            if (dropOriginal && result.hasColumn(col)) result.removeColumn(col);
        }
        return result;
    }

    public java.util.Map<String, List<String>> getCategories() {
        return categories;
    }
}
