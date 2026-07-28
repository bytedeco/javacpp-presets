package org.bytedeco.pytorch.dataframe.feature.encoding;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * One-hot encode categorical columns into INT32 0/1 indicator columns.
 * sklearn-aligned options: {@code handle_unknown}, {@code min_frequency}.
 */
public class OneHotEncoder extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private final boolean dropOriginal;
    private final String prefix;
    /** "error" | "ignore" */
    private String handleUnknown = "ignore";
    /** Absolute count threshold, or fraction in (0,1] when &lt;= 1. Null = keep all. */
    private Double minFrequency = null;
    /** Fitted categories per input column (after min_frequency filter). */
    private final Map<String, List<String>> categories = new LinkedHashMap<>();
    private final Map<String, List<String>> dummyColumnNames = new LinkedHashMap<>();

    public OneHotEncoder(String... columns) {
        this(true, null, columns);
    }

    public OneHotEncoder(boolean dropOriginal, String prefix, String... columns) {
        super(columns);
        this.dropOriginal = dropOriginal;
        this.prefix = prefix;
    }

    public OneHotEncoder setHandleUnknown(String handleUnknown) {
        this.handleUnknown = handleUnknown == null ? "ignore" : handleUnknown.toLowerCase(Locale.ROOT);
        return this;
    }

    /**
     * @param minFrequency absolute count if &gt; 1, else fraction of rows in (0,1]
     */
    public OneHotEncoder setMinFrequency(double minFrequency) {
        this.minFrequency = minFrequency;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        categories.clear();
        dummyColumnNames.clear();
        int nRows = X.rowCount();
        for (String col : columns) {
            Map<String, Integer> freq = new LinkedHashMap<>();
            Column c = X.column(col);
            for (int i = 0; i < c.size(); i++) {
                Object v = DataValues.unwrap(c.get(i));
                String key = v == null ? "null" : v.toString();
                freq.merge(key, 1, Integer::sum);
            }
            int minCount;
            if (minFrequency == null) {
                minCount = 1;
            } else if (minFrequency > 1.0) {
                minCount = (int) Math.ceil(minFrequency);
            } else {
                minCount = Math.max(1, (int) Math.ceil(minFrequency * nRows));
            }
            List<String> kept = new ArrayList<>();
            for (Map.Entry<String, Integer> e : freq.entrySet()) {
                if (e.getValue() >= minCount) kept.add(e.getKey());
            }
            // if everything filtered, keep most frequent
            if (kept.isEmpty() && !freq.isEmpty()) {
                String best = null;
                int bestC = -1;
                for (Map.Entry<String, Integer> e : freq.entrySet()) {
                    if (e.getValue() > bestC) {
                        bestC = e.getValue();
                        best = e.getKey();
                    }
                }
                if (best != null) kept.add(best);
            }
            categories.put(col, kept);
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
                String name = safeName(pfx + "_" + u);
                String base = name;
                int n = 1;
                while (result.hasColumn(name)) name = base + "_" + (n++);
                result.addColumn(name, Column.DType.INT32);
                Column dcol = result.column(name);
                while (dcol.size() < result.rowCount()) dcol.add(0);
                for (int i = 0; i < result.rowCount(); i++) dcol.set(i, 0);
                dummyNames.add(name);
            }
            dummyColumnNames.put(col, dummyNames);

            Column src = X.column(col);
            for (int i = 0; i < result.rowCount(); i++) {
                Object raw = DataValues.unwrap(src.get(i));
                String v = raw == null ? "null" : raw.toString();
                int catIdx = cats.indexOf(v);
                if (catIdx < 0) {
                    if ("error".equals(handleUnknown)) {
                        throw new IllegalArgumentException(
                            "OneHotEncoder: unknown category '" + v + "' in column " + col);
                    }
                    // ignore → all zeros
                    continue;
                }
                result.set(i, dummyNames.get(catIdx), 1);
            }
            if (dropOriginal && result.hasColumn(col)) result.removeColumn(col);
        }
        return result;
    }

    private static String safeName(String s) {
        return s.replaceAll("[^A-Za-z0-9_\\.\\-]+", "_");
    }

    public Map<String, List<String>> getCategories() {
        return categories;
    }

    public Map<String, List<String>> getDummyColumnNames() {
        return dummyColumnNames;
    }

    public String getHandleUnknown() { return handleUnknown; }
    public Double getMinFrequency() { return minFrequency; }
}
