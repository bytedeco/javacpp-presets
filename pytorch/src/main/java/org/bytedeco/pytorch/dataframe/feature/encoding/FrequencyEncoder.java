package org.bytedeco.pytorch.dataframe.feature.encoding;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Encode categories by their relative frequency in the training set.
 * Unknown categories at transform time map to 0.
 */
public class FrequencyEncoder extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    private final Map<String, Map<Object, Double>> freqMaps = new LinkedHashMap<>();
    private boolean replace = true;
    private boolean normalize = true;

    public FrequencyEncoder(String... columns) {
        super(columns);
    }

    public FrequencyEncoder setReplace(boolean replace) {
        this.replace = replace;
        return this;
    }

    public FrequencyEncoder setNormalize(boolean normalize) {
        this.normalize = normalize;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        freqMaps.clear();
        int n = X.rowCount();
        for (String col : columns) {
            Map<Object, Integer> counts = new LinkedHashMap<>();
            Column c = X.column(col);
            for (int i = 0; i < c.size(); i++) {
                Object key = normalizeKey(DataValues.unwrap(c.get(i)));
                counts.merge(key, 1, Integer::sum);
            }
            Map<Object, Double> freq = new HashMap<>();
            for (Map.Entry<Object, Integer> e : counts.entrySet()) {
                double v = normalize ? (n == 0 ? 0.0 : e.getValue() / (double) n) : e.getValue();
                freq.put(e.getKey(), v);
            }
            freqMaps.put(col, freq);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (String col : columns) {
            Map<Object, Double> freq = freqMaps.get(col);
            if (freq == null) continue;
            String outName = replace ? col : col + "_freq";
            if (!replace) {
                if (result.hasColumn(outName)) result.removeColumn(outName);
                result.addColumn(outName, Column.DType.FLOAT64);
                Column oc = result.column(outName);
                while (oc.size() < result.rowCount()) oc.add(null);
            }
            Column src = X.column(col);
            Column dst = result.column(outName);
            for (int i = 0; i < result.rowCount(); i++) {
                Object key = normalizeKey(DataValues.unwrap(src.get(i)));
                dst.set(i, freq.getOrDefault(key, 0.0));
            }
        }
        return result;
    }

    private static Object normalizeKey(Object v) {
        return v == null ? "null" : v;
    }

    public Map<Object, Double> getFrequencies(String column) {
        return freqMaps.get(column);
    }
}
