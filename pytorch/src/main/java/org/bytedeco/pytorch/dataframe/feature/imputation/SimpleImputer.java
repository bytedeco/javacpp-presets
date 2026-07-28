package org.bytedeco.pytorch.dataframe.feature.imputation;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * Simple missing-value imputer. Strategies: mean, median, most_frequent, constant.
 */
public class SimpleImputer extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    private final String strategy;
    private final Object fillValue;
    private final Map<String, Object> statistics = new LinkedHashMap<>();

    public SimpleImputer(String strategy, String... columns) {
        this(strategy, null, columns);
    }

    public SimpleImputer(String strategy, Object fillValue, String... columns) {
        super(columns);
        this.strategy = strategy == null ? "mean" : strategy.toLowerCase(Locale.ROOT);
        this.fillValue = fillValue;
    }

    /** Factory for constant fill — avoids (String, String...) overload ambiguity with fill string. */
    public static SimpleImputer constant(Object fillValue, String... columns) {
        return new SimpleImputer("constant", fillValue, columns);
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        statistics.clear();
        List<String> cols = columns.isEmpty() ? numericNames(X) : columns;
        for (String col : cols) {
            Column c = X.column(col);
            switch (strategy) {
                case "constant" -> statistics.put(col, fillValue);
                case "most_frequent", "mode" -> statistics.put(col, mode(c));
                case "median" -> statistics.put(col, median(c));
                default -> statistics.put(col, mean(c)); // mean
            }
        }
        if (columns.isEmpty()) this.columns = cols;
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (Map.Entry<String, Object> e : statistics.entrySet()) {
            String col = e.getKey();
            if (!result.hasColumn(col)) continue;
            Object fill = e.getValue();
            Column c = result.column(col);
            for (int i = 0; i < result.rowCount(); i++) {
                Object v = c.get(i);
                if (v == null || (v instanceof Number && Double.isNaN(((Number) v).doubleValue()))) {
                    c.set(i, fill);
                }
            }
        }
        return result;
    }

    private static List<String> numericNames(DataFrame X) {
        List<String> out = new ArrayList<>();
        for (Column c : X.columns()) {
            Column.DType d = c.dtype();
            if (d == Column.DType.INT32 || d == Column.DType.INT64
                || d == Column.DType.FLOAT32 || d == Column.DType.FLOAT64) {
                out.add(c.name());
            }
        }
        return out;
    }

    private static Double mean(Column c) {
        double sum = 0; int n = 0;
        for (int i = 0; i < c.size(); i++) {
            double v = DataValues.asDouble(c.get(i));
            if (!Double.isNaN(v)) { sum += v; n++; }
        }
        return n == 0 ? 0.0 : sum / n;
    }

    private static Double median(Column c) {
        List<Double> vals = new ArrayList<>();
        for (int i = 0; i < c.size(); i++) {
            double v = DataValues.asDouble(c.get(i));
            if (!Double.isNaN(v)) vals.add(v);
        }
        if (vals.isEmpty()) return 0.0;
        Collections.sort(vals);
        int n = vals.size();
        return (n % 2 == 1) ? vals.get(n / 2) : (vals.get(n / 2 - 1) + vals.get(n / 2)) / 2.0;
    }

    private static Object mode(Column c) {
        Map<Object, Integer> freq = new LinkedHashMap<>();
        for (int i = 0; i < c.size(); i++) {
            Object v = DataValues.unwrap(c.get(i));
            if (v != null) freq.merge(v, 1, Integer::sum);
        }
        Object best = null; int bestC = -1;
        for (Map.Entry<Object, Integer> e : freq.entrySet()) {
            if (e.getValue() > bestC) { bestC = e.getValue(); best = e.getKey(); }
        }
        return best;
    }

    public Map<String, Object> getStatistics() { return statistics; }
}
