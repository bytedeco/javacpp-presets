package org.bytedeco.pytorch.data.dataframe;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Result of a groupby operation — supports pandas-style groupby aggregation.
 * Created by {@link DataFrame#groupby(String...)}.
 */
public final class GroupedDataFrame {
    private final DataFrame source;
    private final String[] groupByColumns;
    private final Map<String, List<Integer>> groups; // composite key -> row indices

    GroupedDataFrame(DataFrame source, String[] groupByColumns, Map<String, List<Integer>> groups) {
        this.source = source;
        this.groupByColumns = groupByColumns;
        this.groups = groups;
    }

    public DataFrame getSource() { return source; }
    public String[] getGroupByColumns() { return groupByColumns; }
    public Map<String, List<Integer>> getGroups() { return groups; }

    /**
     * Aggregate by column → aggregation function.
     * Result DataFrame has group-by columns + one column per aggregation result.
     */
    public DataFrame agg(Map<String, AggFunction> aggregations) throws Exception {
        DataFrame result = DataFrame.create();
        for (String col : groupByColumns) {
            result.addColumn(col, source.column(col).dtype());
        }
        for (Map.Entry<String, AggFunction> agg : aggregations.entrySet()) {
            AggFunction fn = agg.getValue();
            Column.DType dt = (fn == AggFunction.COUNT || fn == AggFunction.NUNIQUE)
                ? Column.DType.INT64 : Column.DType.FLOAT64;
            if (!result.hasColumn(agg.getKey())) result.addColumn(agg.getKey(), dt);
        }

        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            List<Integer> indices = e.getValue();
            String[] keyParts = e.getKey().split("\\|", -1);
            int ri = result.addRow();
            for (int i = 0; i < groupByColumns.length; i++) {
                Object kv = i < keyParts.length
                    ? parseValue(keyParts[i], source.column(groupByColumns[i]).dtype())
                    : null;
                result.set(ri, groupByColumns[i], kv);
            }
            for (Map.Entry<String, AggFunction> agg : aggregations.entrySet()) {
                String colName = agg.getKey();
                AggFunction fn = agg.getValue();
                Column srcCol = source.column(colName);
                List<Double> vals = new ArrayList<>();
                for (int idx : indices) {
                    vals.add(toDouble(srcCol.get(idx)));
                }
                Object resultVal = aggregate(vals, fn);
                result.set(ri, colName, resultVal);
            }
        }
        return result;
    }

    /**
     * Polars-style expression aggregation.
     * <pre>
     *   df.groupBy("city").agg(
     *     col("amount").sum().alias("total"),
     *     col("amount").mean().alias("avg")
     *   );
     * </pre>
     */
    public DataFrame agg(Expression... aggs) throws Exception {
        DataFrame result = DataFrame.create();
        for (String col : groupByColumns) {
            result.addColumn(col, source.column(col).dtype());
        }
        // pre-create output columns from first evaluation names
        String[] outNames = new String[aggs.length];
        for (int i = 0; i < aggs.length; i++) {
            outNames[i] = aggs[i].suggestedName();
            // dtype unknown yet — use FLOAT64 placeholder, refine on first group
            result.addColumn(outNames[i], Column.DType.FLOAT64);
        }

        boolean dtypesFixed = false;
        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            int[] idx = e.getValue().stream().mapToInt(Integer::intValue).toArray();
            DataFrame gdf = source.loc(idx);
            String[] keyParts = e.getKey().split("\\|", -1);

            // evaluate aggs on group frame
            Object[] aggVals = new Object[aggs.length];
            Column.DType[] aggTypes = new Column.DType[aggs.length];
            for (int i = 0; i < aggs.length; i++) {
                Column c = aggs[i].evaluate(gdf);
                // aggregation expressions broadcast scalar to all rows — take first
                Object v = c.size() > 0 ? c.get(0) : null;
                // if non-agg multi-row, still take first (documented)
                aggVals[i] = v;
                aggTypes[i] = c.dtype();
            }

            if (!dtypesFixed) {
                // rebuild result columns with correct dtypes for agg outputs
                DataFrame rebuilt = DataFrame.create();
                for (String col : groupByColumns) {
                    rebuilt.addColumn(col, source.column(col).dtype());
                }
                for (int i = 0; i < aggs.length; i++) {
                    rebuilt.addColumn(outNames[i], aggTypes[i]);
                }
                result = rebuilt;
                dtypesFixed = true;
            }

            int ri = result.addEmptyRow();
            for (int i = 0; i < groupByColumns.length; i++) {
                Object kv = i < keyParts.length
                    ? parseValue(keyParts[i], source.column(groupByColumns[i]).dtype())
                    : null;
                result.set(ri, groupByColumns[i], kv);
            }
            for (int i = 0; i < aggs.length; i++) {
                result.set(ri, outNames[i], aggVals[i]);
            }
        }
        return result;
    }

    /** groupby.size() — row count per group. */
    public DataFrame size() throws Exception {
        DataFrame result = DataFrame.create();
        for (String col : groupByColumns) {
            result.addColumn(col, source.column(col).dtype());
        }
        result.addColumn("size", Column.DType.INT64);

        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            String[] keyParts = e.getKey().split("\\|", -1);
            int rowIdx = result.addEmptyRow();
            for (int i = 0; i < groupByColumns.length; i++) {
                result.set(rowIdx, groupByColumns[i],
                    parseValue(i < keyParts.length ? keyParts[i] : "", source.column(groupByColumns[i]).dtype()));
            }
            result.set(rowIdx, "size", (long) e.getValue().size());
        }
        return result;
    }

    /** groupby.count() — same as size() but explicit. */
    public DataFrame count() throws Exception { return size(); }

    /** groupby.apply(func) — apply function to each group DataFrame; concatenate results. */
    public DataFrame apply(java.util.function.Function<DataFrame, DataFrame> func) throws Exception {
        List<DataFrame> parts = new ArrayList<>();
        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            int[] idx = e.getValue().stream().mapToInt(Integer::intValue).toArray();
            DataFrame gdf = source.loc(idx);
            DataFrame applied = func.apply(gdf);
            if (applied != null && applied.rowCount() > 0) parts.add(applied);
        }
        return parts.isEmpty() ? DataFrame.create() : DataFrame.concat(parts, 0);
    }

    /** groupby.filter(predicate) — keep groups where predicate(group) is true. */
    public DataFrame filter(java.util.function.Predicate<DataFrame> predicate) throws Exception {
        List<Integer> keepRows = new ArrayList<>();
        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            int[] idx = e.getValue().stream().mapToInt(Integer::intValue).toArray();
            DataFrame gdf = source.loc(idx);
            if (predicate.test(gdf)) keepRows.addAll(e.getValue());
        }
        Collections.sort(keepRows);
        int[] rows = keepRows.stream().mapToInt(Integer::intValue).toArray();
        return source.loc(rows);
    }

    // ---- helpers ----

    private static Double toDouble(Object v) {
        if (v == null) return Double.NaN;
        if (v instanceof Number) return ((Number) v).doubleValue();
        return Double.NaN;
    }

    private static Object parseValue(String s, Column.DType dtype) {
        if (s == null || s.isEmpty() || "null".equals(s)) return null;
        try {
            return switch (dtype) {
                case INT32 -> Integer.parseInt(s);
                case INT64 -> Long.parseLong(s);
                case FLOAT32 -> Float.parseFloat(s);
                case FLOAT64 -> Double.parseDouble(s);
                case BOOLEAN -> Boolean.parseBoolean(s);
                default -> s;
            };
        } catch (Exception e) { return s; }
    }

    private static Object aggregate(List<Double> vals, AggFunction fn) {
        List<Double> nonNaN = vals.stream()
            .filter(d -> !Double.isNaN(d))
            .collect(Collectors.toList());
        if (nonNaN.isEmpty()) return null;

        switch (fn) {
            case SUM:   return nonNaN.stream().mapToDouble(Double::doubleValue).sum();
            case MEAN:  return nonNaN.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
            case MEDIAN: {
                List<Double> sorted = new ArrayList<>(nonNaN);
                Collections.sort(sorted);
                int n = sorted.size();
                return (n % 2 == 0)
                    ? (sorted.get(n/2-1) + sorted.get(n/2)) / 2.0
                    : sorted.get(n/2);
            }
            case MAX:   return nonNaN.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
            case MIN:   return nonNaN.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
            case COUNT: return (long) vals.size();
            case STD: {
                double mean = nonNaN.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double variance = nonNaN.stream()
                    .mapToDouble(d -> (d - mean) * (d - mean))
                    .sum() / (nonNaN.size() - 1); // sample std (ddof=1)
                return Math.sqrt(variance);
            }
            case VAR: {
                double mean = nonNaN.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                return nonNaN.stream()
                    .mapToDouble(d -> (d - mean) * (d - mean))
                    .sum() / (nonNaN.size() - 1);
            }
            case FIRST:  return vals.get(0);
            case LAST:   return vals.get(vals.size() - 1);
            case MODE: {
                Map<Double, Long> freq = new HashMap<>();
                for (Double d : vals) if (!Double.isNaN(d)) freq.merge(d, 1L, Long::sum);
                return freq.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey).orElse(null);
            }
            case SKEW: {
                double mean = nonNaN.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double std = Math.sqrt(nonNaN.stream()
                    .mapToDouble(d -> (d - mean) * (d - mean)).sum() / (nonNaN.size() - 1));
                if (std == 0) return 0.0;
                double n = nonNaN.size();
                return (nonNaN.stream().mapToDouble(d ->
                    Math.pow((d - mean) / std, 3)).sum()) / n;
            }
            case KURT: {
                double mean = nonNaN.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double std = Math.sqrt(nonNaN.stream()
                    .mapToDouble(d -> (d - mean) * (d - mean)).sum() / (nonNaN.size() - 1));
                if (std == 0) return 0.0;
                double n = nonNaN.size();
                return (nonNaN.stream().mapToDouble(d ->
                    Math.pow((d - mean) / std, 4)).sum()) / n - 3; // excess kurtosis
            }
            default: return null;
        }
    }
}
