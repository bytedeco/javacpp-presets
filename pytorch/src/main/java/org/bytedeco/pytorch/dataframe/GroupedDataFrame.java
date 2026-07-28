package org.bytedeco.pytorch.dataframe;
import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Result of a groupby operation — supports pandas-style groupby aggregation
 * and advanced group controls (ngroup / cumcount / nth / transform / rolling / etc.).
 * Created by {@link DataFrame#groupby(String...)}.
 */
public final class GroupedDataFrame {
    private final DataFrame source;
    private final String[] groupByColumns;
    private final Map<String, List<Integer>> groups; // composite key -> row indices (insertion order)

    GroupedDataFrame(DataFrame source, String[] groupByColumns, Map<String, List<Integer>> groups) {
        this.source = source;
        this.groupByColumns = groupByColumns;
        this.groups = groups;
    }

    public DataFrame getSource() { return source; }
    public String[] getGroupByColumns() { return groupByColumns; }

    /** Pandas {@code GroupBy.groups} — group key → row index list. */
    public Map<String, List<Integer>> getGroups() { return groups; }

    /** Pandas {@code GroupBy.groups} alias. */
    public Map<String, List<Integer>> groups() { return groups; }

    /**
     * Pandas {@code GroupBy.indices} — group key → int[] of row positions.
     */
    public Map<String, int[]> indices() {
        Map<String, int[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            out.put(e.getKey(), e.getValue().stream().mapToInt(Integer::intValue).toArray());
        }
        return out;
    }

    /** Number of groups. */
    public int ngroups() { return groups.size(); }

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
    public DataFrame apply(Function<DataFrame, DataFrame> func) throws Exception {
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
    public DataFrame filter(Predicate<DataFrame> predicate) throws Exception {
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

    // ================================================================
    // Advanced group controls (Pandas 21–40)
    // ================================================================

    /**
     * Pandas {@code GroupBy.ngroup()} — integer group id 0..G-1 for each source row
     * (order of first appearance of the group key).
     */
    public Column ngroup() {
        Map<String, Integer> idMap = new LinkedHashMap<>();
        int gid = 0;
        for (String k : groups.keySet()) idMap.put(k, gid++);
        Column out = new Column("ngroup", Column.DType.INT64);
        long[] vals = new long[source.rowCount()];
        Arrays.fill(vals, -1L);
        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            int id = idMap.get(e.getKey());
            for (int idx : e.getValue()) vals[idx] = id;
        }
        for (long v : vals) out.add(v);
        return out;
    }

    /**
     * Pandas {@code GroupBy.cumcount()} — 0-based row number within each group.
     */
    public Column cumcount() {
        Column out = new Column("cumcount", Column.DType.INT64);
        long[] vals = new long[source.rowCount()];
        for (List<Integer> idxs : groups.values()) {
            for (int i = 0; i < idxs.size(); i++) vals[idxs.get(i)] = i;
        }
        for (long v : vals) out.add(v);
        return out;
    }

    /**
     * Pandas {@code GroupBy.nth(n)} — take the n-th row of each group (0-based).
     * Negative n counts from the end (−1 = last).
     */
    public DataFrame nth(int n) {
        List<Integer> keep = new ArrayList<>();
        for (List<Integer> idxs : groups.values()) {
            if (idxs.isEmpty()) continue;
            int pos = n >= 0 ? n : idxs.size() + n;
            if (pos >= 0 && pos < idxs.size()) keep.add(idxs.get(pos));
        }
        Collections.sort(keep);
        return source.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Polars {@code group_by().head(n)} — first n rows of each group.
     */
    public DataFrame head(int n) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        List<Integer> keep = new ArrayList<>();
        for (List<Integer> idxs : groups.values()) {
            int take = Math.min(n, idxs.size());
            for (int i = 0; i < take; i++) keep.add(idxs.get(i));
        }
        Collections.sort(keep);
        return source.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Polars {@code group_by().tail(n)} — last n rows of each group.
     */
    public DataFrame tail(int n) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        List<Integer> keep = new ArrayList<>();
        for (List<Integer> idxs : groups.values()) {
            int start = Math.max(0, idxs.size() - n);
            for (int i = start; i < idxs.size(); i++) keep.add(idxs.get(i));
        }
        Collections.sort(keep);
        return source.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Polars {@code group_by().sample(n)} — random sample of up to n rows per group.
     */
    public DataFrame sample(int n) {
        return sample(n, null);
    }

    public DataFrame sample(int n, Long seed) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        Random rng = seed == null ? new Random() : new Random(seed);
        List<Integer> keep = new ArrayList<>();
        for (List<Integer> idxs : groups.values()) {
            if (idxs.isEmpty()) continue;
            List<Integer> copy = new ArrayList<>(idxs);
            Collections.shuffle(copy, rng);
            int take = Math.min(n, copy.size());
            keep.addAll(copy.subList(0, take));
        }
        Collections.sort(keep);
        return source.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Pandas {@code GroupBy.get_group(name)} — extract one group by composite key
     * (parts joined with {@code "|"} in group-column order).
     */
    public DataFrame getGroup(Object... keyParts) {
        String key = joinKey(keyParts);
        List<Integer> idxs = groups.get(key);
        if (idxs == null) {
            // try single-part string match
            if (keyParts.length == 1 && groups.containsKey(String.valueOf(keyParts[0]))) {
                idxs = groups.get(String.valueOf(keyParts[0]));
            }
        }
        if (idxs == null) {
            throw new IllegalArgumentException("Group not found: " + key + " (known=" + groups.keySet() + ")");
        }
        return source.loc(idxs.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Pandas {@code GroupBy.transform(func)} — apply per-group and broadcast
     * results back to original row order (same length as source).
     * <p>{@code func} receives a group DataFrame and must return a DataFrame
     * with the same row count as the group (columns may differ).
     */
    public DataFrame transform(Function<DataFrame, DataFrame> func) throws Exception {
        // materialize each group result, then stitch by original index
        Map<Integer, Map<String, Object>> byRow = new HashMap<>();
        List<String> outCols = null;
        Map<String, Column.DType> outTypes = new LinkedHashMap<>();

        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            List<Integer> idxs = e.getValue();
            int[] arr = idxs.stream().mapToInt(Integer::intValue).toArray();
            DataFrame gdf = source.loc(arr);
            DataFrame applied = func.apply(gdf);
            if (applied == null) continue;
            if (applied.rowCount() != idxs.size()) {
                throw new IllegalArgumentException(
                    "transform func must return same row count as group: expected "
                        + idxs.size() + " got " + applied.rowCount());
            }
            if (outCols == null) {
                outCols = applied.getColumnNames();
                for (String c : outCols) outTypes.put(c, applied.column(c).dtype());
            }
            for (int i = 0; i < idxs.size(); i++) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (String c : outCols) row.put(c, applied.get(i, c));
                byRow.put(idxs.get(i), row);
            }
        }
        if (outCols == null) return DataFrame.create();

        DataFrame result = DataFrame.create();
        for (String c : outCols) result.addColumn(c, outTypes.get(c));
        for (int r = 0; r < source.rowCount(); r++) {
            int ri = result.addEmptyRow();
            Map<String, Object> row = byRow.get(r);
            if (row == null) continue;
            for (String c : outCols) result.set(ri, c, row.get(c));
        }
        return result;
    }

    /**
     * Pandas {@code GroupBy.pipe(func)} — pass this GroupedDataFrame into an external function.
     */
    public <R> R pipe(Function<GroupedDataFrame, R> func) {
        return func.apply(this);
    }

    /**
     * Pandas {@code GroupBy.shift(periods)} — shift values within each group;
     * returns a full-length frame with shifted columns (group keys preserved).
     */
    public DataFrame shift(int periods) {
        DataFrame result = source.copy();
        for (Column col : source.columns()) {
            if (isGroupKey(col.name())) continue;
            Column out = result.column(col.name());
            // null everything first for non-key cols, then fill shifted
            Object[] buf = new Object[source.rowCount()];
            for (List<Integer> idxs : groups.values()) {
                for (int i = 0; i < idxs.size(); i++) {
                    int src = i - periods;
                    buf[idxs.get(i)] = (src >= 0 && src < idxs.size()) ? col.get(idxs.get(src)) : null;
                }
            }
            for (int r = 0; r < source.rowCount(); r++) out.set(r, buf[r]);
        }
        return result;
    }

    /**
     * Pandas {@code GroupBy.diff()} — within-group first difference for numeric columns.
     */
    public DataFrame diff() {
        return diff(1);
    }

    public DataFrame diff(int periods) {
        DataFrame result = source.copy();
        for (Column col : source.columns()) {
            if (isGroupKey(col.name())) continue;
            if (!isNumeric(col.dtype())) continue;
            Column out = result.column(col.name());
            Object[] buf = new Object[source.rowCount()];
            for (List<Integer> idxs : groups.values()) {
                for (int i = 0; i < idxs.size(); i++) {
                    int src = i - periods;
                    if (src < 0 || src >= idxs.size()) {
                        buf[idxs.get(i)] = null;
                    } else {
                        double a = DataValues.asDouble(col.get(idxs.get(i)));
                        double b = DataValues.asDouble(col.get(idxs.get(src)));
                        buf[idxs.get(i)] = (Double.isNaN(a) || Double.isNaN(b)) ? null : a - b;
                    }
                }
            }
            for (int r = 0; r < source.rowCount(); r++) out.set(r, buf[r]);
        }
        return result;
    }

    /**
     * Pandas {@code GroupBy.rank(method, ascending)} — rank within each group for all numeric columns.
     * Methods: average, min, max, dense, ordinal.
     */
    public DataFrame rank(String method, boolean ascending) {
        String m = method == null ? "average" : method.toLowerCase(Locale.ROOT);
        DataFrame result = source.copy();
        for (Column col : source.columns()) {
            if (isGroupKey(col.name())) continue;
            if (!isNumeric(col.dtype())) continue;
            String outName = col.name() + "_rank";
            if (result.hasColumn(outName)) result.removeColumn(outName);
            result.addColumn(outName, Column.DType.FLOAT64);
            Column out = result.column(outName);
            while (out.size() < result.rowCount()) out.add(null);

            for (List<Integer> idxs : groups.values()) {
                // stable sort of group indices by value
                List<Integer> order = new ArrayList<>(idxs);
                order.sort((a, b) -> {
                    int cmp = Expression.compareVals(col.get(a), col.get(b));
                    return ascending ? cmp : -cmp;
                });
                int i = 0;
                while (i < order.size()) {
                    int j = i;
                    Object v = col.get(order.get(i));
                    while (j < order.size() && Objects.equals(v, col.get(order.get(j)))) j++;
                    // ranks in [i+1, j] (1-based)
                    double rankVal;
                    switch (m) {
                        case "min" -> rankVal = i + 1.0;
                        case "max" -> rankVal = j;
                        case "dense" -> {
                            // dense: count distinct previous
                            int dense = 1;
                            Object prev = null;
                            for (int k = 0; k <= i; k++) {
                                Object cur = col.get(order.get(k));
                                if (k == 0 || !Objects.equals(prev, cur)) {
                                    if (k > 0) dense++;
                                    prev = cur;
                                }
                            }
                            rankVal = dense;
                        }
                        case "ordinal" -> {
                            for (int k = i; k < j; k++) out.set(order.get(k), (double) (k + 1));
                            i = j;
                            continue;
                        }
                        default -> rankVal = (i + 1 + j) / 2.0; // average
                    }
                    for (int k = i; k < j; k++) out.set(order.get(k), rankVal);
                    i = j;
                }
            }
        }
        return result;
    }

    public DataFrame rank() {
        return rank("average", true);
    }

    /**
     * Pandas {@code GroupBy.value_counts()} — frequency of each value of {@code column}
     * within each group. Result has group keys + value + count.
     */
    public DataFrame valueCounts(String column) {
        return valueCounts(column, false);
    }

    public DataFrame valueCounts(String column, boolean normalize) {
        Column src = source.column(column);
        DataFrame result = DataFrame.create();
        for (String g : groupByColumns) result.addColumn(g, source.column(g).dtype());
        // value column may collide with a group key — rename to value_<col>
        String valueCol = column;
        if (result.hasColumn(valueCol)) valueCol = "value_" + column;
        result.addColumn(valueCol, src.dtype());
        String countCol = normalize ? "proportion" : "count";
        result.addColumn(countCol, normalize ? Column.DType.FLOAT64 : Column.DType.INT64);

        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            String[] keyParts = e.getKey().split("\\|", -1);
            Map<Object, Integer> freq = new LinkedHashMap<>();
            for (int idx : e.getValue()) {
                Object v = src.get(idx);
                freq.merge(v, 1, Integer::sum);
            }
            int total = e.getValue().size();
            for (Map.Entry<Object, Integer> fe : freq.entrySet()) {
                int ri = result.addEmptyRow();
                for (int i = 0; i < groupByColumns.length; i++) {
                    result.set(ri, groupByColumns[i],
                        parseValue(i < keyParts.length ? keyParts[i] : "",
                            source.column(groupByColumns[i]).dtype()));
                }
                result.set(ri, valueCol, fe.getKey());
                if (normalize) result.set(ri, countCol, fe.getValue() / (double) total);
                else result.set(ri, countCol, (long) fe.getValue());
            }
        }
        return result;
    }

    /**
     * Polars-style {@code group_by().agg(col.implode())} helper — pack a column's
     * values per group into a LIST column.
     */
    public DataFrame implode(String column) throws Exception {
        Column src = source.column(column);
        DataFrame result = DataFrame.create();
        for (String g : groupByColumns) result.addColumn(g, source.column(g).dtype());
        result.addColumn(column, Column.DType.LIST);
        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            String[] keyParts = e.getKey().split("\\|", -1);
            List<Object> list = new ArrayList<>(e.getValue().size());
            for (int idx : e.getValue()) list.add(src.get(idx));
            int ri = result.addEmptyRow();
            for (int i = 0; i < groupByColumns.length; i++) {
                result.set(ri, groupByColumns[i],
                    parseValue(i < keyParts.length ? keyParts[i] : "",
                        source.column(groupByColumns[i]).dtype()));
            }
            result.set(ri, column, list);
        }
        return result;
    }

    // ================================================================
    // Group-wise rolling / expanding (Pandas GroupBy.rolling / .expanding)
    // ================================================================

    /**
     * Pandas {@code GroupBy.rolling(window)} — rolling window applied independently
     * within each group; result is aligned to original row order.
     */
    public GroupRolling rolling(int window) {
        return rolling(window, window);
    }

    public GroupRolling rolling(int window, int minPeriods) {
        return new GroupRolling(this, window, minPeriods);
    }

    /** Pandas {@code GroupBy.expanding()}. */
    public GroupExpanding expanding() {
        return expanding(1);
    }

    public GroupExpanding expanding(int minPeriods) {
        return new GroupExpanding(this, minPeriods);
    }

    /**
     * Group-wise rolling aggregator. Methods return a full-length DataFrame with
     * the aggregated column (null/NaN where window not full).
     */
    public static final class GroupRolling {
        private final GroupedDataFrame parent;
        private final int window;
        private final int minPeriods;

        GroupRolling(GroupedDataFrame parent, int window, int minPeriods) {
            if (window <= 0) throw new IllegalArgumentException("window must be > 0");
            this.parent = parent;
            this.window = window;
            this.minPeriods = Math.max(1, minPeriods);
        }

        public DataFrame mean(String col) { return reduce(col, "rolling_mean", Op.MEAN); }
        public DataFrame sum(String col)  { return reduce(col, "rolling_sum", Op.SUM); }
        public DataFrame min(String col)  { return reduce(col, "rolling_min", Op.MIN); }
        public DataFrame max(String col)  { return reduce(col, "rolling_max", Op.MAX); }
        public DataFrame std(String col)  { return reduce(col, "rolling_std", Op.STD); }
        public DataFrame count(String col){ return reduce(col, "rolling_count", Op.COUNT); }

        private DataFrame reduce(String col, String outName, Op op) {
            DataFrame src = parent.source;
            Column c = src.column(col);
            Object[] out = new Object[src.rowCount()];
            for (List<Integer> idxs : parent.groups.values()) {
                for (int i = 0; i < idxs.size(); i++) {
                    int start = Math.max(0, i - window + 1);
                    List<Double> win = new ArrayList<>();
                    for (int j = start; j <= i; j++) {
                        double v = DataValues.asDouble(c.get(idxs.get(j)));
                        if (!Double.isNaN(v)) win.add(v);
                    }
                    out[idxs.get(i)] = win.size() < minPeriods ? null : applyOp(win, op);
                }
            }
            return withOutColumn(src, outName, out);
        }
    }

    public static final class GroupExpanding {
        private final GroupedDataFrame parent;
        private final int minPeriods;

        GroupExpanding(GroupedDataFrame parent, int minPeriods) {
            this.parent = parent;
            this.minPeriods = Math.max(1, minPeriods);
        }

        public DataFrame mean(String col) { return reduce(col, "expanding_mean", Op.MEAN); }
        public DataFrame sum(String col)  { return reduce(col, "expanding_sum", Op.SUM); }
        public DataFrame min(String col)  { return reduce(col, "expanding_min", Op.MIN); }
        public DataFrame max(String col)  { return reduce(col, "expanding_max", Op.MAX); }
        public DataFrame std(String col)  { return reduce(col, "expanding_std", Op.STD); }
        public DataFrame count(String col){ return reduce(col, "expanding_count", Op.COUNT); }

        private DataFrame reduce(String col, String outName, Op op) {
            DataFrame src = parent.source;
            Column c = src.column(col);
            Object[] out = new Object[src.rowCount()];
            for (List<Integer> idxs : parent.groups.values()) {
                List<Double> win = new ArrayList<>();
                for (int i = 0; i < idxs.size(); i++) {
                    double v = DataValues.asDouble(c.get(idxs.get(i)));
                    if (!Double.isNaN(v)) win.add(v);
                    out[idxs.get(i)] = win.size() < minPeriods ? null : applyOp(win, op);
                }
            }
            return withOutColumn(src, outName, out);
        }
    }

    private enum Op { MEAN, SUM, MIN, MAX, STD, COUNT }

    private static Object applyOp(List<Double> win, Op op) {
        if (win.isEmpty()) return null;
        return switch (op) {
            case SUM -> win.stream().mapToDouble(d -> d).sum();
            case MEAN -> win.stream().mapToDouble(d -> d).average().orElse(Double.NaN);
            case MIN -> win.stream().mapToDouble(d -> d).min().orElse(Double.NaN);
            case MAX -> win.stream().mapToDouble(d -> d).max().orElse(Double.NaN);
            case COUNT -> (double) win.size();
            case STD -> {
                int n = win.size();
                if (n < 2) yield null;
                double mean = win.stream().mapToDouble(d -> d).average().orElse(0);
                double ss = 0;
                for (double v : win) ss += (v - mean) * (v - mean);
                yield Math.sqrt(ss / (n - 1));
            }
        };
    }

    private static DataFrame withOutColumn(DataFrame src, String outName, Object[] vals) {
        DataFrame result = src.copy();
        if (result.hasColumn(outName)) result.removeColumn(outName);
        result.addColumn(outName, Column.DType.FLOAT64);
        Column out = result.column(outName);
        while (out.size() < result.rowCount()) out.add(null);
        for (int i = 0; i < vals.length; i++) out.set(i, vals[i]);
        return result;
    }

    // ---- helpers ----

    private boolean isGroupKey(String name) {
        for (String g : groupByColumns) if (g.equals(name)) return true;
        return false;
    }

    private static boolean isNumeric(Column.DType dt) {
        return dt == Column.DType.INT32 || dt == Column.DType.INT64
            || dt == Column.DType.FLOAT32 || dt == Column.DType.FLOAT64;
    }

    private static String joinKey(Object... parts) {
        if (parts == null || parts.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append('|');
            sb.append(parts[i] == null ? "null" : parts[i].toString());
        }
        return sb.toString();
    }

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
