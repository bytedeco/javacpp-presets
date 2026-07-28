package org.bytedeco.pytorch.dataframe;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Lazy DataFrame — expression-based plan that evaluates on {@link #collect()}.
 * Supports rule-based optimization (predicate pushdown, filter merge).
 */
public final class LazyDataFrame {

    private final DataFrame source;
    private final List<LazyOp> plan;

    LazyDataFrame(DataFrame source) {
        this(source, List.of());
    }

    private LazyDataFrame(DataFrame source, List<LazyOp> plan) {
        this.source = source;
        this.plan = List.copyOf(plan);
    }

    private LazyDataFrame append(LazyOp op) {
        List<LazyOp> next = new ArrayList<>(plan.size() + 1);
        next.addAll(plan);
        next.add(op);
        return new LazyDataFrame(source, next);
    }

    // ---- plan ops (package-visible for optimizer introspection) ----

    sealed interface LazyOp permits
        SelectNames, SelectExprs, WithColumn, Filter, Sort, Limit, Head, Tail,
        Drop, Rename, Unique, TopK, DropNulls, Concat, Cache, SetSorted, GroupByAgg,
        OptimizationToggle, GroupByHead, GroupByTail, MapBatches {
        String describe();
        DataFrame apply(DataFrame df) throws Exception;
        default Set<String> producedColumns() { return Set.of(); }
        default Set<String> referencedColumns() { return Set.of(); }
    }

    record SelectNames(String[] cols) implements LazyOp {
        @Override public String describe() { return "SELECT " + Arrays.toString(cols); }
        @Override public DataFrame apply(DataFrame df) { return df.select(cols); }
        @Override public Set<String> producedColumns() { return Set.of(cols); }
        @Override public Set<String> referencedColumns() { return Set.of(cols); }
    }

    record SelectExprs(Expression[] exprs) implements LazyOp {
        @Override public String describe() { return "SELECT_EXPR " + Arrays.toString(exprs); }
        @Override public DataFrame apply(DataFrame df) {
            DataFrame result = DataFrame.create();
            for (Expression e : exprs) {
                Column c = e.evaluate(df);
                if (result.hasColumn(c.name())) result.removeColumn(c.name());
                result.addColumn(c);
            }
            return result;
        }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>();
            for (Expression e : exprs) s.addAll(e.referencedColumns());
            return s;
        }
    }

    record WithColumn(String name, Expression expr) implements LazyOp {
        @Override public String describe() { return "WITH_COLUMN " + name + " = " + expr; }
        @Override public DataFrame apply(DataFrame df) { return df.withColumn(name, expr); }
        @Override public Set<String> producedColumns() { return Set.of(name); }
        @Override public Set<String> referencedColumns() { return expr.referencedColumns(); }
    }

    record Filter(Expression condition) implements LazyOp {
        @Override public String describe() { return "FILTER " + condition; }
        @Override public DataFrame apply(DataFrame df) { return df.filter(condition); }
        @Override public Set<String> referencedColumns() { return condition.referencedColumns(); }
    }

    record Sort(Expression[] by, Boolean nullsLast, Boolean maintainOrder) implements LazyOp {
        @Override public String describe() { return "SORT " + Arrays.toString(by); }
        @Override public DataFrame apply(DataFrame df) { return sortByExpressions(df, by); }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>();
            for (Expression e : by) s.addAll(e.referencedColumns());
            return s;
        }
    }

    record Limit(int n) implements LazyOp {
        @Override public String describe() { return "LIMIT " + n; }
        @Override public DataFrame apply(DataFrame df) { return df.head(n); }
    }

    record Head(int n) implements LazyOp {
        @Override public String describe() { return "HEAD " + n; }
        @Override public DataFrame apply(DataFrame df) { return df.head(n); }
    }

    record Tail(int n) implements LazyOp {
        @Override public String describe() { return "TAIL " + n; }
        @Override public DataFrame apply(DataFrame df) { return df.tail(n); }
    }

    record Drop(String[] cols) implements LazyOp {
        @Override public String describe() { return "DROP " + Arrays.toString(cols); }
        @Override public DataFrame apply(DataFrame df) { return df.drop(cols); }
    }

    record Rename(String oldName, String newName) implements LazyOp {
        @Override public String describe() { return "RENAME " + oldName + " → " + newName; }
        @Override public DataFrame apply(DataFrame df) {
            DataFrame result = df.copy();
            result.renameColumn(oldName, newName);
            return result;
        }
    }

    record Unique() implements LazyOp {
        @Override public String describe() { return "UNIQUE"; }
        @Override public DataFrame apply(DataFrame df) { return df.dropDuplicates(); }
    }

    record TopK(int k, String col, boolean reverse) implements LazyOp {
        @Override public String describe() {
            return "TOP_K " + k + " by " + col + (reverse ? " ASC" : " DESC");
        }
        @Override public DataFrame apply(DataFrame df) {
            DataFrame sorted = df.sortValues(col, reverse);
            return sorted.head(k);
        }
    }

    record DropNulls() implements LazyOp {
        @Override public String describe() { return "DROP_NULLS"; }
        @Override public DataFrame apply(DataFrame df) throws Exception { return df.dropna(); }
    }

    record Concat(LazyDataFrame[] others) implements LazyOp {
        @Override public String describe() { return "CONCAT +" + others.length + " frames"; }
        @Override public DataFrame apply(DataFrame df) throws Exception {
            List<DataFrame> parts = new ArrayList<>();
            parts.add(df);
            for (LazyDataFrame o : others) parts.add(o.collect());
            return DataFrame.concat(parts, 0);
        }
    }

    record Cache() implements LazyOp {
        @Override public String describe() { return "CACHE"; }
        @Override public DataFrame apply(DataFrame df) { return df; }
    }

    record SetSorted(String col, boolean descending, boolean nullsLast) implements LazyOp {
        @Override public String describe() {
            return "SET_SORTED " + col + (descending ? " DESC" : " ASC");
        }
        @Override public DataFrame apply(DataFrame df) { return df; }
    }

    record GroupByAgg(String[] keys, Expression[] aggs) implements LazyOp {
        @Override public String describe() {
            return "GROUP_BY " + Arrays.toString(keys) + " AGG " + Arrays.toString(aggs);
        }
        @Override public DataFrame apply(DataFrame df) throws Exception {
            return df.groupBy(keys).agg(aggs);
        }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(Arrays.asList(keys));
            for (Expression e : aggs) s.addAll(e.referencedColumns());
            return s;
        }
    }

    /** Plan marker: when enabled=false, collect() skips LazyOptimizer. */
    record OptimizationToggle(boolean enabled) implements LazyOp {
        @Override public String describe() {
            return "OPTIMIZATION_TOGGLE " + (enabled ? "ON" : "OFF");
        }
        @Override public DataFrame apply(DataFrame df) { return df; }
    }

    record GroupByHead(String[] keys, int n) implements LazyOp {
        @Override public String describe() {
            return "GROUP_BY " + Arrays.toString(keys) + " HEAD " + n;
        }
        @Override public DataFrame apply(DataFrame df) {
            return df.groupBy(keys).head(n);
        }
        @Override public Set<String> referencedColumns() {
            return new LinkedHashSet<>(Arrays.asList(keys));
        }
    }

    record GroupByTail(String[] keys, int n) implements LazyOp {
        @Override public String describe() {
            return "GROUP_BY " + Arrays.toString(keys) + " TAIL " + n;
        }
        @Override public DataFrame apply(DataFrame df) {
            return df.groupBy(keys).tail(n);
        }
        @Override public Set<String> referencedColumns() {
            return new LinkedHashSet<>(Arrays.asList(keys));
        }
    }

    /**
     * Polars {@code map_batches} — apply a batch UDF to the whole frame (or
     * streaming chunks when collected with streaming=true).
     */
    record MapBatches(java.util.function.Function<DataFrame, DataFrame> fn, String label) implements LazyOp {
        @Override public String describe() {
            return "MAP_BATCHES" + (label == null || label.isEmpty() ? "" : " " + label);
        }
        @Override public DataFrame apply(DataFrame df) {
            DataFrame out = fn.apply(df);
            return out == null ? DataFrame.create() : out;
        }
    }

    // ---- public API ----

    public DataFrame collect() { return collect(true); }

    public DataFrame collect(boolean optimize) {
        try {
            boolean doOpt = optimize;
            // honor last OptimizationToggle in plan
            for (LazyOp op : plan) {
                if (op instanceof OptimizationToggle t) doOpt = t.enabled();
            }
            List<LazyOp> ops = doOpt ? LazyOptimizer.optimize(plan) : plan;
            // strip toggle markers before apply
            List<LazyOp> runnable = new ArrayList<>(ops.size());
            for (LazyOp op : ops) {
                if (!(op instanceof OptimizationToggle)) runnable.add(op);
            }
            DataFrame df = source.copy();
            for (LazyOp op : runnable) {
                df = op.apply(df);
            }
            return df;
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Lazy collect failed: " + e.getMessage(), e);
        }
    }

    /** Polars {@code collect_no_optimization()} — force logical plan execution. */
    public DataFrame collectNoOptimization() { return collect(false); }

    /**
     * Polars {@code collect(streaming=True)} analogue.
     * When {@code streaming} is true and the plan is a pure projection/filter/limit
     * chain, processes the source in row batches of {@code batchRows} to bound peak heap.
     * Complex plans (join/groupby) fall back to regular {@link #collect()}.
     */
    public DataFrame collect(boolean optimize, boolean streaming, int batchRows) {
        if (!streaming || !isStreamingFriendly()) return collect(optimize);
        int batch = Math.max(1, batchRows);
        try {
            boolean doOpt = optimize;
            for (LazyOp op : plan) {
                if (op instanceof OptimizationToggle t) doOpt = t.enabled();
            }
            List<LazyOp> ops = doOpt ? LazyOptimizer.optimize(plan) : plan;
            List<LazyOp> runnable = new ArrayList<>(ops.size());
            for (LazyOp op : ops) {
                if (!(op instanceof OptimizationToggle)) runnable.add(op);
            }
            List<DataFrame> parts = new ArrayList<>();
            int n = source.rowCount();
            for (int start = 0; start < n; start += batch) {
                int end = Math.min(n, start + batch);
                int[] idx = new int[end - start];
                for (int i = 0; i < idx.length; i++) idx[i] = start + i;
                DataFrame chunk = source.loc(idx);
                for (LazyOp op : runnable) chunk = op.apply(chunk);
                if (chunk.rowCount() > 0) parts.add(chunk);
            }
            if (parts.isEmpty()) return DataFrame.create();
            if (parts.size() == 1) return parts.get(0);
            return DataFrame.vstack(parts);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Lazy streaming collect failed: " + e.getMessage(), e);
        }
    }

    public DataFrame collectStreaming(int batchRows) {
        return collect(true, true, batchRows);
    }

    private boolean isStreamingFriendly() {
        for (LazyOp op : plan) {
            if (op instanceof GroupByAgg || op instanceof Concat || op instanceof Sort
                || op instanceof TopK || op instanceof GroupByHead || op instanceof GroupByTail) {
                return false;
            }
        }
        return true;
    }

    public LazyDataFrame select(String... cols) { return append(new SelectNames(cols)); }
    public LazyDataFrame select(Expression... exprs) { return append(new SelectExprs(exprs)); }
    public LazyDataFrame withColumn(String name, Expression expr) { return append(new WithColumn(name, expr)); }

    /** Polars {@code with_columns} — add/replace multiple expression columns. */
    public LazyDataFrame withColumns(Expression... exprs) {
        LazyDataFrame cur = this;
        if (exprs == null) return cur;
        for (Expression e : exprs) {
            cur = cur.withColumn(e.suggestedName(), e);
        }
        return cur;
    }

    public LazyDataFrame filter(Expression condition) { return append(new Filter(condition)); }
    public LazyDataFrame sort(Expression... by) { return append(new Sort(by, null, null)); }
    public LazyDataFrame sort(Expression by, boolean nullsLast, boolean maintainOrder) {
        return append(new Sort(new Expression[]{by}, nullsLast, maintainOrder));
    }
    public LazyDataFrame sort(String col, boolean descending) {
        Expression key = descending ? Functions.desc(col) : Functions.asc(col);
        return sort(key);
    }
    public LazyDataFrame limit(int n) { return append(new Limit(n)); }
    public LazyDataFrame head(int n)  { return append(new Head(n)); }
    public LazyDataFrame tail(int n)  { return append(new Tail(n)); }
    public LazyDataFrame drop(String... cols) { return append(new Drop(cols)); }
    public LazyDataFrame rename(String oldName, String newName) { return append(new Rename(oldName, newName)); }
    public LazyDataFrame cache() { return append(new Cache()); }
    public LazyDataFrame unique() { return append(new Unique()); }
    public LazyDataFrame topK(int k, String col, boolean reverse) { return append(new TopK(k, col, reverse)); }
    public LazyDataFrame topK(int k, String col, boolean descending, boolean nullLast, boolean maintainOrder) {
        return topK(k, col, !descending);
    }
    public LazyDataFrame setSorted(String col, boolean descending, boolean nullsLast) {
        return append(new SetSorted(col, descending, nullsLast));
    }
    public LazyDataFrame dropNulls() { return append(new DropNulls()); }
    public LazyDataFrame concat(LazyDataFrame... others) { return append(new Concat(others)); }

    /**
     * Polars {@code optimization_toggle} analogue — when {@code enabled=false},
     * subsequent {@link #collect()} skips the optimizer (same as {@link #collectNoOptimization()}).
     * Stored as a no-op plan marker for explain() visibility.
     */
    public LazyDataFrame optimizationToggle(boolean enabled) {
        return append(new OptimizationToggle(enabled));
    }

    /**
     * Polars {@code map_batches(function)} — lazy batch UDF.
     * When collected with {@link #collectStreaming(int)}, the UDF is applied
     * per streaming chunk (true batch map); otherwise once on the full frame.
     */
    public LazyDataFrame mapBatches(java.util.function.Function<DataFrame, DataFrame> fn) {
        return mapBatches(fn, null);
    }

    public LazyDataFrame mapBatches(java.util.function.Function<DataFrame, DataFrame> fn, String label) {
        Objects.requireNonNull(fn, "fn");
        return append(new MapBatches(fn, label));
    }

    /** Start a lazy group-by; call {@link LazyGroupBy#agg(Expression...)} to append plan op. */
    public LazyGroupBy groupBy(String... keys) {
        return new LazyGroupBy(this, keys);
    }

    public static final class LazyGroupBy {
        private final LazyDataFrame parent;
        private final String[] keys;
        LazyGroupBy(LazyDataFrame parent, String[] keys) {
            this.parent = parent;
            this.keys = keys;
        }
        public LazyDataFrame agg(Expression... aggs) {
            return parent.append(new GroupByAgg(keys, aggs));
        }
        /** Polars {@code group_by().head(n)} — materializes via eager GroupedDataFrame. */
        public LazyDataFrame head(int n) {
            return parent.append(new GroupByHead(keys, n));
        }
        public LazyDataFrame tail(int n) {
            return parent.append(new GroupByTail(keys, n));
        }
    }

    public String explain() { return explain(true); }

    public String explain(boolean optimized) {
        List<LazyOp> ops = optimized ? LazyOptimizer.optimize(plan) : plan;
        StringBuilder sb = new StringBuilder();
        sb.append("LazyFrame plan (").append(ops.size()).append(" ops")
          .append(optimized ? ", optimized" : ", logical").append("):\n");
        sb.append("  SOURCE [").append(source.rowCount()).append(" rows x ")
          .append(source.columnCount()).append(" cols]\n");
        int i = 0;
        for (LazyOp op : ops) {
            sb.append("  ").append(++i).append(". ").append(op.describe()).append('\n');
        }
        return sb.toString();
    }

    /** JSON-ish plan dump (Polars {@code explain(format="json")} light analogue). */
    public String explainJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"source_rows\":").append(source.rowCount())
          .append(",\"source_cols\":").append(source.columnCount())
          .append(",\"ops\":[");
        for (int i = 0; i < plan.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append("{\"i\":").append(i)
              .append(",\"op\":\"").append(escapeJson(plan.get(i).describe())).append("\"}");
        }
        sb.append("]}");
        return sb.toString();
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    /** Expose plan size for tests. */
    public int planSize() { return plan.size(); }
    public int optimizedPlanSize() { return LazyOptimizer.optimize(plan).size(); }

    @Override public String toString() {
        return "LazyDataFrame(ops=" + plan.size() + ", sourceRows=" + source.rowCount() + ")";
    }

    static DataFrame sortByExpressions(DataFrame df, Expression[] by) {
        int n = df.rowCount();
        Object[][] keys = new Object[n][by.length];
        boolean[] descending = new boolean[by.length];
        for (int c = 0; c < by.length; c++) {
            descending[c] = by[c].isSortDescending();
            for (int r = 0; r < n; r++) {
                keys[r][c] = by[c].eval(r, df);
            }
        }
        List<Integer> order = IntStream.range(0, n).boxed()
            .sorted((a, b) -> {
                for (int c = 0; c < by.length; c++) {
                    int cmp = Expression.compareVals(keys[a][c], keys[b][c]);
                    if (cmp != 0) return descending[c] ? -cmp : cmp;
                }
                return 0;
            })
            .collect(Collectors.toList());
        int[] idx = order.stream().mapToInt(Integer::intValue).toArray();
        return df.loc(idx);
    }
}
