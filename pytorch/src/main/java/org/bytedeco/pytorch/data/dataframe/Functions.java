package org.bytedeco.pytorch.data.dataframe;

import java.time.*;

import org.bytedeco.pytorch.data.dataframe.window.WindowFrame;
import org.bytedeco.pytorch.data.dataframe.window.WindowFunction;
import org.bytedeco.pytorch.data.dataframe.window.WindowSpec;

/**
 * Top-level Polars-style factory functions.
 *
 * <p>Typical usage:
 * <pre>
 *   import static org.bytedeco.pytorch.data.dataframe.Functions.*;
 *
 *   df.lazy()
 *     .withColumn("x2", col("x").plus(lit(1)))
 *     .filter(col("x").gt(lit(0)))
 *     .sort(asc("x"))
 *     .collect();
 *
 *   // Spark-style windows
 *   WindowSpec w = window().partitionBy("dept").orderBy(asc("salary"));
 *   df.withColumn("rn", row_number().over(w));
 * </pre>
 */
public final class Functions {
    private Functions() {}

    /** Column reference expression. */
    public static Expression col(String name) {
        return Expression.col(name);
    }

    /** Literal expression. */
    public static Expression lit(Object value) {
        return Expression.lit(value);
    }

    /** Ascending sort key for {@link LazyDataFrame#sort(Expression...)}. */
    public static Expression asc(String name) {
        return new Expression.SortKeyExpr(col(name), false);
    }

    /** Descending sort key for {@link LazyDataFrame#sort(Expression...)}. */
    public static Expression desc(String name) {
        return new Expression.SortKeyExpr(col(name), true);
    }

    /** Ascending sort key from an existing expression. */
    public static Expression asc(Expression expr) {
        return new Expression.SortKeyExpr(expr, false);
    }

    /** Descending sort key from an existing expression. */
    public static Expression desc(Expression expr) {
        return new Expression.SortKeyExpr(expr, true);
    }

    /**
     * Start a when/then/otherwise chain.
     * <pre>
     *   when(col("x").gt(0), "pos").when(col("x").lt(0), "neg").otherwise("zero")
     * </pre>
     */
    public static When when(Expression condition, Object value) {
        return new When().when(condition, value);
    }

    // ---- free aggregation helpers (mirror scala-polars functions) ----

    public static Expression sum(String name)     { return col(name).sum(); }
    public static Expression sum(Expression e)    { return e.sum(); }
    public static Expression min(String name)     { return col(name).min(); }
    public static Expression min(Expression e)    { return e.min(); }
    public static Expression max(String name)     { return col(name).max(); }
    public static Expression max(Expression e)    { return e.max(); }
    public static Expression mean(String name)    { return col(name).mean(); }
    public static Expression mean(Expression e)   { return e.mean(); }
    public static Expression median(String name)  { return col(name).median(); }
    public static Expression median(Expression e) { return e.median(); }
    public static Expression std(String name)     { return col(name).std(); }
    public static Expression std(Expression e)    { return e.std(); }
    public static Expression count(String name)   { return col(name).count(); }
    public static Expression count(Expression e)  { return e.count(); }
    public static Expression first(String name)   { return col(name).first(); }
    public static Expression first(Expression e)  { return e.first(); }
    public static Expression last(String name)    { return col(name).last(); }
    public static Expression last(Expression e)   { return e.last(); }
    public static Expression nUnique(String name) { return col(name).nUnique(); }
    public static Expression nUnique(Expression e){ return e.nUnique(); }
    public static Expression cumSum(String name)  { return col(name).cumSum(); }
    public static Expression cumSum(Expression e) { return e.cumSum(); }
    public static Expression cumMin(String name)  { return col(name).cumMin(); }
    public static Expression cumMin(Expression e) { return e.cumMin(); }
    public static Expression cumMax(String name)  { return col(name).cumMax(); }
    public static Expression cumMax(Expression e) { return e.cumMax(); }
    public static Expression abs(String name)     { return col(name).abs(); }
    public static Expression abs(Expression e)    { return e.abs(); }
    public static Expression sqrt(String name)    { return col(name).sqrt(); }
    public static Expression sqrt(Expression e)   { return e.sqrt(); }
    public static Expression square(String name)  { return col(name).square(); }
    public static Expression square(Expression e) { return e.square(); }
    public static Expression exp(String name)     { return col(name).exp(); }
    public static Expression exp(Expression e)    { return e.exp(); }
    public static Expression log(String name)     { return col(name).log(); }
    public static Expression log(Expression e)    { return e.log(); }
    public static Expression log2(String name)    { return col(name).log2(); }
    public static Expression log2(Expression e)   { return e.log2(); }
    public static Expression log10(String name)   { return col(name).log10(); }
    public static Expression log10(Expression e)  { return e.log10(); }
    public static Expression mode(String name)    { return col(name).mode(); }
    public static Expression mode(Expression e)   { return e.mode(); }
    public static Expression argMax(String name)  { return col(name).argMax(); }
    public static Expression argMax(Expression e) { return e.argMax(); }
    public static Expression argMin(String name)  { return col(name).argMin(); }
    public static Expression argMin(Expression e) { return e.argMin(); }
    public static Expression unique(String name)  { return col(name).unique(); }
    public static Expression unique(Expression e) { return e.unique(); }
    /** Series-style rank (not window rank — use {@link #rank()} for Spark window). */
    public static Expression seriesRank(String name) { return col(name).rank(); }
    public static Expression seriesRank(Expression e){ return e.rank(); }
    public static Expression quantile(String name, double q) { return col(name).quantile(q); }
    public static Expression quantile(Expression e, double q) { return e.quantile(q); }
    public static Expression var(String name)     { return col(name).var(); }
    public static Expression var(Expression e)    { return e.var(); }
    public static Expression product(String name) { return col(name).product(); }
    public static Expression product(Expression e){ return e.product(); }
    public static Expression len(String name)     { return col(name).len(); }
    public static Expression len(Expression e)    { return e.len(); }

    public static Expression date(int year, int month, int day) {
        return lit(LocalDate.of(year, month, day));
    }

    public static Expression datetime(int year, int month, int day, int hour, int minute, int second) {
        return lit(LocalDateTime.of(year, month, day, hour, minute, second));
    }

    public static Expression time(int hour, int minute, int second) {
        return lit(LocalTime.of(hour, minute, second));
    }

    // ================================================================
    // AI / multimodal embedding (Daft functions.ai.*)
    // ================================================================

    /** Embed text column — see {@link org.bytedeco.pytorch.data.dataframe.ai.AiFunctions#embedText}. */
    public static Expression embedText(String column, String modelId) {
        return org.bytedeco.pytorch.data.dataframe.ai.AiFunctions.embedText(column, modelId);
    }

    public static Expression embedImage(String column, String modelId) {
        return org.bytedeco.pytorch.data.dataframe.ai.AiFunctions.embedImage(column, modelId);
    }

    public static Expression embedAudio(String column, String modelId) {
        return org.bytedeco.pytorch.data.dataframe.ai.AiFunctions.embedAudio(column, modelId);
    }

    public static Expression embedVideo(String column, String modelId) {
        return org.bytedeco.pytorch.data.dataframe.ai.AiFunctions.embedVideo(column, modelId);
    }

    public static Expression embed(String column, String modelId) {
        return org.bytedeco.pytorch.data.dataframe.ai.AiFunctions.embed(column, modelId);
    }

    // ================================================================
    // Spark-style window functions
    // ================================================================

    /** Empty window spec; chain {@code .partitionBy(...).orderBy(...).rowsBetween(...)}. */
    public static WindowSpec window() {
        return WindowSpec.empty();
    }

    public static WindowSpec partitionBy(String... cols) {
        return window().partitionBy(cols);
    }

    public static long unboundedPreceding() { return WindowFrame.UNBOUNDED_PRECEDING; }
    public static long unboundedFollowing() { return WindowFrame.UNBOUNDED_FOLLOWING; }
    public static long currentRow() { return WindowFrame.CURRENT_ROW; }

    public static Expression row_number() {
        return Expression.windowFn(WindowFunction.rowNumber());
    }
    public static Expression rowNumber() { return row_number(); }

    public static Expression rank() {
        return Expression.windowFn(WindowFunction.rank());
    }

    public static Expression dense_rank() {
        return Expression.windowFn(WindowFunction.denseRank());
    }
    public static Expression denseRank() { return dense_rank(); }

    public static Expression percent_rank() {
        return Expression.windowFn(WindowFunction.percentRank());
    }
    public static Expression percentRank() { return percent_rank(); }

    public static Expression ntile(int n) {
        return Expression.windowFn(WindowFunction.ntile(n));
    }

    public static Expression cume_dist() {
        return Expression.windowFn(WindowFunction.cumeDist());
    }
    public static Expression cumeDist() { return cume_dist(); }

    public static Expression lag(String colName, int n) {
        return lag(col(colName), n, null);
    }
    public static Expression lag(Expression e, int n) {
        return lag(e, n, null);
    }
    public static Expression lag(Expression e, int n, Object defaultValue) {
        return Expression.windowFn(WindowFunction.lag(e, n, defaultValue));
    }

    public static Expression lead(String colName, int n) {
        return lead(col(colName), n, null);
    }
    public static Expression lead(Expression e, int n) {
        return lead(e, n, null);
    }
    public static Expression lead(Expression e, int n, Object defaultValue) {
        return Expression.windowFn(WindowFunction.lead(e, n, defaultValue));
    }
}

