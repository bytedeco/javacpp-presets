package org.bytedeco.pytorch.dataframe;
import org.bytedeco.pytorch.dataframe.dtype.ListViewData;
import org.bytedeco.pytorch.dataframe.dtype.StructData;
import org.bytedeco.pytorch.dataframe.io.ComplexCellCodec;
import org.bytedeco.pytorch.dataframe.window.WindowExecutor;
import org.bytedeco.pytorch.dataframe.window.WindowFunction;
import org.bytedeco.pytorch.dataframe.window.WindowSpec;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.time.*;
import java.time.format.DateTimeFormatter;

/**
 * Polars-style expression tree evaluated against a {@link DataFrame}.
 *
 * <p>Build with {@link Functions#col(String)}, {@link Functions#lit(Object)}, then chain:
 * <pre>
 *   col("x").plus(lit(1)).lessThan(col("y"))
 * </pre>
 *
 * <p>Java method names mirror scala-polars {@code Column} (plus, lessThanEqualTo, …)
 * and also provide short aliases used in DataFrame javadoc (gt, lt, eq, …).
 */
public class Expression {

    // ---- sort helpers (used by LazyDataFrame.sort) ----
    boolean sortDescending;
    boolean isSortKey;

    /** Public for cross-package expression nodes (e.g. {@code ai.AiFunctions.EmbedExpr}). */
    public Expression() {}

    // ================================================================
    // Factory helpers (also on Functions)
    // ================================================================

    public static Expression col(String name) { return new ColExpr(name); }

    public static Expression lit(Object value) { return new LitExpr(value); }

    // ================================================================
    // Binary arithmetic
    // ================================================================

    public Expression plus(Object other)     { return bin(Op.PLUS, other); }
    public Expression minus(Object other)    { return bin(Op.MINUS, other); }
    public Expression multiply(Object other) { return bin(Op.MULTIPLY, other); }
    public Expression divide(Object other)   { return bin(Op.DIVIDE, other); }
    public Expression mod(Object other)      { return bin(Op.MODULUS, other); }
    public Expression floorDiv(Object other) { return bin(Op.FLOOR_DIV, other); }
    public Expression pow(double exponent)   { return new PowExpr(this, exponent); }
    /** Element-wise power with a scalar or another expression (Polars {@code .pow}). */
    public Expression pow(Object other) {
        if (other instanceof Number) return pow(((Number) other).doubleValue());
        return new PowExprBinary(this, toExpr(other));
    }
    /** Square — alias for {@code pow(2)}. */
    public Expression square() { return pow(2.0); }

    // ================================================================
    // Comparison
    // ================================================================

    public Expression equalTo(Object other)            { return bin(Op.EQ, other); }
    public Expression notEqualTo(Object other)         { return bin(Op.NE, other); }
    public Expression lessThan(Object other)           { return bin(Op.LT, other); }
    public Expression lessThanEqualTo(Object other)    { return bin(Op.LE, other); }
    public Expression greaterThan(Object other)        { return bin(Op.GT, other); }
    public Expression greaterThanEqualTo(Object other) { return bin(Op.GE, other); }

    /** Short aliases (DataFrame javadoc style). */
    public Expression eq(Object other) { return equalTo(other); }
    public Expression ne(Object other) { return notEqualTo(other); }
    public Expression lt(Object other) { return lessThan(other); }
    public Expression le(Object other) { return lessThanEqualTo(other); }
    public Expression gt(Object other) { return greaterThan(other); }
    public Expression ge(Object other) { return greaterThanEqualTo(other); }

    // ================================================================
    // Boolean
    // ================================================================

    public Expression and(Object other) { return bin(Op.AND, other); }
    public Expression or(Object other)  { return bin(Op.OR, other); }
    public Expression not()             { return new UnaryExpr(UnaryOp.NOT, this); }

    // ================================================================
    // Null / NaN checks
    // ================================================================

    public Expression isNull()    { return new UnaryExpr(UnaryOp.IS_NULL, this); }
    public Expression isNotNull() { return new UnaryExpr(UnaryOp.IS_NOT_NULL, this); }
    /** Pandas {@code isna()} alias for {@link #isNull()}. */
    public Expression isna()      { return isNull(); }
    /** Pandas {@code notna()} alias for {@link #isNotNull()}. */
    public Expression notna()     { return isNotNull(); }
    public Expression isNaN()     { return new UnaryExpr(UnaryOp.IS_NAN, this); }
    public Expression isNotNaN()  { return new UnaryExpr(UnaryOp.IS_NOT_NAN, this); }
    public Expression isFinite()  { return new UnaryExpr(UnaryOp.IS_FINITE, this); }
    public Expression isInfinite(){ return new UnaryExpr(UnaryOp.IS_INFINITE, this); }

    // ================================================================
    // Predicates
    // ================================================================

    public Expression isBetween(Object lower, Object upper) {
        return new BetweenExpr(this, toExpr(lower), toExpr(upper));
    }

    public Expression isIn(Object... values) {
        Expression[] exprs = new Expression[values.length];
        for (int i = 0; i < values.length; i++) exprs[i] = toExpr(values[i]);
        return new IsInExpr(this, exprs);
    }

    public Expression like(String pattern) {
        return new LikeExpr(this, pattern);
    }

    // ================================================================
    // Math unary
    // ================================================================

    public Expression neg()    { return new UnaryExpr(UnaryOp.NEG, this); }
    /** Daft {@code negate()} alias for {@link #neg()}. */
    public Expression negate() { return neg(); }
    public Expression abs()    { return new UnaryExpr(UnaryOp.ABS, this); }
    public Expression floor()  { return new UnaryExpr(UnaryOp.FLOOR, this); }
    public Expression ceil()   { return new UnaryExpr(UnaryOp.CEIL, this); }
    public Expression sign()   { return new UnaryExpr(UnaryOp.SIGN, this); }
    public Expression sqrt()   { return new UnaryExpr(UnaryOp.SQRT, this); }
    public Expression cbrt()   { return new UnaryExpr(UnaryOp.CBRT, this); }
    public Expression exp()    { return new UnaryExpr(UnaryOp.EXP, this); }
    /** {@code exp(x) - 1} with better precision near 0 (Daft {@code expm1}). */
    public Expression expm1()  { return new UnaryExpr(UnaryOp.EXPM1, this); }
    public Expression log()    { return new UnaryExpr(UnaryOp.LOG, this); }
    public Expression log(double base) { return new LogExpr(this, base); }
    public Expression log10()  { return log(10.0); }
    /** Base-2 logarithm. */
    public Expression log2()   { return log(2.0); }
    public Expression log1p()  { return new UnaryExpr(UnaryOp.LOG1P, this); }
    public Expression round()  { return round(0); }
    public Expression round(int decimals) { return new RoundExpr(this, decimals); }
    public Expression truncate() { return truncate(0); }
    public Expression truncate(int decimals) { return new TruncateExpr(this, decimals); }
    /** Alias of {@link #truncate()} — Pandas/Polars {@code trunc}. */
    public Expression trunc() { return truncate(); }
    public Expression trunc(int decimals) { return truncate(decimals); }

    public Expression sin()    { return new UnaryExpr(UnaryOp.SIN, this); }
    public Expression cos()    { return new UnaryExpr(UnaryOp.COS, this); }
    public Expression tan()    { return new UnaryExpr(UnaryOp.TAN, this); }
    public Expression cot()    { return new UnaryExpr(UnaryOp.COT, this); }
    public Expression arcsin() { return new UnaryExpr(UnaryOp.ARCSIN, this); }
    public Expression arccos() { return new UnaryExpr(UnaryOp.ARCCOS, this); }
    public Expression arctan() { return new UnaryExpr(UnaryOp.ARCTAN, this); }
    public Expression sinh()   { return new UnaryExpr(UnaryOp.SINH, this); }
    public Expression cosh()   { return new UnaryExpr(UnaryOp.COSH, this); }
    public Expression tanh()   { return new UnaryExpr(UnaryOp.TANH, this); }
    public Expression arcsinh(){ return new UnaryExpr(UnaryOp.ARCSINH, this); }
    public Expression arccosh(){ return new UnaryExpr(UnaryOp.ARCCOSH, this); }
    public Expression arctanh(){ return new UnaryExpr(UnaryOp.ARCTANH, this); }
    public Expression degrees(){ return new UnaryExpr(UnaryOp.DEGREES, this); }
    public Expression radians(){ return new UnaryExpr(UnaryOp.RADIANS, this); }

    // ================================================================
    // Column-window-ish (row-position aware)
    // ================================================================

    public Expression shift() { return shift(1); }
    public Expression shift(long periods) { return new ShiftExpr(this, periods); }

    public Expression cumSum() { return cumSum(false); }
    public Expression cumSum(boolean reverse) { return new CumSumExpr(this, reverse); }
    /** Cumulative minimum (Pandas {@code cummin} / Polars {@code cum_min}). */
    public Expression cumMin() { return cumMin(false); }
    public Expression cumMin(boolean reverse) { return new CumMinExpr(this, reverse); }
    public Expression cummin() { return cumMin(); }
    /** Cumulative maximum (Pandas {@code cummax} / Polars {@code cum_max}). */
    public Expression cumMax() { return cumMax(false); }
    public Expression cumMax(boolean reverse) { return new CumMaxExpr(this, reverse); }
    public Expression cummax() { return cumMax(); }

    public Expression diff() { return diff(1); }
    public Expression diff(long n) { return new DiffExpr(this, n); }

    public Expression pctChange() { return pctChange(1); }
    public Expression pctChange(long n) { return new PctChangeExpr(this, n); }

    // ---- Rolling / expanding (expression form) ----

    /** Rolling sum over a fixed window (Polars {@code rolling_sum}). */
    public Expression rollingSum(int window) { return new RollingExpr(this, window, RollingOp.SUM); }
    public Expression rollingMean(int window) { return new RollingExpr(this, window, RollingOp.MEAN); }
    public Expression rollingMax(int window) { return new RollingExpr(this, window, RollingOp.MAX); }
    public Expression rollingMin(int window) { return new RollingExpr(this, window, RollingOp.MIN); }
    public Expression rollingStd(int window) { return new RollingExpr(this, window, RollingOp.STD); }
    public Expression rollingVar(int window) { return new RollingExpr(this, window, RollingOp.VAR); }

    /** Expanding-window mean from the start of the series (Pandas {@code expanding().mean()}). */
    public Expression expandingMean() { return new ExpandingExpr(this, ExpandingOp.MEAN); }
    public Expression expandingSum()  { return new ExpandingExpr(this, ExpandingOp.SUM); }
    public Expression expandingMin()  { return new ExpandingExpr(this, ExpandingOp.MIN); }
    public Expression expandingMax()  { return new ExpandingExpr(this, ExpandingOp.MAX); }
    public Expression expandingStd()  { return new ExpandingExpr(this, ExpandingOp.STD); }

    /**
     * Rank values (average method by default, ascending).
     * Methods: {@code "average"}, {@code "min"}, {@code "max"}, {@code "dense"}, {@code "ordinal"}.
     */
    public Expression rank() { return rank("average", true); }
    public Expression rank(String method) { return rank(method, true); }
    public Expression rank(String method, boolean ascending) {
        return new RankExpr(this, method == null ? "average" : method, ascending);
    }

    public Expression clip(Object lower, Object upper) {
        return new ClipExpr(this, toExpr(lower), toExpr(upper));
    }
    public Expression clipMin(Object lower) { return new ClipMinExpr(this, toExpr(lower)); }
    public Expression clipMax(Object upper) { return new ClipMaxExpr(this, toExpr(upper)); }

    public Expression fillNull(Object value) { return new FillNullExpr(this, toExpr(value)); }
    /** Pandas {@code fillna} alias for {@link #fillNull(Object)}. */
    public Expression fillna(Object value) { return fillNull(value); }
    /** Daft {@code if_null} alias for {@link #fillNull(Object)}. */
    public Expression ifNull(Object value) { return fillNull(value); }

    /**
     * Return the first non-null among {@code this} and {@code others}
     * (Daft/SQL {@code coalesce}).
     */
    public Expression coalesce(Object... others) {
        Expression[] rest = new Expression[others == null ? 0 : others.length];
        if (others != null) {
            for (int i = 0; i < others.length; i++) rest[i] = toExpr(others[i]);
        }
        return new CoalesceExpr(this, rest);
    }

    /** Element-wise value replace (Daft {@code replace}). */
    public Expression replace(Object fromVal, Object toVal) {
        return new ReplaceExpr(this, toExpr(fromVal), toExpr(toVal));
    }

    /**
     * Element-wise filter: keep value where {@code cond} is true, else null
     * (Pandas {@code Series.where(cond)}).
     */
    public Expression where(Object cond) {
        return new WhereExpr(this, toExpr(cond));
    }

    /**
     * Filter rows of this expression by a boolean condition (Polars-style).
     * Evaluates to the same length with non-matching rows set to null.
     * Prefer {@link DataFrame#filter(Expression)} for table-level row filtering.
     */
    public Expression filter(Object cond) {
        return where(cond);
    }

    // ================================================================
    // Aggregations (whole-column reduce → scalar broadcast per row)
    // ================================================================

    public Expression sum()    { return new AggExpr(this, AggOp.SUM); }
    public Expression min()    { return new AggExpr(this, AggOp.MIN); }
    public Expression max()    { return new AggExpr(this, AggOp.MAX); }
    public Expression mean()   { return new AggExpr(this, AggOp.MEAN); }
    public Expression median() { return new AggExpr(this, AggOp.MEDIAN); }
    public Expression std()    { return std(1); }
    public Expression std(int ddof) { return new AggExpr(this, AggOp.STD, ddof); }
    public Expression var()    { return variance(1); }
    public Expression variance() { return variance(1); }
    public Expression variance(int ddof) { return new AggExpr(this, AggOp.VAR, ddof); }
    public Expression product(){ return new AggExpr(this, AggOp.PRODUCT); }
    public Expression count()  { return new AggExpr(this, AggOp.COUNT); }
    public Expression len()    { return new AggExpr(this, AggOp.LEN); }
    public Expression nUnique(){ return new AggExpr(this, AggOp.NUNIQUE); }
    public Expression nullCount() { return new AggExpr(this, AggOp.NULL_COUNT); }
    public Expression first()  { return new AggExpr(this, AggOp.FIRST); }
    public Expression last()   { return new AggExpr(this, AggOp.LAST); }
    public Expression quantile(double q) { return new QuantileExpr(this, q); }
    public Expression any()    { return new AggExpr(this, AggOp.ANY); }
    public Expression all()    { return new AggExpr(this, AggOp.ALL); }

    /** Index of the first maximum value (Pandas {@code argmax} / Polars {@code arg_max}). */
    public Expression argMax() { return new AggExpr(this, AggOp.ARGMAX); }
    public Expression argmax() { return argMax(); }
    /** Index of the first minimum value (Pandas {@code argmin} / Polars {@code arg_min}). */
    public Expression argMin() { return new AggExpr(this, AggOp.ARGMIN); }
    public Expression argmin() { return argMin(); }
    /** Most frequent value (first mode on ties). */
    public Expression mode()   { return new AggExpr(this, AggOp.MODE); }

    /**
     * Unique values of this expression as a variable-length column
     * (Polars {@code unique()}). Prefer {@link DataFrame#unique(String...)} for table de-dup.
     */
    public Expression unique() { return new UniqueExpr(this); }

    /**
     * Frequency counts of each distinct value — evaluates to a 2-column DataFrame
     * via {@link #valueCountsAsDataFrame(DataFrame)} (Polars {@code value_counts}).
     * As a column expression, returns the count of the value at each row.
     */
    public Expression valueCounts() { return new ValueCountsExpr(this); }

    /**
     * Materialize value_counts as a small DataFrame with columns {@code value} and {@code count}.
     */
    public DataFrame valueCountsAsDataFrame(DataFrame df) {
        Map<Object, Long> counts = new LinkedHashMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object v = eval(i, df);
            if (v != null) counts.merge(v, 1L, Long::sum);
        }
        DataFrame out = DataFrame.create();
        out.addColumn("value", Column.DType.STRING);
        out.addColumn("count", Column.DType.INT64);
        for (Map.Entry<Object, Long> e : counts.entrySet()) {
            int ri = out.addEmptyRow();
            out.set(ri, "value", e.getKey() == null ? null : e.getKey().toString());
            out.set(ri, "count", e.getValue());
        }
        return out;
    }

    /**
     * Bin continuous values into discrete intervals (Pandas {@code cut} / Polars {@code cut}).
     * {@code bins} are right-edges of intervals; labels are optional category names.
     * <pre>
     *   col("age").cut(new double[]{18, 35, 60, 100}, new String[]{"youth","adult","senior","elder"})
     * </pre>
     */
    public Expression cut(double[] bins) { return cut(bins, null); }
    public Expression cut(double[] bins, String[] labels) {
        return new CutExpr(this, bins, labels);
    }

    /**
     * Equal-frequency binning (Pandas {@code qcut} / Polars {@code qcut}).
     * @param quantiles number of quantile bins (e.g. 4 = quartiles)
     */
    public Expression qcut(int quantiles) {
        return new QcutExpr(this, quantiles);
    }

    /**
     * Fill nulls with a strategy: {@code "forward"} / {@code "backward"} / {@code "mean"} /
     * {@code "min"} / {@code "max"} / {@code "zero"} / {@code "one"}.
     */
    public Expression fillNull(String strategy) {
        return new FillNullStrategyExpr(this, strategy == null ? "forward" : strategy);
    }

    /** Hash of the value (seeded). Polars {@code hash}. */
    public Expression hash(long seed) { return new HashExpr(this, seed); }
    public Expression hash() { return hash(0L); }

    /** Mark duplicate elements (first occurrence false). Polars {@code is_duplicated}. */
    public Expression isDuplicated() { return new IsDuplicatedExpr(this); }

    /** True on the first occurrence of each distinct value. Polars {@code is_first_distinct}. */
    public Expression isFirstDistinct() { return new IsFirstDistinctExpr(this, true); }

    /** True on the last occurrence of each distinct value. Polars {@code is_last_distinct}. */
    public Expression isLastDistinct() { return new IsFirstDistinctExpr(this, false); }

    /**
     * Shrink numeric dtype to the smallest that can hold all values
     * (Polars {@code shrink_dtype}). Evaluates to the same values; cast is advisory
     * via {@link #suggestedName()} — prefer {@link DataFrame} level for real cast.
     */
    public Expression shrinkDtype() { return new ShrinkDtypeExpr(this); }

    /** Dense rank alias (method=dense). */
    public Expression rankDense() { return rank("dense", true); }

    /** Exponentially weighted moving mean (Polars {@code ewm_mean}). */
    public Expression ewmMean(double alpha) { return new EwmMeanExpr(this, alpha); }

    /**
     * Element-wise UDF (Polars {@code map_elements}). Applied row by row;
     * return type inferred from first non-null result (default STRING).
     */
    public Expression mapElements(java.util.function.Function<Object, Object> fn) {
        return new MapElementsExpr(this, fn);
    }

    /**
     * Cumulative count of non-null values (optionally reverse).
     * Polars {@code cum_count}.
     */
    public Expression cumCount() { return cumCount(false); }
    public Expression cumCount(boolean reverse) { return new CumCountExpr(this, reverse); }

    /**
     * Round to {@code n} significant figures (Polars {@code round_sig_figs}).
     */
    public Expression roundSigFigs(int n) { return new RoundSigFigsExpr(this, n); }

    /**
     * Window convenience: {@code expr.over("g1","g2")} partitions by columns
     * with default whole-partition frame.
     */
    public Expression over(String... partitionBy) {
        return over(WindowSpec.empty()
            .partitionBy(partitionBy));
    }

    // ---- list / struct namespaces (Polars) ----

    /** Polars-style list namespace: {@code col("tags").list().first()}. */
    public ListNameSpace list() { return new ListNameSpace(this); }

    /** Polars-style struct namespace: {@code col("meta").struct().field("k")}. */
    public StructNameSpace struct() { return new StructNameSpace(this); }

    // ---- horizontal (static multi-column) ----

    /** Row-wise max across expressions (Polars {@code max_horizontal}). */
    public static Expression maxHorizontal(Expression... exprs) {
        return new HorizontalExpr(HorizontalOp.MAX, exprs);
    }
    public static Expression minHorizontal(Expression... exprs) {
        return new HorizontalExpr(HorizontalOp.MIN, exprs);
    }
    public static Expression sumHorizontal(Expression... exprs) {
        return new HorizontalExpr(HorizontalOp.SUM, exprs);
    }

    // ================================================================
    // Meta
    // ================================================================

    public Expression alias(String name) { return new AliasExpr(this, name); }
    public Expression as(String name)    { return alias(name); }

    public Expression cast(Column.DType dtype) { return new CastExpr(this, dtype); }
    /** Daft/Pandas {@code astype} alias for {@link #cast(Column.DType)}. */
    public Expression astype(Column.DType dtype) { return cast(dtype); }

    /**
     * Apply this expression as a window function over {@code spec}.
     * <ul>
     *   <li>On an aggregation ({@code col("x").sum().over(w)}) → framed partition aggregate</li>
     *   <li>On a ranking/offset window expr → evaluate with the given spec</li>
     *   <li>Otherwise wraps as FIRST over the value expression</li>
     * </ul>
     */
    public Expression over(WindowSpec spec) {
        if (this instanceof AggExpr) {
            AggExpr a = (AggExpr) this;
            WindowFunction.Kind kind = aggOpToWindowKind(a.op);
            WindowFunction wf =
                WindowFunction.agg(kind, a.child);
            return new WindowExpr(wf, spec);
        }
        if (this instanceof WindowExpr) {
            WindowExpr w = (WindowExpr) this;
            return new WindowExpr(w.fn, spec);
        }
        // treat as value expression → FIRST over window
        WindowFunction wf =
            WindowFunction.agg(
                WindowFunction.Kind.FIRST, this);
        return new WindowExpr(wf, spec);
    }

    private static WindowFunction.Kind aggOpToWindowKind(AggOp op) {
        switch (op) {
            case SUM: return WindowFunction.Kind.SUM;
            case MEAN: return WindowFunction.Kind.MEAN;
            case MIN: return WindowFunction.Kind.MIN;
            case MAX: return WindowFunction.Kind.MAX;
            case COUNT: case LEN: return WindowFunction.Kind.COUNT;
            case STD: return WindowFunction.Kind.STD;
            case VAR: return WindowFunction.Kind.VAR;
            case FIRST: return WindowFunction.Kind.FIRST;
            case LAST: return WindowFunction.Kind.LAST;
            case MEDIAN: return WindowFunction.Kind.MEDIAN;
            case PRODUCT: return WindowFunction.Kind.PRODUCT;
            default: return WindowFunction.Kind.FIRST;
        }
    }

    /** String namespace (subset of Polars .str). */
    public StrNameSpace str() { return new StrNameSpace(this); }

    /** Temporal namespace (Polars-style .dt) — also available via {@link #dt()}. */
    // note: dt() is defined later near evaluate API

    /** Image namespace (Daft-style multimodal). */
    public MultimodalExpressions.ImageNameSpace image() {
        return new MultimodalExpressions.ImageNameSpace(this);
    }
    /** Audio namespace (Daft-style multimodal). */
    public MultimodalExpressions.AudioNameSpace audio() {
        return new MultimodalExpressions.AudioNameSpace(this);
    }
    /** Video namespace (Daft-style multimodal). */
    public MultimodalExpressions.VideoNameSpace video() {
        return new MultimodalExpressions.VideoNameSpace(this);
    }
    /** Tensor / embedding namespace (Daft-style). */
    public MultimodalExpressions.TensorNameSpace tensor() {
        return new MultimodalExpressions.TensorNameSpace(this);
    }
    /** Text / NLP namespace (Daft-style). */
    public MultimodalExpressions.TextNameSpace text() {
        return new MultimodalExpressions.TextNameSpace(this);
    }

    // ================================================================
    // Evaluation API
    // ================================================================

    /**
     * Evaluate this expression for a single row.
     * Subclasses override; default throws.
     */
    public Object eval(int row, DataFrame df) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + ".eval not implemented");
    }

    /**
     * Evaluate over the whole DataFrame, producing a new Column.
     * Aggregation / window expressions may override for efficiency.
     */
    public Column evaluate(DataFrame df) {
        int n = df.rowCount();
        List<Object> data = new ArrayList<>(n);
        Column.DType dtype = null;
        String name = suggestedName();
        for (int i = 0; i < n; i++) {
            Object v = eval(i, df);
            data.add(v);
            if (dtype == null && v != null) dtype = inferDType(v);
        }
        if (dtype == null) dtype = Column.DType.STRING;
        return new Column(name, dtype, data);
    }

    /** Suggested output column name (overridden by ColExpr / AliasExpr). */
    public String suggestedName() { return "expr"; }

    /** Column names this expression depends on (for optimizer pushdown). */
    public Set<String> referencedColumns() { return Set.of(); }

    /** Temporal namespace (Polars-style .dt). */
    public DtNameSpace dt() { return new DtNameSpace(this); }

    /** True if this expression is a sort key created via asc/desc. */
    public boolean isSortKey() { return isSortKey; }
    public boolean isSortDescending() { return sortDescending; }

    /** If this is a sort key, return the underlying expression; otherwise {@code this}. */
    public Expression sortChild() { return this; }

    // ================================================================
    // Internals
    // ================================================================

    private Expression bin(Op op, Object other) {
        return new BinaryExpr(op, this, toExpr(other));
    }

    static Expression toExpr(Object value) {
        if (value instanceof Expression) return (Expression) value;
        return new LitExpr(value);
    }

    static Column.DType inferDType(Object v) {
        if (v == null) return Column.DType.STRING;
        if (v instanceof Integer) return Column.DType.INT32;
        if (v instanceof Long) return Column.DType.INT64;
        if (v instanceof Float) return Column.DType.FLOAT32;
        if (v instanceof Double) return Column.DType.FLOAT64;
        if (v instanceof Boolean) return Column.DType.BOOLEAN;
        if (v instanceof LocalDate) return Column.DType.DATE;
        if (v instanceof LocalTime) return Column.DType.TIME;
        if (v instanceof Instant || v instanceof LocalDateTime || v instanceof ZonedDateTime)
            return Column.DType.DATETIME;
        if (v instanceof Duration) return Column.DType.DURATION;
        if (v instanceof Number) return Column.DType.FLOAT64;
        return Column.DType.STRING;
    }

    static Double toDouble(Object v) {
        if (v == null) return null;
        if (v instanceof Number) return ((Number) v).doubleValue();
        if (v instanceof Boolean) return ((Boolean) v) ? 1.0 : 0.0;
        try { return Double.parseDouble(v.toString()); }
        catch (Exception e) { return null; }
    }

    static int compareVals(Object a, Object b) {
        if (a == null && b == null) return 0;
        if (a == null) return -1;
        if (b == null) return 1;
        if (a instanceof Number && b instanceof Number)
            return Double.compare(((Number) a).doubleValue(), ((Number) b).doubleValue());
        if (a instanceof Boolean && b instanceof Boolean)
            return Boolean.compare((Boolean) a, (Boolean) b);
        return a.toString().compareTo(b.toString());
    }

    static Object promoteNumber(double result, Object left, Object right) {
        // Keep integer if both sides integer and result is whole
        boolean leftInt = left instanceof Integer || left instanceof Long;
        boolean rightInt = right == null || right instanceof Integer || right instanceof Long;
        if (leftInt && rightInt && result == Math.rint(result)
                && !Double.isInfinite(result) && !Double.isNaN(result)) {
            long lr = (long) result;
            if (left instanceof Integer && (right == null || right instanceof Integer)
                    && lr >= Integer.MIN_VALUE && lr <= Integer.MAX_VALUE) {
                return (int) lr;
            }
            return lr;
        }
        return result;
    }

    /** Truthiness used by boolean expressions and {@link DataFrame#filter}. */
    public static boolean isTrue(Object v) {
        if (v == null) return false;
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof Number) return ((Number) v).doubleValue() != 0;
        return true;
    }

    // ================================================================
    // Operator enums
    // ================================================================

    enum Op {
        PLUS, MINUS, MULTIPLY, DIVIDE, MODULUS, FLOOR_DIV,
        EQ, NE, LT, LE, GT, GE, AND, OR
    }

    enum UnaryOp {
        NOT, IS_NULL, IS_NOT_NULL, IS_NAN, IS_NOT_NAN, IS_FINITE, IS_INFINITE,
        NEG, ABS, FLOOR, CEIL, SIGN, SQRT, CBRT, EXP, EXPM1, LOG, LOG1P,
        SIN, COS, TAN, COT, ARCSIN, ARCCOS, ARCTAN,
        SINH, COSH, TANH, ARCSINH, ARCCOSH, ARCTANH, DEGREES, RADIANS
    }

    enum AggOp {
        SUM, MIN, MAX, MEAN, MEDIAN, STD, VAR, PRODUCT, COUNT, LEN,
        NUNIQUE, NULL_COUNT, FIRST, LAST, ANY, ALL, ARGMAX, ARGMIN, MODE
    }

    enum RollingOp { SUM, MEAN, MAX, MIN, STD, VAR }
    enum ExpandingOp { SUM, MEAN, MAX, MIN, STD }

    // ================================================================
    // Node types
    // ================================================================

    static final class ColExpr extends Expression {
        final String name;
        ColExpr(String name) { this.name = name; }
        @Override public Object eval(int row, DataFrame df) {
            return df.get(row, name);
        }
        @Override public String suggestedName() { return name; }
        @Override public Set<String> referencedColumns() { return Set.of(name); }
        @Override public String toString() { return "col(" + name + ")"; }
    }

    static final class LitExpr extends Expression {
        final Object value;
        LitExpr(Object value) { this.value = value; }
        @Override public Object eval(int row, DataFrame df) { return value; }
        @Override public String suggestedName() {
            return value == null ? "null" : "lit(" + value + ")";
        }
        @Override public String toString() { return "lit(" + value + ")"; }
    }

    static final class BinaryExpr extends Expression {
        final Op op;
        final Expression left, right;
        BinaryExpr(Op op, Expression left, Expression right) {
            this.op = op; this.left = left; this.right = right;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object a = left.eval(row, df);
            Object b = right.eval(row, df);
            switch (op) {
                case AND: return isTrue(a) && isTrue(b);
                case OR:  return isTrue(a) || isTrue(b);
                case EQ:
                    if (a == null || b == null) return a == null && b == null;
                    return compareVals(a, b) == 0;
                case NE:
                    if (a == null || b == null) return !(a == null && b == null);
                    return compareVals(a, b) != 0;
                case LT: case LE: case GT: case GE: {
                    if (a == null || b == null) return null;
                    int c = compareVals(a, b);
                    return switch (op) {
                        case LT -> c < 0;
                        case LE -> c <= 0;
                        case GT -> c > 0;
                        case GE -> c >= 0;
                        default -> null;
                    };
                }
                case PLUS: case MINUS: case MULTIPLY: case DIVIDE: case MODULUS: case FLOOR_DIV: {
                    if (a == null || b == null) return null;
                    // string concat for PLUS if either is non-number string
                    if (op == Op.PLUS && (!(a instanceof Number) || !(b instanceof Number))
                            && !(a instanceof Boolean) && !(b instanceof Boolean)) {
                        return String.valueOf(a) + String.valueOf(b);
                    }
                    Double da = toDouble(a), db = toDouble(b);
                    if (da == null || db == null) return null;
                    double r = switch (op) {
                        case PLUS -> da + db;
                        case MINUS -> da - db;
                        case MULTIPLY -> da * db;
                        case DIVIDE -> db == 0 ? Double.NaN : da / db;
                        case MODULUS -> db == 0 ? Double.NaN : da % db;
                        case FLOOR_DIV -> db == 0 ? Double.NaN : Math.floor(da / db);
                        default -> Double.NaN;
                    };
                    return promoteNumber(r, a, b);
                }
                default: return null;
            }
        }
        @Override public String suggestedName() {
            return left.suggestedName() + "_" + op.name().toLowerCase() + "_" + right.suggestedName();
        }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(left.referencedColumns());
            s.addAll(right.referencedColumns());
            return s;
        }
        @Override public String toString() { return "(" + left + " " + op + " " + right + ")"; }
    }

    static final class UnaryExpr extends Expression {
        final UnaryOp op;
        final Expression child;
        UnaryExpr(UnaryOp op, Expression child) { this.op = op; this.child = child; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            switch (op) {
                case NOT: return !isTrue(v);
                case IS_NULL: return v == null;
                case IS_NOT_NULL: return v != null;
                case IS_NAN: {
                    if (v == null) return false;
                    Double d = toDouble(v);
                    return d != null && Double.isNaN(d);
                }
                case IS_NOT_NAN: {
                    if (v == null) return true;
                    Double d = toDouble(v);
                    return d == null || !Double.isNaN(d);
                }
                case IS_FINITE: {
                    if (v == null) return null;
                    Double d = toDouble(v);
                    return d != null && Double.isFinite(d);
                }
                case IS_INFINITE: {
                    if (v == null) return null;
                    Double d = toDouble(v);
                    return d != null && Double.isInfinite(d);
                }
                default: {
                    if (v == null) return null;
                    Double d = toDouble(v);
                    if (d == null) return null;
                    double r = switch (op) {
                        case NEG -> -d;
                        case ABS -> Math.abs(d);
                        case FLOOR -> Math.floor(d);
                        case CEIL -> Math.ceil(d);
                        case SIGN -> d > 0 ? 1 : (d < 0 ? -1 : 0);
                        case SQRT -> Math.sqrt(d);
                        case CBRT -> Math.cbrt(d);
                        case EXP -> Math.exp(d);
                        case EXPM1 -> Math.expm1(d);
                        case LOG -> Math.log(d);
                        case LOG1P -> Math.log1p(d);
                        case SIN -> Math.sin(d);
                        case COS -> Math.cos(d);
                        case TAN -> Math.tan(d);
                        case COT -> 1.0 / Math.tan(d);
                        case ARCSIN -> Math.asin(d);
                        case ARCCOS -> Math.acos(d);
                        case ARCTAN -> Math.atan(d);
                        case SINH -> Math.sinh(d);
                        case COSH -> Math.cosh(d);
                        case TANH -> Math.tanh(d);
                        case ARCSINH -> Math.log(d + Math.sqrt(d * d + 1));
                        case ARCCOSH -> Math.log(d + Math.sqrt(d * d - 1));
                        case ARCTANH -> 0.5 * Math.log((1 + d) / (1 - d));
                        case DEGREES -> Math.toDegrees(d);
                        case RADIANS -> Math.toRadians(d);
                        default -> d;
                    };
                    if (op == UnaryOp.NEG || op == UnaryOp.ABS || op == UnaryOp.SIGN
                            || op == UnaryOp.FLOOR || op == UnaryOp.CEIL) {
                        return promoteNumber(r, v, null);
                    }
                    return r;
                }
            }
        }
        @Override public String suggestedName() {
            return op.name().toLowerCase() + "(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
        @Override public String toString() { return op + "(" + child + ")"; }
    }

    static final class AliasExpr extends Expression {
        final Expression child;
        final String name;
        AliasExpr(Expression child, String name) { this.child = child; this.name = name; }
        @Override public Object eval(int row, DataFrame df) { return child.eval(row, df); }
        @Override public Column evaluate(DataFrame df) {
            Column c = child.evaluate(df);
            return new Column(name, c.dtype(), c.data());
        }
        @Override public String suggestedName() { return name; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
        @Override public String toString() { return child + ".alias(" + name + ")"; }
    }

    /**
     * Window function expression. Requires a {@link WindowSpec}
     * (via {@link #over}); evaluates to a full column aligned with source rows.
     */
    static final class WindowExpr extends Expression {
        final WindowFunction fn;
        final WindowSpec spec;
        private DataFrame cachedDf;
        private Column cachedCol;

        WindowExpr(WindowFunction fn,
                   WindowSpec spec) {
            this.fn = fn;
            this.spec = spec;
        }

        private Column compute(DataFrame df) {
            if (cachedDf == df && cachedCol != null) return cachedCol;
            if (spec == null) {
                throw new IllegalStateException(
                    "Window function " + fn.suggestedName() + " requires .over(WindowSpec)");
            }
            cachedCol = WindowExecutor.evaluate(df, fn, spec);
            cachedDf = df;
            return cachedCol;
        }

        @Override public Object eval(int row, DataFrame df) {
            return compute(df).get(row);
        }

        @Override public Column evaluate(DataFrame df) {
            return compute(df);
        }

        @Override public String suggestedName() { return fn.suggestedName(); }

        @Override public String toString() {
            return fn.suggestedName() + ".over(" + spec + ")";
        }
    }

    /** Factory used by {@link Functions} for ranking/offset window functions (spec applied later). */
    static Expression windowFn(WindowFunction fn) {
        return new WindowExpr(fn, null);
    }

    static final class CastExpr extends Expression {
        final Expression child;
        final Column.DType dtype;
        CastExpr(Expression child, Column.DType dtype) { this.child = child; this.dtype = dtype; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            return switch (dtype) {
                case INT32 -> {
                    Double d = toDouble(v);
                    yield d == null ? null : d.intValue();
                }
                case INT64 -> {
                    Double d = toDouble(v);
                    yield d == null ? null : d.longValue();
                }
                case FLOAT32 -> {
                    Double d = toDouble(v);
                    yield d == null ? null : d.floatValue();
                }
                case FLOAT64 -> toDouble(v);
                case BOOLEAN -> isTrue(v);
                case STRING -> v.toString();
                default -> v;
            };
        }
        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            List<Object> data = new ArrayList<>(n);
            for (int i = 0; i < n; i++) data.add(eval(i, df));
            return new Column(suggestedName(), dtype, data);
        }
        @Override public String suggestedName() { return child.suggestedName(); }
        @Override public String toString() { return child + ".cast(" + dtype + ")"; }
    }

    static final class BetweenExpr extends Expression {
        final Expression child, lower, upper;
        BetweenExpr(Expression child, Expression lower, Expression upper) {
            this.child = child; this.lower = lower; this.upper = upper;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            Object lo = lower.eval(row, df);
            Object hi = upper.eval(row, df);
            if (v == null || lo == null || hi == null) return null;
            return compareVals(v, lo) >= 0 && compareVals(v, hi) <= 0;
        }
        @Override public String suggestedName() { return child.suggestedName() + "_between"; }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(child.referencedColumns());
            s.addAll(lower.referencedColumns()); s.addAll(upper.referencedColumns());
            return s;
        }
    }

    static final class IsInExpr extends Expression {
        final Expression child;
        final Expression[] values;
        IsInExpr(Expression child, Expression[] values) {
            this.child = child; this.values = values;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            for (Expression e : values) {
                Object o = e.eval(row, df);
                if (o != null && compareVals(v, o) == 0) return true;
            }
            return false;
        }
        @Override public String suggestedName() { return child.suggestedName() + "_is_in"; }
    }

    static final class LikeExpr extends Expression {
        final Expression child;
        final Pattern pattern;
        final String raw;
        LikeExpr(Expression child, String sqlPattern) {
            this.child = child;
            this.raw = sqlPattern;
            // SQL LIKE: % → .*, _ → .
            StringBuilder re = new StringBuilder("^");
            for (int i = 0; i < sqlPattern.length(); i++) {
                char c = sqlPattern.charAt(i);
                if (c == '%') re.append(".*");
                else if (c == '_') re.append('.');
                else if ("\\.[]{}()*+-?^$|".indexOf(c) >= 0) re.append('\\').append(c);
                else re.append(c);
            }
            re.append('$');
            this.pattern = Pattern.compile(re.toString(), Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            return pattern.matcher(v.toString()).matches();
        }
        @Override public String suggestedName() { return child.suggestedName() + "_like"; }
        @Override public String toString() { return child + ".like(" + raw + ")"; }
    }

    static final class PowExpr extends Expression {
        final Expression child;
        final double exponent;
        PowExpr(Expression child, double exponent) { this.child = child; this.exponent = exponent; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Double d = toDouble(v);
            return d == null ? null : Math.pow(d, exponent);
        }
        @Override public String suggestedName() { return "pow(" + child.suggestedName() + ")"; }
    }

    static final class PowExprBinary extends Expression {
        final Expression left, right;
        PowExprBinary(Expression left, Expression right) { this.left = left; this.right = right; }
        @Override public Object eval(int row, DataFrame df) {
            Object a = left.eval(row, df);
            Object b = right.eval(row, df);
            if (a == null || b == null) return null;
            Double da = toDouble(a), db = toDouble(b);
            if (da == null || db == null) return null;
            return Math.pow(da, db);
        }
        @Override public String suggestedName() {
            return "pow(" + left.suggestedName() + "," + right.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(left.referencedColumns());
            s.addAll(right.referencedColumns());
            return s;
        }
    }

    static final class LogExpr extends Expression {
        final Expression child;
        final double base;
        LogExpr(Expression child, double base) { this.child = child; this.base = base; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Double d = toDouble(v);
            if (d == null) return null;
            return Math.log(d) / Math.log(base);
        }
        @Override public String suggestedName() { return "log(" + child.suggestedName() + ")"; }
    }

    static final class RoundExpr extends Expression {
        final Expression child;
        final int decimals;
        RoundExpr(Expression child, int decimals) { this.child = child; this.decimals = decimals; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Double d = toDouble(v);
            if (d == null) return null;
            double factor = Math.pow(10, decimals);
            // banker's round via Math.rint
            return Math.rint(d * factor) / factor;
        }
        @Override public String suggestedName() { return "round(" + child.suggestedName() + ")"; }
    }

    static final class TruncateExpr extends Expression {
        final Expression child;
        final int decimals;
        TruncateExpr(Expression child, int decimals) { this.child = child; this.decimals = decimals; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Double d = toDouble(v);
            if (d == null) return null;
            double factor = Math.pow(10, decimals);
            return d >= 0 ? Math.floor(d * factor) / factor : Math.ceil(d * factor) / factor;
        }
        @Override public String suggestedName() { return "truncate(" + child.suggestedName() + ")"; }
    }

    static final class ShiftExpr extends Expression {
        final Expression child;
        final long periods;
        ShiftExpr(Expression child, long periods) { this.child = child; this.periods = periods; }
        @Override public Object eval(int row, DataFrame df) {
            long src = (long) row - periods;
            if (src < 0 || src >= df.rowCount()) return null;
            return child.eval((int) src, df);
        }
        @Override public String suggestedName() { return "shift(" + child.suggestedName() + ")"; }
    }

    static final class CumSumExpr extends Expression {
        final Expression child;
        final boolean reverse;
        // cache per-dataframe evaluation
        private DataFrame cachedDf;
        private double[] prefix;
        CumSumExpr(Expression child, boolean reverse) { this.child = child; this.reverse = reverse; }
        private void ensure(DataFrame df) {
            if (cachedDf == df && prefix != null) return;
            int n = df.rowCount();
            prefix = new double[n];
            if (!reverse) {
                double run = 0;
                for (int i = 0; i < n; i++) {
                    Double d = toDouble(child.eval(i, df));
                    if (d != null && !Double.isNaN(d)) run += d;
                    prefix[i] = run;
                }
            } else {
                double run = 0;
                for (int i = n - 1; i >= 0; i--) {
                    Double d = toDouble(child.eval(i, df));
                    if (d != null && !Double.isNaN(d)) run += d;
                    prefix[i] = run;
                }
            }
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            return prefix[row];
        }
        @Override public String suggestedName() { return "cumsum(" + child.suggestedName() + ")"; }
    }

    static final class CumMinExpr extends Expression {
        final Expression child;
        final boolean reverse;
        private DataFrame cachedDf;
        private Object[] prefix;
        CumMinExpr(Expression child, boolean reverse) { this.child = child; this.reverse = reverse; }
        private void ensure(DataFrame df) {
            if (cachedDf == df && prefix != null) return;
            int n = df.rowCount();
            prefix = new Object[n];
            if (!reverse) {
                Double run = null;
                for (int i = 0; i < n; i++) {
                    Double d = toDouble(child.eval(i, df));
                    if (d != null && !Double.isNaN(d)) {
                        run = run == null ? d : Math.min(run, d);
                    }
                    prefix[i] = run;
                }
            } else {
                Double run = null;
                for (int i = n - 1; i >= 0; i--) {
                    Double d = toDouble(child.eval(i, df));
                    if (d != null && !Double.isNaN(d)) {
                        run = run == null ? d : Math.min(run, d);
                    }
                    prefix[i] = run;
                }
            }
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            return prefix[row];
        }
        @Override public String suggestedName() { return "cummin(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class CumMaxExpr extends Expression {
        final Expression child;
        final boolean reverse;
        private DataFrame cachedDf;
        private Object[] prefix;
        CumMaxExpr(Expression child, boolean reverse) { this.child = child; this.reverse = reverse; }
        private void ensure(DataFrame df) {
            if (cachedDf == df && prefix != null) return;
            int n = df.rowCount();
            prefix = new Object[n];
            if (!reverse) {
                Double run = null;
                for (int i = 0; i < n; i++) {
                    Double d = toDouble(child.eval(i, df));
                    if (d != null && !Double.isNaN(d)) {
                        run = run == null ? d : Math.max(run, d);
                    }
                    prefix[i] = run;
                }
            } else {
                Double run = null;
                for (int i = n - 1; i >= 0; i--) {
                    Double d = toDouble(child.eval(i, df));
                    if (d != null && !Double.isNaN(d)) {
                        run = run == null ? d : Math.max(run, d);
                    }
                    prefix[i] = run;
                }
            }
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            return prefix[row];
        }
        @Override public String suggestedName() { return "cummax(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class DiffExpr extends Expression {
        final Expression child;
        final long n;
        DiffExpr(Expression child, long n) { this.child = child; this.n = n; }
        @Override public Object eval(int row, DataFrame df) {
            long prev = (long) row - n;
            if (prev < 0) return null;
            Object cur = child.eval(row, df);
            Object p = child.eval((int) prev, df);
            if (cur == null || p == null) return null;
            Double dc = toDouble(cur), dp = toDouble(p);
            if (dc == null || dp == null) return null;
            return promoteNumber(dc - dp, cur, p);
        }
        @Override public String suggestedName() { return "diff(" + child.suggestedName() + ")"; }
    }

    static final class PctChangeExpr extends Expression {
        final Expression child;
        final long n;
        PctChangeExpr(Expression child, long n) { this.child = child; this.n = n; }
        @Override public Object eval(int row, DataFrame df) {
            long prev = (long) row - n;
            if (prev < 0) return null;
            Object cur = child.eval(row, df);
            Object p = child.eval((int) prev, df);
            if (cur == null || p == null) return null;
            Double dc = toDouble(cur), dp = toDouble(p);
            if (dc == null || dp == null || dp == 0) return Double.NaN;
            return (dc - dp) / dp;
        }
        @Override public String suggestedName() { return "pct_change(" + child.suggestedName() + ")"; }
    }

    static final class ClipExpr extends Expression {
        final Expression child, lower, upper;
        ClipExpr(Expression child, Expression lower, Expression upper) {
            this.child = child; this.lower = lower; this.upper = upper;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Object lo = lower.eval(row, df);
            Object hi = upper.eval(row, df);
            if (lo != null && compareVals(v, lo) < 0) return lo;
            if (hi != null && compareVals(v, hi) > 0) return hi;
            return v;
        }
        @Override public String suggestedName() { return "clip(" + child.suggestedName() + ")"; }
    }

    static final class ClipMinExpr extends Expression {
        final Expression child, lower;
        ClipMinExpr(Expression child, Expression lower) { this.child = child; this.lower = lower; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Object lo = lower.eval(row, df);
            if (lo != null && compareVals(v, lo) < 0) return lo;
            return v;
        }
        @Override public String suggestedName() { return "clip_min(" + child.suggestedName() + ")"; }
    }

    static final class ClipMaxExpr extends Expression {
        final Expression child, upper;
        ClipMaxExpr(Expression child, Expression upper) { this.child = child; this.upper = upper; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Object hi = upper.eval(row, df);
            if (hi != null && compareVals(v, hi) > 0) return hi;
            return v;
        }
        @Override public String suggestedName() { return "clip_max(" + child.suggestedName() + ")"; }
    }

    static final class FillNullExpr extends Expression {
        final Expression child, fill;
        FillNullExpr(Expression child, Expression fill) { this.child = child; this.fill = fill; }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            return v != null ? v : fill.eval(row, df);
        }
        @Override public String suggestedName() { return child.suggestedName(); }
    }

    static final class CoalesceExpr extends Expression {
        final Expression first;
        final Expression[] rest;
        CoalesceExpr(Expression first, Expression[] rest) {
            this.first = first;
            this.rest = rest == null ? new Expression[0] : rest;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = first.eval(row, df);
            if (v != null) return v;
            for (Expression e : rest) {
                v = e.eval(row, df);
                if (v != null) return v;
            }
            return null;
        }
        @Override public String suggestedName() { return "coalesce(" + first.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(first.referencedColumns());
            for (Expression e : rest) s.addAll(e.referencedColumns());
            return s;
        }
    }

    static final class ReplaceExpr extends Expression {
        final Expression child, from, to;
        ReplaceExpr(Expression child, Expression from, Expression to) {
            this.child = child; this.from = from; this.to = to;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            Object f = from.eval(row, df);
            if (v == null && f == null) return to.eval(row, df);
            if (v != null && f != null && compareVals(v, f) == 0) return to.eval(row, df);
            if (v != null && f != null && String.valueOf(v).equals(String.valueOf(f))) return to.eval(row, df);
            return v;
        }
        @Override public String suggestedName() { return "replace(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(child.referencedColumns());
            s.addAll(from.referencedColumns());
            s.addAll(to.referencedColumns());
            return s;
        }
    }

    static final class QuantileExpr extends Expression {
        final Expression child;
        final double q;
        private DataFrame cachedDf;
        private Object cached;
        QuantileExpr(Expression child, double q) { this.child = child; this.q = q; }
        private Object compute(DataFrame df) {
            if (cachedDf == df) return cached;
            List<Double> vals = new ArrayList<>();
            for (int i = 0; i < df.rowCount(); i++) {
                Double d = toDouble(child.eval(i, df));
                if (d != null && !Double.isNaN(d)) vals.add(d);
            }
            Collections.sort(vals);
            if (vals.isEmpty()) { cached = null; }
            else {
                double pos = q * (vals.size() - 1);
                int lo = (int) Math.floor(pos), hi = (int) Math.ceil(pos);
                cached = (vals.get(lo) + vals.get(hi)) / 2.0;
            }
            cachedDf = df;
            return cached;
        }
        @Override public Object eval(int row, DataFrame df) { return compute(df); }
        @Override public String suggestedName() { return "quantile(" + child.suggestedName() + ")"; }
    }

    static final class AggExpr extends Expression {
        final Expression child;
        final AggOp op;
        final int ddof;
        private DataFrame cachedDf;
        private Object cached;
        AggExpr(Expression child, AggOp op) { this(child, op, 1); }
        AggExpr(Expression child, AggOp op, int ddof) {
            this.child = child; this.op = op; this.ddof = ddof;
        }
        private Object compute(DataFrame df) {
            if (cachedDf == df) return cached;
            int n = df.rowCount();
            switch (op) {
                case LEN: cached = (long) n; break;
                case FIRST: cached = n > 0 ? child.eval(0, df) : null; break;
                case LAST: cached = n > 0 ? child.eval(n - 1, df) : null; break;
                case COUNT: {
                    long c = 0;
                    for (int i = 0; i < n; i++) if (child.eval(i, df) != null) c++;
                    cached = c;
                    break;
                }
                case NULL_COUNT: {
                    long c = 0;
                    for (int i = 0; i < n; i++) if (child.eval(i, df) == null) c++;
                    cached = c;
                    break;
                }
                case NUNIQUE: {
                    Set<Object> s = new HashSet<>();
                    for (int i = 0; i < n; i++) {
                        Object v = child.eval(i, df);
                        if (v != null) s.add(v);
                    }
                    cached = (long) s.size();
                    break;
                }
                case ARGMAX: {
                    double best = Double.NEGATIVE_INFINITY;
                    long bestIdx = -1;
                    for (int i = 0; i < n; i++) {
                        Double d = toDouble(child.eval(i, df));
                        if (d != null && !Double.isNaN(d) && d > best) {
                            best = d;
                            bestIdx = i;
                        }
                    }
                    cached = bestIdx < 0 ? null : bestIdx;
                    break;
                }
                case ARGMIN: {
                    double best = Double.POSITIVE_INFINITY;
                    long bestIdx = -1;
                    for (int i = 0; i < n; i++) {
                        Double d = toDouble(child.eval(i, df));
                        if (d != null && !Double.isNaN(d) && d < best) {
                            best = d;
                            bestIdx = i;
                        }
                    }
                    cached = bestIdx < 0 ? null : bestIdx;
                    break;
                }
                case MODE: {
                    Map<Object, Long> freq = new LinkedHashMap<>();
                    for (int i = 0; i < n; i++) {
                        Object v = child.eval(i, df);
                        if (v != null) freq.merge(v, 1L, Long::sum);
                    }
                    Object best = null;
                    long bestC = -1;
                    for (Map.Entry<Object, Long> e : freq.entrySet()) {
                        if (e.getValue() > bestC) {
                            bestC = e.getValue();
                            best = e.getKey();
                        }
                    }
                    cached = best;
                    break;
                }
                case ANY: {
                    boolean r = false;
                    for (int i = 0; i < n; i++) {
                        Object v = child.eval(i, df);
                        if (v != null && isTrue(v)) { r = true; break; }
                    }
                    cached = r;
                    break;
                }
                case ALL: {
                    boolean r = true;
                    for (int i = 0; i < n; i++) {
                        Object v = child.eval(i, df);
                        if (v != null && !isTrue(v)) { r = false; break; }
                    }
                    cached = r;
                    break;
                }
                default: {
                    List<Double> vals = new ArrayList<>();
                    for (int i = 0; i < n; i++) {
                        Double d = toDouble(child.eval(i, df));
                        if (d != null && !Double.isNaN(d)) vals.add(d);
                    }
                    if (vals.isEmpty()) { cached = null; break; }
                    cached = switch (op) {
                        case SUM -> vals.stream().mapToDouble(Double::doubleValue).sum();
                        case MIN -> vals.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
                        case MAX -> vals.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
                        case MEAN -> vals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
                        case PRODUCT -> {
                            double p = 1;
                            for (double d : vals) p *= d;
                            yield p;
                        }
                        case MEDIAN -> {
                            List<Double> s = new ArrayList<>(vals);
                            Collections.sort(s);
                            int m = s.size();
                            yield m % 2 == 0 ? (s.get(m/2-1) + s.get(m/2)) / 2.0 : s.get(m/2);
                        }
                        case STD, VAR -> {
                            double mean = vals.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                            double ss = 0;
                            for (double d : vals) ss += (d - mean) * (d - mean);
                            int denom = vals.size() - ddof;
                            double variance = denom > 0 ? ss / denom : Double.NaN;
                            yield op == AggOp.STD ? Math.sqrt(variance) : variance;
                        }
                        default -> null;
                    };
                    break;
                }
            }
            cachedDf = df;
            return cached;
        }
        @Override public Object eval(int row, DataFrame df) { return compute(df); }
        @Override public String suggestedName() {
            return op.name().toLowerCase() + "(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
        @Override public String toString() { return suggestedName(); }
    }

    /** Sort-key expression wrapping a column reference. */
    static final class SortKeyExpr extends Expression {
        final Expression child;
        SortKeyExpr(Expression child, boolean descending) {
            this.child = child;
            this.sortDescending = descending;
            this.isSortKey = true;
        }
        @Override public Object eval(int row, DataFrame df) { return child.eval(row, df); }
        @Override public Expression sortChild() { return child; }
        @Override public String suggestedName() { return child.suggestedName(); }
        @Override public String toString() {
            return (sortDescending ? "desc(" : "asc(") + child + ")";
        }
    }

    /** When/then chain result. */
    static final class WhenThenExpr extends Expression {
        final List<Expression> conditions;
        final List<Expression> values;
        final Expression otherwise;
        WhenThenExpr(List<Expression> conditions, List<Expression> values, Expression otherwise) {
            this.conditions = conditions;
            this.values = values;
            this.otherwise = otherwise;
        }
        @Override public Object eval(int row, DataFrame df) {
            for (int i = 0; i < conditions.size(); i++) {
                if (isTrue(conditions.get(i).eval(row, df))) {
                    return values.get(i).eval(row, df);
                }
            }
            return otherwise == null ? null : otherwise.eval(row, df);
        }
        @Override public String suggestedName() { return "when"; }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>();
            for (Expression c : conditions) s.addAll(c.referencedColumns());
            for (Expression v : values) s.addAll(v.referencedColumns());
            if (otherwise != null) s.addAll(otherwise.referencedColumns());
            return s;
        }
        @Override public String toString() { return "when(...)"; }
    }

    // ================================================================
    // Additional elementwise / window / transform nodes
    // ================================================================

    static final class WhereExpr extends Expression {
        final Expression child, cond;
        WhereExpr(Expression child, Expression cond) { this.child = child; this.cond = cond; }
        @Override public Object eval(int row, DataFrame df) {
            return isTrue(cond.eval(row, df)) ? child.eval(row, df) : null;
        }
        @Override public String suggestedName() { return child.suggestedName(); }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(child.referencedColumns());
            s.addAll(cond.referencedColumns());
            return s;
        }
    }

    static final class RollingExpr extends Expression {
        final Expression child;
        final int window;
        final RollingOp op;
        private DataFrame cachedDf;
        private Object[] cached;
        RollingExpr(Expression child, int window, RollingOp op) {
            this.child = child;
            this.window = Math.max(1, window);
            this.op = op;
        }
        private void ensure(DataFrame df) {
            if (cachedDf == df && cached != null) return;
            int n = df.rowCount();
            cached = new Object[n];
            for (int i = 0; i < n; i++) {
                int start = Math.max(0, i - window + 1);
                List<Double> vals = new ArrayList<>(window);
                for (int j = start; j <= i; j++) {
                    Double d = toDouble(child.eval(j, df));
                    if (d != null && !Double.isNaN(d)) vals.add(d);
                }
                cached[i] = reduceWindow(vals, op);
            }
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            return cached[row];
        }
        @Override public String suggestedName() {
            return "rolling_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ExpandingExpr extends Expression {
        final Expression child;
        final ExpandingOp op;
        private DataFrame cachedDf;
        private Object[] cached;
        ExpandingExpr(Expression child, ExpandingOp op) {
            this.child = child;
            this.op = op;
        }
        private void ensure(DataFrame df) {
            if (cachedDf == df && cached != null) return;
            int n = df.rowCount();
            cached = new Object[n];
            List<Double> vals = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                Double d = toDouble(child.eval(i, df));
                if (d != null && !Double.isNaN(d)) vals.add(d);
                cached[i] = reduceExpanding(vals, op);
            }
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            return cached[row];
        }
        @Override public String suggestedName() {
            return "expanding_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    private static Object reduceWindow(List<Double> vals, RollingOp op) {
        if (vals.isEmpty()) return null;
        return switch (op) {
            case SUM -> vals.stream().mapToDouble(Double::doubleValue).sum();
            case MEAN -> vals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
            case MAX -> vals.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
            case MIN -> vals.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
            case STD, VAR -> {
                double mean = vals.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double ss = 0;
                for (double d : vals) ss += (d - mean) * (d - mean);
                int denom = vals.size() - 1;
                double variance = denom > 0 ? ss / denom : Double.NaN;
                yield op == RollingOp.STD ? Math.sqrt(variance) : variance;
            }
        };
    }

    private static Object reduceExpanding(List<Double> vals, ExpandingOp op) {
        if (vals.isEmpty()) return null;
        return switch (op) {
            case SUM -> vals.stream().mapToDouble(Double::doubleValue).sum();
            case MEAN -> vals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
            case MAX -> vals.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
            case MIN -> vals.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
            case STD -> {
                if (vals.size() < 2) yield null;
                double mean = vals.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double ss = 0;
                for (double d : vals) ss += (d - mean) * (d - mean);
                yield Math.sqrt(ss / (vals.size() - 1));
            }
        };
    }

    static final class RankExpr extends Expression {
        final Expression child;
        final String method;
        final boolean ascending;
        private DataFrame cachedDf;
        private Object[] ranks;
        RankExpr(Expression child, String method, boolean ascending) {
            this.child = child;
            this.method = method.toLowerCase(Locale.ROOT);
            this.ascending = ascending;
        }
        private void ensure(DataFrame df) {
            if (cachedDf == df && ranks != null) return;
            int n = df.rowCount();
            ranks = new Object[n];
            List<int[]> pairs = new ArrayList<>(n); // [index, ...]
            List<Double> vals = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                Double d = toDouble(child.eval(i, df));
                if (d == null || Double.isNaN(d)) {
                    ranks[i] = null;
                } else {
                    pairs.add(new int[]{i});
                    vals.add(d);
                }
            }
            Integer[] order = IntStream.range(0, pairs.size()).boxed()
                .sorted((a, b) -> {
                    int c = Double.compare(vals.get(a), vals.get(b));
                    return ascending ? c : -c;
                })
                .toArray(Integer[]::new);

            // assign ranks based on method
            int i = 0;
            while (i < order.length) {
                int j = i;
                while (j + 1 < order.length
                        && Double.compare(vals.get(order[j + 1]), vals.get(order[i])) == 0) {
                    j++;
                }
                // ties from i..j inclusive
                double avgRank = (i + 1 + j + 1) / 2.0;
                int minRank = i + 1;
                int maxRank = j + 1;
                int denseRank = 0;
                // dense: count distinct groups before
                // recompute dense as 1-based group index
                // We'll compute dense separately via running counter
                for (int k = i; k <= j; k++) {
                    int row = pairs.get(order[k])[0];
                    ranks[row] = switch (method) {
                        case "min" -> (double) minRank;
                        case "max" -> (double) maxRank;
                        case "ordinal" -> (double) (k + 1);
                        case "dense" -> null; // filled below
                        default -> avgRank; // average
                    };
                }
                i = j + 1;
            }
            if ("dense".equals(method)) {
                double prev = Double.NaN;
                int dense = 0;
                for (int oi : order) {
                    double v = vals.get(oi);
                    if (Double.isNaN(prev) || Double.compare(v, prev) != 0) {
                        dense++;
                        prev = v;
                    }
                    ranks[pairs.get(oi)[0]] = (double) dense;
                }
            }
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            return ranks[row];
        }
        @Override public String suggestedName() { return "rank(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class UniqueExpr extends Expression {
        final Expression child;
        private DataFrame cachedDf;
        private List<Object> uniques;
        UniqueExpr(Expression child) { this.child = child; }
        private void ensure(DataFrame df) {
            if (cachedDf == df && uniques != null) return;
            LinkedHashSet<Object> set = new LinkedHashSet<>();
            for (int i = 0; i < df.rowCount(); i++) {
                Object v = child.eval(i, df);
                if (v != null) set.add(v);
            }
            uniques = new ArrayList<>(set);
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            // broadcast first unique for row-wise eval; prefer evaluate()
            return uniques.isEmpty() ? null : uniques.get(0);
        }
        @Override public Column evaluate(DataFrame df) {
            ensure(df);
            Column.DType dtype = null;
            for (Object v : uniques) {
                if (v != null) { dtype = inferDType(v); break; }
            }
            if (dtype == null) dtype = Column.DType.STRING;
            return new Column("unique(" + child.suggestedName() + ")", dtype, new ArrayList<>(uniques));
        }
        @Override public String suggestedName() { return "unique(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ValueCountsExpr extends Expression {
        final Expression child;
        private DataFrame cachedDf;
        private Map<Object, Long> counts;
        ValueCountsExpr(Expression child) { this.child = child; }
        private void ensure(DataFrame df) {
            if (cachedDf == df && counts != null) return;
            counts = new LinkedHashMap<>();
            for (int i = 0; i < df.rowCount(); i++) {
                Object v = child.eval(i, df);
                if (v != null) counts.merge(v, 1L, Long::sum);
            }
            cachedDf = df;
        }
        @Override public Object eval(int row, DataFrame df) {
            ensure(df);
            Object v = child.eval(row, df);
            if (v == null) return null;
            Long c = counts.get(v);
            return c == null ? 0L : c;
        }
        @Override public String suggestedName() { return "value_counts(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class CutExpr extends Expression {
        final Expression child;
        final double[] bins;
        final String[] labels;
        CutExpr(Expression child, double[] bins, String[] labels) {
            this.child = child;
            this.bins = bins == null ? new double[0] : bins.clone();
            this.labels = labels;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Double d = toDouble(v);
            if (d == null || Double.isNaN(d)) return null;
            // bins are right edges: (-inf, b0], (b0, b1], ..., (b_{n-2}, b_{n-1}]
            int idx = -1;
            for (int i = 0; i < bins.length; i++) {
                if (d <= bins[i]) { idx = i; break; }
            }
            if (idx < 0) return null; // above all bins
            if (labels != null && idx < labels.length) return labels[idx];
            double left = idx == 0 ? Double.NEGATIVE_INFINITY : bins[idx - 1];
            return "(" + left + ", " + bins[idx] + "]";
        }
        @Override public String suggestedName() { return "cut(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    // ================================================================
    // String namespace
    // ================================================================

    public static final class StrNameSpace {
        private final Expression parent;
        StrNameSpace(Expression parent) { this.parent = parent; }

        public Expression toUpperCase() { return new StrExpr(parent, StrOp.UPPER); }
        public Expression toUppercase() { return toUpperCase(); }
        public Expression toLowerCase() { return new StrExpr(parent, StrOp.LOWER); }
        public Expression toLowercase() { return toLowerCase(); }
        public Expression length()      { return new StrExpr(parent, StrOp.LENGTH); }
        /** Polars {@code str.len_bytes()} alias. */
        public Expression lenBytes()    { return length(); }
        /** Polars {@code str.len_chars()} alias. */
        public Expression lenChars()    { return length(); }
        public Expression len()         { return length(); }
        public Expression contains(String s) { return new StrExpr(parent, StrOp.CONTAINS, s); }
        public Expression startsWith(String s) { return new StrExpr(parent, StrOp.STARTS_WITH, s); }
        public Expression endsWith(String s) { return new StrExpr(parent, StrOp.ENDS_WITH, s); }
        public Expression replace(String target, String replacement) {
            return new StrExpr(parent, StrOp.REPLACE, target, replacement);
        }
        public Expression replaceAll(String target, String replacement) {
            return new StrExpr(parent, StrOp.REPLACE_ALL, target, replacement);
        }
        public Expression replaceRegex(String pattern, String replacement) {
            return new StrExpr(parent, StrOp.REPLACE_REGEX, pattern, replacement);
        }
        public Expression strip() { return new StrExpr(parent, StrOp.STRIP); }
        public Expression lstrip() { return new StrExpr(parent, StrOp.LSTRIP); }
        public Expression rstrip() { return new StrExpr(parent, StrOp.RSTRIP); }
        public Expression slice(int start, int length) {
            return new StrExpr(parent, StrOp.SLICE, String.valueOf(start), String.valueOf(length));
        }
        public Expression split(String by) { return new StrExpr(parent, StrOp.SPLIT, by); }
        public Expression zfill(int width) { return new StrExpr(parent, StrOp.ZFILL, String.valueOf(width)); }
        public Expression padStart(int width, String ch) {
            return new StrExpr(parent, StrOp.PAD_START, String.valueOf(width), ch == null || ch.isEmpty() ? " " : ch.substring(0,1));
        }
        public Expression padEnd(int width, String ch) {
            return new StrExpr(parent, StrOp.PAD_END, String.valueOf(width), ch == null || ch.isEmpty() ? " " : ch.substring(0,1));
        }
        public Expression countMatches(String pattern) { return new StrExpr(parent, StrOp.COUNT_MATCHES, pattern); }
        public Expression extract(String pattern, int group) {
            return new StrExpr(parent, StrOp.EXTRACT, pattern, String.valueOf(group));
        }
        public Expression toInteger() { return new StrExpr(parent, StrOp.TO_INT); }
        public Expression toDouble() { return new StrExpr(parent, StrOp.TO_DOUBLE); }
        public Expression strptime(String format) { return new StrExpr(parent, StrOp.STRPTIME, format); }
        public Expression strftime(String format) { return new StrExpr(parent, StrOp.STRFTIME, format); }
    }

    enum StrOp {
        UPPER, LOWER, LENGTH, CONTAINS, STARTS_WITH, ENDS_WITH, REPLACE, REPLACE_ALL, REPLACE_REGEX,
        STRIP, LSTRIP, RSTRIP, SLICE, SPLIT, ZFILL, PAD_START, PAD_END, COUNT_MATCHES, EXTRACT,
        TO_INT, TO_DOUBLE, STRPTIME, STRFTIME
    }

    static final class StrExpr extends Expression {
        final Expression child;
        final StrOp op;
        final String arg1, arg2;
        StrExpr(Expression child, StrOp op) { this(child, op, null, null); }
        StrExpr(Expression child, StrOp op, String arg1) { this(child, op, arg1, null); }
        StrExpr(Expression child, StrOp op, String arg1, String arg2) {
            this.child = child; this.op = op; this.arg1 = arg1; this.arg2 = arg2;
        }
        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null && op != StrOp.STRFTIME) return null;
            String s = v == null ? null : v.toString();
            try {
                return switch (op) {
                    case UPPER -> s.toUpperCase();
                    case LOWER -> s.toLowerCase();
                    case LENGTH -> s.length();
                    case CONTAINS -> s.contains(arg1);
                    case STARTS_WITH -> s.startsWith(arg1);
                    case ENDS_WITH -> s.endsWith(arg1);
                    case REPLACE -> s.replace(arg1, arg2 == null ? "" : arg2);
                    case REPLACE_ALL -> s.replace(arg1, arg2 == null ? "" : arg2);
                    case REPLACE_REGEX -> s.replaceAll(arg1, arg2 == null ? "" : arg2);
                    case STRIP -> s.strip();
                    case LSTRIP -> s.stripLeading();
                    case RSTRIP -> s.stripTrailing();
                    case SLICE -> {
                        int start = Integer.parseInt(arg1);
                        int len = Integer.parseInt(arg2);
                        if (start < 0) start = Math.max(0, s.length() + start);
                        int end = Math.min(s.length(), start + len);
                        if (start >= s.length()) yield "";
                        yield s.substring(start, end);
                    }
                    case SPLIT -> {
                        // v1: join parts with '|' for STRING column compatibility
                        String[] parts = s.split(java.util.regex.Pattern.quote(arg1), -1);
                        yield String.join("|", parts);
                    }
                    case ZFILL -> {
                        int w = Integer.parseInt(arg1);
                        if (s.length() >= w) yield s;
                        yield "0".repeat(w - s.length()) + s;
                    }
                    case PAD_START -> {
                        int w = Integer.parseInt(arg1);
                        char ch = arg2.charAt(0);
                        if (s.length() >= w) yield s;
                        yield String.valueOf(ch).repeat(w - s.length()) + s;
                    }
                    case PAD_END -> {
                        int w = Integer.parseInt(arg1);
                        char ch = arg2.charAt(0);
                        if (s.length() >= w) yield s;
                        yield s + String.valueOf(ch).repeat(w - s.length());
                    }
                    case COUNT_MATCHES -> {
                        java.util.regex.Matcher m = Pattern.compile(arg1).matcher(s);
                        int c = 0;
                        while (m.find()) c++;
                        yield c;
                    }
                    case EXTRACT -> {
                        java.util.regex.Matcher m = Pattern.compile(arg1).matcher(s);
                        int g = Integer.parseInt(arg2);
                        yield m.find() ? m.group(g) : null;
                    }
                    case TO_INT -> {
                        String t = s.trim();
                        if (t.contains(".")) yield (int) Double.parseDouble(t);
                        yield Integer.parseInt(t);
                    }
                    case TO_DOUBLE -> Double.parseDouble(s.trim());
                    case STRPTIME -> {
                        DateTimeFormatter fmt = DateTimeFormatter.ofPattern(arg1);
                        try { yield LocalDateTime.parse(s, fmt); }
                        catch (Exception e1) {
                            try { yield LocalDate.parse(s, fmt).atStartOfDay(); }
                            catch (Exception e2) { yield null; }
                        }
                    }
                    case STRFTIME -> {
                        DateTimeFormatter fmt = DateTimeFormatter.ofPattern(arg1);
                        if (v instanceof LocalDate ld) yield ld.format(fmt);
                        if (v instanceof LocalDateTime ldt) yield ldt.format(fmt);
                        if (v instanceof Instant in) yield LocalDateTime.ofInstant(in, ZoneOffset.UTC).format(fmt);
                        if (v instanceof LocalTime lt) yield lt.format(fmt);
                        yield s;
                    }
                };
            } catch (Exception ex) {
                return null;
            }
        }
        @Override public String suggestedName() {
            return "str_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    // ================================================================
    // Temporal namespace
    // ================================================================

    public static final class DtNameSpace {
        private final Expression parent;
        DtNameSpace(Expression parent) { this.parent = parent; }
        public Expression year() { return new DtExpr(parent, DtOp.YEAR); }
        public Expression month() { return new DtExpr(parent, DtOp.MONTH); }
        public Expression day() { return new DtExpr(parent, DtOp.DAY); }
        public Expression hour() { return new DtExpr(parent, DtOp.HOUR); }
        public Expression minute() { return new DtExpr(parent, DtOp.MINUTE); }
        public Expression second() { return new DtExpr(parent, DtOp.SECOND); }
        public Expression epochMilli() { return new DtExpr(parent, DtOp.EPOCH_MILLI); }
        public Expression toLocalDate() { return new DtExpr(parent, DtOp.TO_DATE); }
        public Expression toLocalTime() { return new DtExpr(parent, DtOp.TO_TIME); }
    }

    enum DtOp { YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, EPOCH_MILLI, TO_DATE, TO_TIME }

    static final class DtExpr extends Expression {
        final Expression child;
        final DtOp op;
        DtExpr(Expression child, DtOp op) { this.child = child; this.op = op; }

        static LocalDateTime toLdt(Object v) {
            if (v == null) return null;
            if (v instanceof LocalDateTime ldt) return ldt;
            if (v instanceof LocalDate ld) return ld.atStartOfDay();
            if (v instanceof Instant in) return LocalDateTime.ofInstant(in, ZoneOffset.UTC);
            if (v instanceof ZonedDateTime zdt) return zdt.toLocalDateTime();
            if (v instanceof Number n) return LocalDateTime.ofInstant(Instant.ofEpochMilli(n.longValue()), ZoneOffset.UTC);
            try { return LocalDateTime.parse(v.toString()); } catch (Exception e) {}
            try { return LocalDate.parse(v.toString()).atStartOfDay(); } catch (Exception e) {}
            return null;
        }

        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            if (op == DtOp.TO_TIME) {
                if (v instanceof LocalTime lt) return lt;
                LocalDateTime ldt = toLdt(v);
                return ldt == null ? null : ldt.toLocalTime();
            }
            LocalDateTime ldt = toLdt(v);
            if (ldt == null) return null;
            return switch (op) {
                case YEAR -> ldt.getYear();
                case MONTH -> ldt.getMonthValue();
                case DAY -> ldt.getDayOfMonth();
                case HOUR -> ldt.getHour();
                case MINUTE -> ldt.getMinute();
                case SECOND -> ldt.getSecond();
                case EPOCH_MILLI -> ldt.toInstant(ZoneOffset.UTC).toEpochMilli();
                case TO_DATE -> ldt.toLocalDate();
                case TO_TIME -> ldt.toLocalTime();
            };
        }
        @Override public String suggestedName() {
            return "dt_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    // ================================================================
    // List / Struct namespaces (Polars advanced nested)
    // ================================================================

    /** Polars {@code Expr.list.*} namespace over list-typed cells. */
    public static final class ListNameSpace {
        private final Expression parent;
        ListNameSpace(Expression parent) { this.parent = parent; }

        public Expression first()  { return new ListExpr(parent, ListOp.FIRST, null, 0, 0); }
        public Expression last()   { return new ListExpr(parent, ListOp.LAST, null, 0, 0); }
        public Expression get(int index) { return new ListExpr(parent, ListOp.GET, null, index, 0); }
        public Expression slice(int offset, int length) {
            return new ListExpr(parent, ListOp.SLICE, null, offset, length);
        }
        public Expression join(String separator) {
            return new ListExpr(parent, ListOp.JOIN, separator == null ? "," : separator, 0, 0);
        }
        public Expression contains(Object item) {
            return new ListExpr(parent, ListOp.CONTAINS, item, 0, 0);
        }
        public Expression unique() { return new ListExpr(parent, ListOp.UNIQUE, null, 0, 0); }
        public Expression sort()   { return new ListExpr(parent, ListOp.SORT, null, 0, 0); }
        public Expression lengths(){ return new ListExpr(parent, ListOp.LENGTHS, null, 0, 0); }
        public Expression len()    { return lengths(); }
        public Expression reverse(){ return new ListExpr(parent, ListOp.REVERSE, null, 0, 0); }
        public Expression sum()    { return new ListExpr(parent, ListOp.SUM, null, 0, 0); }
        public Expression mean()   { return new ListExpr(parent, ListOp.MEAN, null, 0, 0); }
        public Expression min()    { return new ListExpr(parent, ListOp.MIN, null, 0, 0); }
        public Expression max()    { return new ListExpr(parent, ListOp.MAX, null, 0, 0); }

        /**
         * Evaluate an expression over list elements. For each row, materializes a
         * one-column mini-frame of list items named {@code "item"} and evaluates
         * {@code expr} row-wise, collecting results back into a list.
         * <p>Example: {@code col("nums").list().eval(col("item").multiply(2))}
         */
        public Expression eval(Expression expr) {
            return new ListEvalExpr(parent, expr);
        }
    }

    /** Polars {@code Expr.struct.*} namespace over struct/map cells. */
    public static final class StructNameSpace {
        private final Expression parent;
        StructNameSpace(Expression parent) { this.parent = parent; }

        /** Extract a named field from struct/map cell. */
        public Expression field(String name) {
            return new StructFieldExpr(parent, name);
        }

        /**
         * Unnest is a table-level op; as expression, returns the struct itself
         * (use {@link DataFrame#unnest(String)} for multi-column expand).
         */
        public Expression unnest() {
            return parent;
        }
    }

    enum ListOp {
        FIRST, LAST, GET, SLICE, JOIN, CONTAINS, UNIQUE, SORT,
        LENGTHS, REVERSE, SUM, MEAN, MIN, MAX
    }

    static final class ListExpr extends Expression {
        final Expression child;
        final ListOp op;
        final Object arg;
        final int i0, i1;
        ListExpr(Expression child, ListOp op, Object arg, int i0, int i1) {
            this.child = child; this.op = op; this.arg = arg; this.i0 = i0; this.i1 = i1;
        }

        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            List<?> list = asList(v);
            if (list == null) return null;
            return switch (op) {
                case FIRST -> list.isEmpty() ? null : list.get(0);
                case LAST  -> list.isEmpty() ? null : list.get(list.size() - 1);
                case GET -> {
                    int idx = i0 >= 0 ? i0 : list.size() + i0;
                    yield (idx >= 0 && idx < list.size()) ? list.get(idx) : null;
                }
                case SLICE -> {
                    int from = Math.max(0, i0);
                    int to = Math.min(list.size(), from + Math.max(0, i1));
                    if (from >= list.size()) yield List.of();
                    yield new ArrayList<>(list.subList(from, to));
                }
                case JOIN -> {
                    String sep = arg == null ? "," : arg.toString();
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < list.size(); i++) {
                        if (i > 0) sb.append(sep);
                        Object e = list.get(i);
                        if (e != null) sb.append(e);
                    }
                    yield sb.toString();
                }
                case CONTAINS -> {
                    for (Object e : list) if (Objects.equals(e, arg)) yield true;
                    yield false;
                }
                case UNIQUE -> {
                    LinkedHashSet<Object> set = new LinkedHashSet<>(list);
                    yield new ArrayList<>(set);
                }
                case SORT -> {
                    List<Object> copy = new ArrayList<>(list);
                    copy.sort((a, b) -> compareVals(a, b));
                    yield copy;
                }
                case LENGTHS -> list.size();
                case REVERSE -> {
                    List<Object> copy = new ArrayList<>(list);
                    Collections.reverse(copy);
                    yield copy;
                }
                case SUM, MEAN, MIN, MAX -> {
                    double sum = 0; int cnt = 0;
                    double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
                    for (Object e : list) {
                        Double d = toDouble(e);
                        if (d == null || Double.isNaN(d)) continue;
                        sum += d; cnt++;
                        if (d < min) min = d;
                        if (d > max) max = d;
                    }
                    if (cnt == 0) yield null;
                    yield switch (op) {
                        case SUM -> sum;
                        case MEAN -> sum / cnt;
                        case MIN -> min;
                        case MAX -> max;
                        default -> null;
                    };
                }
            };
        }

        @Override public String suggestedName() {
            return "list_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ListEvalExpr extends Expression {
        final Expression child;
        final Expression inner;
        ListEvalExpr(Expression child, Expression inner) {
            this.child = child; this.inner = inner;
        }

        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            List<?> list = asList(v);
            if (list == null) return null;
            // mini one-col frame
            DataFrame mini = DataFrame.create();
            mini.addColumn("item", Column.DType.STRING);
            // better: infer from first non-null
            Column.DType dt = Column.DType.STRING;
            for (Object e : list) {
                if (e != null) { dt = inferDType(e); break; }
            }
            DataFrame m = DataFrame.create();
            m.addColumn("item", dt);
            for (Object e : list) {
                int ri = m.addEmptyRow();
                m.set(ri, "item", e);
            }
            List<Object> out = new ArrayList<>(list.size());
            for (int i = 0; i < m.rowCount(); i++) {
                out.add(inner.eval(i, m));
            }
            return out;
        }

        @Override public String suggestedName() {
            return "list_eval(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class StructFieldExpr extends Expression {
        final Expression child;
        final String field;
        StructFieldExpr(Expression child, String field) {
            this.child = child; this.field = field;
        }

        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            if (v instanceof Map<?, ?> map) return map.get(field);
            if (v instanceof StructData sd) {
                return sd.getFieldValue(field);
            }
            // try ComplexCellCodec map view
            Map<String, Object> m =
                ComplexCellCodec.asStringMap(v);
            return m == null ? null : m.get(field);
        }

        @Override public String suggestedName() {
            return child.suggestedName() + "." + field;
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    // ---- advanced expr nodes ----

    static final class QcutExpr extends Expression {
        final Expression child;
        final int quantiles;
        QcutExpr(Expression child, int quantiles) {
            this.child = child; this.quantiles = Math.max(2, quantiles);
        }

        @Override public Object eval(int row, DataFrame df) {
            // evaluate whole-column once via evaluate()
            return null; // overridden
        }

        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            double[] vals = new double[n];
            boolean[] valid = new boolean[n];
            List<Double> sorted = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                Double d = toDouble(child.eval(i, df));
                if (d == null || Double.isNaN(d)) { valid[i] = false; }
                else { vals[i] = d; valid[i] = true; sorted.add(d); }
            }
            Collections.sort(sorted);
            Column out = new Column(suggestedName(), Column.DType.INT32);
            if (sorted.isEmpty()) {
                for (int i = 0; i < n; i++) out.add(null);
                return out;
            }
            double[] edges = new double[quantiles + 1];
            edges[0] = sorted.get(0);
            edges[quantiles] = sorted.get(sorted.size() - 1);
            for (int q = 1; q < quantiles; q++) {
                double pos = (sorted.size() - 1) * (q / (double) quantiles);
                int lo = (int) Math.floor(pos), hi = (int) Math.ceil(pos);
                edges[q] = lo == hi ? sorted.get(lo)
                    : sorted.get(lo) * (1 - (pos - lo)) + sorted.get(hi) * (pos - lo);
            }
            for (int i = 0; i < n; i++) {
                if (!valid[i]) { out.add(null); continue; }
                int bin = quantiles - 1;
                for (int q = 0; q < quantiles; q++) {
                    boolean last = q == quantiles - 1;
                    if (last) {
                        if (vals[i] >= edges[q] && vals[i] <= edges[q + 1]) { bin = q; break; }
                    } else if (vals[i] >= edges[q] && vals[i] < edges[q + 1]) {
                        bin = q; break;
                    }
                }
                out.add(bin);
            }
            return out;
        }

        @Override public String suggestedName() {
            return "qcut(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class FillNullStrategyExpr extends Expression {
        final Expression child;
        final String strategy;
        FillNullStrategyExpr(Expression child, String strategy) {
            this.child = child; this.strategy = strategy.toLowerCase(Locale.ROOT);
        }

        @Override public Object eval(int row, DataFrame df) {
            return null;
        }

        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            Object[] vals = new Object[n];
            for (int i = 0; i < n; i++) vals[i] = child.eval(i, df);

            switch (strategy) {
                case "forward", "ffill" -> {
                    Object last = null;
                    for (int i = 0; i < n; i++) {
                        if (vals[i] == null) vals[i] = last;
                        else last = vals[i];
                    }
                }
                case "backward", "bfill" -> {
                    Object next = null;
                    for (int i = n - 1; i >= 0; i--) {
                        if (vals[i] == null) vals[i] = next;
                        else next = vals[i];
                    }
                }
                case "zero" -> {
                    for (int i = 0; i < n; i++) if (vals[i] == null) vals[i] = 0;
                }
                case "one" -> {
                    for (int i = 0; i < n; i++) if (vals[i] == null) vals[i] = 1;
                }
                case "mean", "min", "max" -> {
                    double sum = 0, min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
                    int cnt = 0;
                    for (Object v : vals) {
                        Double d = toDouble(v);
                        if (d == null || Double.isNaN(d)) continue;
                        sum += d; cnt++;
                        if (d < min) min = d;
                        if (d > max) max = d;
                    }
                    Object fill = null;
                    if (cnt > 0) {
                        fill = switch (strategy) {
                            case "mean" -> sum / cnt;
                            case "min" -> min;
                            case "max" -> max;
                            default -> null;
                        };
                    }
                    for (int i = 0; i < n; i++) if (vals[i] == null) vals[i] = fill;
                }
                default -> { /* leave nulls */ }
            }
            Column out = new Column(suggestedName(), Column.DType.FLOAT64);
            // refine dtype from first non-null
            Column.DType dt = Column.DType.FLOAT64;
            for (Object v : vals) {
                if (v != null) { dt = inferDType(v); break; }
            }
            out = new Column(suggestedName(), dt);
            for (Object v : vals) out.add(v);
            return out;
        }

        @Override public String suggestedName() {
            return "fill_null(" + child.suggestedName() + "," + strategy + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class HashExpr extends Expression {
        final Expression child;
        final long seed;
        HashExpr(Expression child, long seed) { this.child = child; this.seed = seed; }

        @Override public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            long h = seed;
            if (v != null) {
                h ^= v.hashCode() * 0x9E3779B97F4A7C15L;
                h = Long.rotateLeft(h, 13);
            }
            return h;
        }

        @Override public String suggestedName() {
            return "hash(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class IsDuplicatedExpr extends Expression {
        final Expression child;
        IsDuplicatedExpr(Expression child) { this.child = child; }

        @Override public Object eval(int row, DataFrame df) { return null; }

        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            Map<Object, Integer> first = new HashMap<>();
            boolean[] dup = new boolean[n];
            for (int i = 0; i < n; i++) {
                Object v = child.eval(i, df);
                Integer prev = first.putIfAbsent(v, i);
                if (prev != null) {
                    dup[i] = true;
                    dup[prev] = true; // all occurrences marked
                }
            }
            Column out = new Column(suggestedName(), Column.DType.BOOLEAN);
            for (boolean b : dup) out.add(b);
            return out;
        }

        @Override public String suggestedName() {
            return "is_duplicated(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class IsFirstDistinctExpr extends Expression {
        final Expression child;
        final boolean first;
        IsFirstDistinctExpr(Expression child, boolean first) {
            this.child = child; this.first = first;
        }

        @Override public Object eval(int row, DataFrame df) { return null; }

        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            boolean[] mark = new boolean[n];
            if (first) {
                Set<Object> seen = new HashSet<>();
                for (int i = 0; i < n; i++) {
                    Object v = child.eval(i, df);
                    mark[i] = seen.add(v);
                }
            } else {
                Set<Object> seen = new HashSet<>();
                for (int i = n - 1; i >= 0; i--) {
                    Object v = child.eval(i, df);
                    mark[i] = seen.add(v);
                }
            }
            Column out = new Column(suggestedName(), Column.DType.BOOLEAN);
            for (boolean b : mark) out.add(b);
            return out;
        }

        @Override public String suggestedName() {
            return (first ? "is_first_distinct(" : "is_last_distinct(") + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ShrinkDtypeExpr extends Expression {
        final Expression child;
        ShrinkDtypeExpr(Expression child) { this.child = child; }

        @Override public Object eval(int row, DataFrame df) {
            return child.eval(row, df);
        }

        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            Object[] vals = new Object[n];
            boolean allInt = true;
            long min = Long.MAX_VALUE, max = Long.MIN_VALUE;
            for (int i = 0; i < n; i++) {
                Object v = child.eval(i, df);
                vals[i] = v;
                if (v == null) continue;
                if (!(v instanceof Number)) { allInt = false; }
                else {
                    double d = ((Number) v).doubleValue();
                    if (d != Math.rint(d) || d < Long.MIN_VALUE || d > Long.MAX_VALUE) allInt = false;
                    else {
                        long lv = ((Number) v).longValue();
                        if (lv < min) min = lv;
                        if (lv > max) max = lv;
                    }
                }
            }
            Column.DType dt;
            if (allInt) {
                if (min >= Integer.MIN_VALUE && max <= Integer.MAX_VALUE) dt = Column.DType.INT32;
                else dt = Column.DType.INT64;
            } else {
                dt = Column.DType.FLOAT64;
            }
            Column out = new Column(suggestedName(), dt);
            for (Object v : vals) {
                if (v == null) { out.add(null); continue; }
                if (dt == Column.DType.INT32) out.add(((Number) v).intValue());
                else if (dt == Column.DType.INT64) out.add(((Number) v).longValue());
                else out.add(v instanceof Number ? ((Number) v).doubleValue() : v);
            }
            return out;
        }

        @Override public String suggestedName() {
            return "shrink(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class EwmMeanExpr extends Expression {
        final Expression child;
        final double alpha;
        EwmMeanExpr(Expression child, double alpha) {
            this.child = child;
            this.alpha = alpha <= 0 || alpha > 1 ? 0.5 : alpha;
        }

        @Override public Object eval(int row, DataFrame df) { return null; }

        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            Column out = new Column(suggestedName(), Column.DType.FLOAT64);
            Double prev = null;
            for (int i = 0; i < n; i++) {
                Double d = toDouble(child.eval(i, df));
                if (d == null || Double.isNaN(d)) {
                    out.add(prev);
                } else if (prev == null) {
                    prev = d;
                    out.add(d);
                } else {
                    prev = alpha * d + (1 - alpha) * prev;
                    out.add(prev);
                }
            }
            return out;
        }

        @Override public String suggestedName() {
            return "ewm_mean(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class MapElementsExpr extends Expression {
        final Expression child;
        final java.util.function.Function<Object, Object> fn;
        MapElementsExpr(Expression child, java.util.function.Function<Object, Object> fn) {
            this.child = child; this.fn = fn;
        }

        @Override public Object eval(int row, DataFrame df) {
            return fn.apply(child.eval(row, df));
        }

        @Override public String suggestedName() {
            return "map(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class CumCountExpr extends Expression {
        final Expression child;
        final boolean reverse;
        CumCountExpr(Expression child, boolean reverse) {
            this.child = child; this.reverse = reverse;
        }

        @Override public Object eval(int row, DataFrame df) { return null; }

        @Override public Column evaluate(DataFrame df) {
            int n = df.rowCount();
            Column out = new Column(suggestedName(), Column.DType.INT64);
            long[] vals = new long[n];
            if (!reverse) {
                long c = 0;
                for (int i = 0; i < n; i++) {
                    if (child.eval(i, df) != null) c++;
                    vals[i] = c;
                }
            } else {
                long c = 0;
                for (int i = n - 1; i >= 0; i--) {
                    if (child.eval(i, df) != null) c++;
                    vals[i] = c;
                }
            }
            for (long v : vals) out.add(v);
            return out;
        }

        @Override public String suggestedName() {
            return "cum_count(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class RoundSigFigsExpr extends Expression {
        final Expression child;
        final int n;
        RoundSigFigsExpr(Expression child, int n) {
            this.child = child; this.n = Math.max(1, n);
        }

        @Override public Object eval(int row, DataFrame df) {
            Double d = toDouble(child.eval(row, df));
            if (d == null || Double.isNaN(d) || d == 0.0) return d;
            double order = Math.floor(Math.log10(Math.abs(d)));
            double factor = Math.pow(10, n - 1 - order);
            return Math.round(d * factor) / factor;
        }

        @Override public String suggestedName() {
            return "round_sig(" + child.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    enum HorizontalOp { MAX, MIN, SUM }

    static final class HorizontalExpr extends Expression {
        final HorizontalOp op;
        final Expression[] exprs;
        HorizontalExpr(HorizontalOp op, Expression[] exprs) {
            this.op = op;
            this.exprs = exprs == null ? new Expression[0] : exprs;
        }

        @Override public Object eval(int row, DataFrame df) {
            if (exprs.length == 0) return null;
            if (op == HorizontalOp.SUM) {
                double s = 0; boolean any = false;
                for (Expression e : exprs) {
                    Double d = toDouble(e.eval(row, df));
                    if (d != null && !Double.isNaN(d)) { s += d; any = true; }
                }
                return any ? s : null;
            }
            Double best = null;
            for (Expression e : exprs) {
                Double d = toDouble(e.eval(row, df));
                if (d == null || Double.isNaN(d)) continue;
                if (best == null) best = d;
                else if (op == HorizontalOp.MAX && d > best) best = d;
                else if (op == HorizontalOp.MIN && d < best) best = d;
            }
            return best;
        }

        @Override public String suggestedName() {
            return op.name().toLowerCase() + "_horizontal";
        }

        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>();
            for (Expression e : exprs) s.addAll(e.referencedColumns());
            return s;
        }
    }

    /** Coerce cell value to a List view (supports List, arrays, ListViewData). */
    @SuppressWarnings("unchecked")
    static List<?> asList(Object v) {
        if (v == null) return null;
        if (v instanceof List<?> l) return l;
        if (v instanceof ListViewData lvd) {
            return lvd.getViewElements();
        }
        if (v instanceof int[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (int x : a) out.add(x);
            return out;
        }
        if (v instanceof long[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (long x : a) out.add(x);
            return out;
        }
        if (v instanceof float[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (float x : a) out.add(x);
            return out;
        }
        if (v instanceof double[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (double x : a) out.add(x);
            return out;
        }
        if (v instanceof Object[] a) return Arrays.asList(a);
        return null;
    }
}

