package org.bytedeco.pytorch.data.dataframe.window;

import org.bytedeco.pytorch.data.dataframe.Expression;

/**
 * Describes a window function to evaluate over a {@link WindowSpec}.
 */
public final class WindowFunction {
    public enum Kind {
        ROW_NUMBER, RANK, DENSE_RANK, PERCENT_RANK, NTILE, CUME_DIST,
        LAG, LEAD,
        // aggregates re-used from Expression.AggOp semantics
        SUM, MEAN, MIN, MAX, COUNT, STD, VAR, FIRST, LAST, MEDIAN, PRODUCT
    }

    private final Kind kind;
    private final Expression value;   // value expression for lag/lead/agg; null for ranking
    private final int offset;         // lag/lead offset or ntile buckets
    private final Object defaultValue; // lag/lead default

    private WindowFunction(Kind kind, Expression value, int offset, Object defaultValue) {
        this.kind = kind;
        this.value = value;
        this.offset = offset;
        this.defaultValue = defaultValue;
    }

    public static WindowFunction rowNumber() { return new WindowFunction(Kind.ROW_NUMBER, null, 0, null); }
    public static WindowFunction rank() { return new WindowFunction(Kind.RANK, null, 0, null); }
    public static WindowFunction denseRank() { return new WindowFunction(Kind.DENSE_RANK, null, 0, null); }
    public static WindowFunction percentRank() { return new WindowFunction(Kind.PERCENT_RANK, null, 0, null); }
    public static WindowFunction ntile(int n) { return new WindowFunction(Kind.NTILE, null, n, null); }
    public static WindowFunction cumeDist() { return new WindowFunction(Kind.CUME_DIST, null, 0, null); }

    public static WindowFunction lag(Expression e, int n) { return lag(e, n, null); }
    public static WindowFunction lag(Expression e, int n, Object def) {
        return new WindowFunction(Kind.LAG, e, n, def);
    }
    public static WindowFunction lead(Expression e, int n) { return lead(e, n, null); }
    public static WindowFunction lead(Expression e, int n, Object def) {
        return new WindowFunction(Kind.LEAD, e, n, def);
    }

    public static WindowFunction agg(Kind kind, Expression e) {
        return new WindowFunction(kind, e, 0, null);
    }

    public Kind kind() { return kind; }
    public Expression value() { return value; }
    public int offset() { return offset; }
    public Object defaultValue() { return defaultValue; }

    public String suggestedName() {
        switch (kind) {
            case ROW_NUMBER: return "row_number";
            case RANK: return "rank";
            case DENSE_RANK: return "dense_rank";
            case PERCENT_RANK: return "percent_rank";
            case NTILE: return "ntile";
            case CUME_DIST: return "cume_dist";
            case LAG: return "lag(" + (value == null ? "?" : value.suggestedName()) + ")";
            case LEAD: return "lead(" + (value == null ? "?" : value.suggestedName()) + ")";
            default:
                String vn = value == null ? "?" : value.suggestedName();
                return kind.name().toLowerCase() + "(" + vn + ")";
        }
    }
}
