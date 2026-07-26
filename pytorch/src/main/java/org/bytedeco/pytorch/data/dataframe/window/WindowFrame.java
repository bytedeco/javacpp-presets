package org.bytedeco.pytorch.data.dataframe.window;

/**
 * Window frame bounds for ROWS or RANGE frames.
 * Offsets are relative to the current row: negative = preceding, positive = following.
 * Special constants: UNBOUNDED_PRECEDING, UNBOUNDED_FOLLOWING, CURRENT_ROW (0).
 */
public final class WindowFrame {
    public enum Type { ROWS, RANGE }

    /** Long.MIN_VALUE / 2 to avoid overflow when adding. */
    public static final long UNBOUNDED_PRECEDING = Long.MIN_VALUE / 2;
    public static final long UNBOUNDED_FOLLOWING = Long.MAX_VALUE / 2;
    public static final long CURRENT_ROW = 0L;

    private final Type type;
    private final long start; // inclusive offset from current
    private final long end;   // inclusive offset from current

    private WindowFrame(Type type, long start, long end) {
        this.type = type;
        this.start = start;
        this.end = end;
    }

    public static WindowFrame rows(long start, long end) {
        return new WindowFrame(Type.ROWS, start, end);
    }

    public static WindowFrame range(long start, long end) {
        return new WindowFrame(Type.RANGE, start, end);
    }

    /** Default Spark-like frame when orderBy is present: unbounded preceding → current row. */
    public static WindowFrame defaultOrdered() {
        return rows(UNBOUNDED_PRECEDING, CURRENT_ROW);
    }

    /** Default when no orderBy: whole partition. */
    public static WindowFrame wholePartition() {
        return rows(UNBOUNDED_PRECEDING, UNBOUNDED_FOLLOWING);
    }

    public Type type() { return type; }
    public long start() { return start; }
    public long end() { return end; }

    public boolean isUnboundedPreceding() { return start <= UNBOUNDED_PRECEDING / 2; }
    public boolean isUnboundedFollowing() { return end >= UNBOUNDED_FOLLOWING / 2; }

    @Override
    public String toString() {
        return type + " BETWEEN " + bound(start) + " AND " + bound(end);
    }

    private static String bound(long v) {
        if (v <= UNBOUNDED_PRECEDING / 2) return "UNBOUNDED PRECEDING";
        if (v >= UNBOUNDED_FOLLOWING / 2) return "UNBOUNDED FOLLOWING";
        if (v == 0) return "CURRENT ROW";
        if (v < 0) return (-v) + " PRECEDING";
        return v + " FOLLOWING";
    }
}
