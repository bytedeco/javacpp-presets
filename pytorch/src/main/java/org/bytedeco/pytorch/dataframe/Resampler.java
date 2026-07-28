package org.bytedeco.pytorch.dataframe;

import java.time.*;
import java.util.*;

/**
 * Pandas-style time-series resampling over a sorted key column.
 *
 * <pre>
 *   DataFrame out = df.resample("t", "5s").mean("x");
 *   DataFrame filled = df.resample("t", "1h").asfreq(0.0);
 * </pre>
 *
 * <p>Rule grammar (subset): {@code Ns/Nm/Nh/Nd} for seconds/minutes/hours/days,
 * or plain integer milliseconds when rule is numeric string.
 */
public final class Resampler {
    private final DataFrame source;
    private final String on;
    private final long ruleMillis;
    private final String origin; // "start" | "epoch"
    private final long offsetMillis;

    Resampler(DataFrame source, String on, long ruleMillis, String origin, long offsetMillis) {
        this.source = source;
        this.on = on;
        this.ruleMillis = ruleMillis;
        this.origin = origin == null ? "start" : origin;
        this.offsetMillis = offsetMillis;
    }

    public static Resampler of(DataFrame df, String on, String rule) {
        return of(df, on, rule, "start", 0L);
    }

    public static Resampler of(DataFrame df, String on, String rule, String origin, long offsetMillis) {
        return new Resampler(df, on, parseRule(rule), origin, offsetMillis);
    }

    /** Bin start timestamps + count of rows per bin. */
    public DataFrame count() throws Exception {
        return aggregate(null, "count");
    }

    public DataFrame sum(String col) throws Exception { return aggregate(col, "sum"); }
    public DataFrame mean(String col) throws Exception { return aggregate(col, "mean"); }
    public DataFrame min(String col) throws Exception { return aggregate(col, "min"); }
    public DataFrame max(String col) throws Exception { return aggregate(col, "max"); }
    public DataFrame first(String col) throws Exception { return aggregate(col, "first"); }
    public DataFrame last(String col) throws Exception { return aggregate(col, "last"); }

    /**
     * Reindex to every bin in [min,max], filling empty bins with {@code fillValue}.
     * Pandas {@code Resampler.asfreq}.
     */
    public DataFrame asfreq(Object fillValue) throws Exception {
        Binning b = bin();
        DataFrame result = DataFrame.create();
        result.addColumn(on, Column.DType.INT64);
        // carry all non-key columns
        for (Column c : source.columns()) {
            if (!c.name().equals(on)) result.addColumn(c.name(), c.dtype());
        }
        if (b.bins.isEmpty()) return result;

        long start = b.bins.firstKey();
        long end = b.bins.lastKey();
        // also extend to cover last point's bin
        for (long t = start; t <= end; t += ruleMillis) {
            int ri = result.addEmptyRow();
            result.set(ri, on, t);
            List<Integer> idxs = b.bins.get(t);
            if (idxs == null || idxs.isEmpty()) {
                for (Column c : source.columns()) {
                    if (!c.name().equals(on)) result.set(ri, c.name(), fillValue);
                }
            } else {
                // take first row of bin (asfreq semantics: place existing)
                int src = idxs.get(0);
                for (Column c : source.columns()) {
                    if (!c.name().equals(on)) result.set(ri, c.name(), source.get(src, c.name()));
                }
            }
        }
        return result;
    }

    /**
     * asfreq then interpolate numeric columns (linear).
     */
    public DataFrame interpolate(String method) throws Exception {
        DataFrame grid = asfreq(null);
        String m = method == null ? "linear" : method;
        for (Column c : grid.columns()) {
            if (c.name().equals(on)) continue;
            if (!isNumeric(c.dtype())) continue;
            grid = AdvancedOps.interpolate(grid, c.name(), m);
        }
        return grid;
    }

    // ---- internals ----

    private DataFrame aggregate(String col, String how) throws Exception {
        Binning b = bin();
        DataFrame result = DataFrame.create();
        result.addColumn(on, Column.DType.INT64);
        String outName = col == null ? how : col + "_" + how;
        Column.DType outType = "count".equals(how) ? Column.DType.INT64 : Column.DType.FLOAT64;
        if ("first".equals(how) || "last".equals(how)) {
            outType = col == null ? Column.DType.FLOAT64 : source.column(col).dtype();
        }
        result.addColumn(outName, outType);

        for (Map.Entry<Long, List<Integer>> e : b.bins.entrySet()) {
            int ri = result.addEmptyRow();
            result.set(ri, on, e.getKey());
            List<Integer> idxs = e.getValue();
            Object val = switch (how) {
                case "count" -> (long) idxs.size();
                case "sum", "mean", "min", "max" -> {
                    double s = 0, mn = Double.POSITIVE_INFINITY, mx = Double.NEGATIVE_INFINITY;
                    int n = 0;
                    Column c = source.column(col);
                    for (int i : idxs) {
                        double d = DataValues.asDouble(c.get(i));
                        if (Double.isNaN(d)) continue;
                        s += d; n++;
                        if (d < mn) mn = d;
                        if (d > mx) mx = d;
                    }
                    if (n == 0) yield null;
                    yield switch (how) {
                        case "sum" -> s;
                        case "mean" -> s / n;
                        case "min" -> mn;
                        case "max" -> mx;
                        default -> null;
                    };
                }
                case "first" -> source.get(idxs.get(0), col);
                case "last" -> source.get(idxs.get(idxs.size() - 1), col);
                default -> null;
            };
            result.set(ri, outName, val);
        }
        return result;
    }

    private static final class Binning {
        final NavigableMap<Long, List<Integer>> bins = new TreeMap<>();
        long origin;
    }

    private Binning bin() {
        Column key = source.column(on);
        long min = Long.MAX_VALUE, max = Long.MIN_VALUE;
        long[] times = new long[source.rowCount()];
        for (int i = 0; i < source.rowCount(); i++) {
            long t = toEpochMillis(key.get(i));
            times[i] = t;
            if (t < min) min = t;
            if (t > max) max = t;
        }
        long originMs;
        if ("epoch".equalsIgnoreCase(origin)) originMs = 0L;
        else originMs = (min == Long.MAX_VALUE ? 0L : min);
        originMs += offsetMillis;

        Binning b = new Binning();
        b.origin = originMs;
        if (source.rowCount() == 0) return b;
        for (int i = 0; i < times.length; i++) {
            long t = times[i];
            long bin = originMs + (long) Math.floor((t - originMs) / (double) ruleMillis) * ruleMillis;
            b.bins.computeIfAbsent(bin, k -> new ArrayList<>()).add(i);
        }
        return b;
    }

    static long parseRule(String rule) {
        if (rule == null || rule.isBlank()) throw new IllegalArgumentException("rule required");
        String r = rule.trim().toLowerCase(Locale.ROOT);
        try {
            if (r.endsWith("ms")) return Long.parseLong(r.substring(0, r.length() - 2));
            if (r.endsWith("s")) return Long.parseLong(r.substring(0, r.length() - 1)) * 1000L;
            if (r.endsWith("m")) return Long.parseLong(r.substring(0, r.length() - 1)) * 60_000L;
            if (r.endsWith("h")) return Long.parseLong(r.substring(0, r.length() - 1)) * 3_600_000L;
            if (r.endsWith("d")) return Long.parseLong(r.substring(0, r.length() - 1)) * 86_400_000L;
            return Long.parseLong(r); // raw millis
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("unsupported resample rule: " + rule);
        }
    }

    static long toEpochMillis(Object v) {
        if (v == null) return 0L;
        if (v instanceof Number n) return n.longValue();
        if (v instanceof Instant i) return i.toEpochMilli();
        if (v instanceof LocalDateTime ldt) return ldt.toInstant(ZoneOffset.UTC).toEpochMilli();
        if (v instanceof LocalDate ld) return ld.atStartOfDay().toInstant(ZoneOffset.UTC).toEpochMilli();
        if (v instanceof ZonedDateTime zdt) return zdt.toInstant().toEpochMilli();
        try { return Long.parseLong(v.toString()); } catch (Exception e) {}
        try { return Instant.parse(v.toString()).toEpochMilli(); } catch (Exception e) {}
        throw new IllegalArgumentException("cannot parse time key: " + v);
    }

    private static boolean isNumeric(Column.DType dt) {
        return dt == Column.DType.INT32 || dt == Column.DType.INT64
            || dt == Column.DType.FLOAT32 || dt == Column.DType.FLOAT64;
    }
}
