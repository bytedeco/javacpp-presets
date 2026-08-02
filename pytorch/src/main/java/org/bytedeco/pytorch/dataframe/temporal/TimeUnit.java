package org.bytedeco.pytorch.dataframe.temporal;

/**
 * Temporal resolution units for timestamps, periods, and resampling.
 * Aligned with Arrow Timestamp unit + pandas frequency aliases.
 */
public enum TimeUnit {
    NANOSECOND(1L, "ns"),
    MICROSECOND(1_000L, "us"),
    MILLISECOND(1_000_000L, "ms"),
    SECOND(1_000_000_000L, "s"),
    MINUTE(60_000_000_000L, "m"),
    HOUR(3_600_000_000_000L, "h"),
    DAY(86_400_000_000_000L, "d"),
    WEEK(604_800_000_000_000L, "w"),
    MONTH(-1L, "M"),     // calendar-based
    QUARTER(-1L, "Q"),   // calendar-based
    YEAR(-1L, "Y");      // calendar-based

    private final long nanos;
    private final String alias;

    TimeUnit(long nanos, String alias) {
        this.nanos = nanos;
        this.alias = alias;
    }

    /** Fixed-duration units return nanoseconds; calendar units return -1. */
    public long toNanos() {
        return nanos;
    }

    public long toMillis() {
        if (nanos < 0) {
            throw new UnsupportedOperationException(name() + " is calendar-based, not fixed millis");
        }
        return nanos / 1_000_000L;
    }

    public boolean isCalendarBased() {
        return nanos < 0;
    }

    public String alias() {
        return alias;
    }

    /**
     * Parse pandas/polars-style frequency string: {@code "ns","us","ms","s","m","h","d","w","M","Q","Y"}
     * or multi-unit like {@code "5s"}, {@code "15m"} (returns base unit; multiplier via {@link #parseMultiplier}).
     */
    public static TimeUnit parse(String rule) {
        if (rule == null || rule.isBlank()) {
            throw new IllegalArgumentException("time unit rule required");
        }
        String r = rule.trim();
        // strip leading digits for unit detection
        int i = 0;
        while (i < r.length() && Character.isDigit(r.charAt(i))) i++;
        String unit = (i < r.length() ? r.substring(i) : r).toLowerCase();
        return switch (unit) {
            case "ns", "n", "nanosecond", "nanoseconds" -> NANOSECOND;
            case "us", "µs", "microsecond", "microseconds" -> MICROSECOND;
            case "ms", "millisecond", "milliseconds" -> MILLISECOND;
            case "s", "sec", "second", "seconds" -> SECOND;
            case "t", "min", "minute", "minutes" -> MINUTE; // pandas T/min
            case "m" -> {
                // ambiguous: pandas 'm' is minute in offset aliases sometimes; 'M' is month
                // We treat lowercase m as minute, uppercase M already lowercased → check original
                char last = rule.trim().charAt(rule.trim().length() - 1);
                yield (last == 'M') ? MONTH : MINUTE;
            }
            case "h", "hour", "hours" -> HOUR;
            case "d", "day", "days" -> DAY;
            case "w", "week", "weeks" -> WEEK;
            case "month", "months", "mo" -> MONTH;
            case "q", "quarter", "quarters" -> QUARTER;
            case "y", "a", "year", "years" -> YEAR;
            default -> throw new IllegalArgumentException("unsupported time unit: " + rule);
        };
    }

    /** Multiplier prefix of a rule string, e.g. {@code "5s"} → 5, {@code "ms"} → 1. */
    public static long parseMultiplier(String rule) {
        if (rule == null || rule.isBlank()) return 1L;
        String r = rule.trim();
        int i = 0;
        while (i < r.length() && Character.isDigit(r.charAt(i))) i++;
        if (i == 0) return 1L;
        return Long.parseLong(r.substring(0, i));
    }

    /** Convert a rule like {@code "5s"} / {@code "1h"} to fixed milliseconds; calendar rules throw. */
    public static long ruleToMillis(String rule) {
        TimeUnit u = parse(rule);
        if (u.isCalendarBased()) {
            throw new IllegalArgumentException("calendar unit not fixed-millis: " + rule);
        }
        return parseMultiplier(rule) * u.toMillis();
    }
}
