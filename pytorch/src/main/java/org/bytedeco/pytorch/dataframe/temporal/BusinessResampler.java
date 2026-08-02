package org.bytedeco.pytorch.dataframe.temporal;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Resampler;

/**
 * Business-day aware resampling.
 *
 * <pre>
 *   DataFrame out = df.resampleBusiness("t", "1d").mean("x");
 * </pre>
 */
public final class BusinessResampler {

    private final DataFrame source;
    private final String on;
    private final String rule;
    private final BusinessCalendar calendar;
    private final boolean business;

    public BusinessResampler(DataFrame source, String on, String rule, BusinessCalendar calendar, boolean business) {
        this.source = source;
        this.on = on;
        this.rule = rule;
        this.calendar = calendar == null ? BusinessCalendar.weekendsOnly() : calendar;
        this.business = business;
    }

    public static BusinessResampler of(DataFrame df, String on, String rule) {
        return new BusinessResampler(df, on, rule, BusinessCalendar.weekendsOnly(), true);
    }

    public static BusinessResampler of(DataFrame df, String on, String rule, BusinessCalendar calendar) {
        return new BusinessResampler(df, on, rule, calendar, true);
    }

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
     * Reindex to every business-day bin in [min,max], filling empty bins with {@code fillValue}.
     */
    public DataFrame asfreq(Object fillValue) throws Exception {
        // For business calendar we filter the bins to business days only
        Resampler base = Resampler.of(source, on, rule);
        DataFrame grid = base.asfreq(fillValue);
        if (!business) return grid;

        // Filter to business days
        DataFrame result = DataFrame.create();
        result.addColumn(on, Column.DType.INT64);
        for (Column c : source.columns()) {
            if (!c.name().equals(on)) result.addColumn(c.name(), c.dtype());
        }
        for (int i = 0; i < grid.rowCount(); i++) {
            long t = ((Number) grid.get(i, on)).longValue();
            java.time.LocalDate d = java.time.Instant.ofEpochMilli(t).atZone(java.time.ZoneOffset.UTC).toLocalDate();
            if (calendar.isBusinessDay(d)) {
                int ri = result.addEmptyRow();
                result.set(ri, on, t);
                for (Column c : source.columns()) {
                    if (!c.name().equals(on)) result.set(ri, c.name(), grid.get(i, c.name()));
                }
            }
        }
        return result;
    }

    private DataFrame aggregate(String col, String how) throws Exception {
        Resampler base = Resampler.of(source, on, rule);
        DataFrame raw = switch (how) {
            case "count" -> base.count();
            case "sum" -> base.sum(col);
            case "mean" -> base.mean(col);
            case "min" -> base.min(col);
            case "max" -> base.max(col);
            case "first" -> base.first(col);
            case "last" -> base.last(col);
            default -> throw new IllegalArgumentException("unknown agg: " + how);
        };
        if (!business) return raw;

        // Filter aggregate result to business-day bins
        DataFrame result = DataFrame.create();
        for (Column c : raw.columns()) result.addColumn(c.name(), c.dtype());
        for (int i = 0; i < raw.rowCount(); i++) {
            long t = ((Number) raw.get(i, on)).longValue();
            java.time.LocalDate d = java.time.Instant.ofEpochMilli(t).atZone(java.time.ZoneOffset.UTC).toLocalDate();
            if (calendar.isBusinessDay(d)) {
                int ri = result.addEmptyRow();
                for (Column c : raw.columns()) {
                    result.set(ri, c.name(), raw.get(i, c.name()));
                }
            }
        }
        return result;
    }
}
