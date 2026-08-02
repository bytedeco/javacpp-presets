package org.bytedeco.pytorch.dataframe.temporal;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;

/**
 * Enhanced TimeFeatureExtractor with business-day, quarter, and is_month_end.
 */
public class TemporalFeatureExtractor extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    private boolean includeHour = true;
    private boolean includeWeekday = true;
    private boolean includeDay = false;
    private boolean includeMonth = false;
    private boolean includeQuarter = false;
    private boolean includeIsMonthEnd = false;
    private boolean includeIsBusinessDay = false;
    private boolean dropOriginal = false;
    private ZoneId zone = ZoneId.systemDefault();

    private static final DateTimeFormatter[] FORMATS = new DateTimeFormatter[] {
        DateTimeFormatter.ISO_DATE_TIME,
        DateTimeFormatter.ISO_LOCAL_DATE_TIME,
        DateTimeFormatter.ISO_INSTANT,
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
        DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"),
        DateTimeFormatter.ofPattern("yyyy-MM-dd")
    };

    public TemporalFeatureExtractor(String... columns) {
        super(columns);
    }

    public TemporalFeatureExtractor includeHour(boolean v) { this.includeHour = v; return this; }
    public TemporalFeatureExtractor includeWeekday(boolean v) { this.includeWeekday = v; return this; }
    public TemporalFeatureExtractor includeDay(boolean v) { this.includeDay = v; return this; }
    public TemporalFeatureExtractor includeMonth(boolean v) { this.includeMonth = v; return this; }
    public TemporalFeatureExtractor includeQuarter(boolean v) { this.includeQuarter = v; return this; }
    public TemporalFeatureExtractor includeIsMonthEnd(boolean v) { this.includeIsMonthEnd = v; return this; }
    public TemporalFeatureExtractor includeIsBusinessDay(boolean v) { this.includeIsBusinessDay = v; return this; }
    public TemporalFeatureExtractor dropOriginal(boolean v) { this.dropOriginal = v; return this; }
    public TemporalFeatureExtractor zone(ZoneId zone) { this.zone = zone == null ? ZoneId.systemDefault() : zone; return this; }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            for (Column c : X.columns()) {
                String n = c.name().toLowerCase();
                if (n.contains("time") || n.contains("date") || n.contains("ts") || n.contains("datetime")) {
                    columns = new ArrayList<>();
                    columns.add(c.name());
                    break;
                }
            }
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        DataFrame result = X.copy();
        for (String col : columns) {
            if (!X.hasColumn(col)) continue;
            Column src = X.column(col);
            List<String> outCols = new ArrayList<>();
            if (includeHour) outCols.add(ensureCol(result, col + "_hour"));
            if (includeWeekday) outCols.add(ensureCol(result, col + "_weekday"));
            if (includeDay) outCols.add(ensureCol(result, col + "_day"));
            if (includeMonth) outCols.add(ensureCol(result, col + "_month"));
            if (includeQuarter) outCols.add(ensureCol(result, col + "_quarter"));
            if (includeIsMonthEnd) outCols.add(ensureCol(result, col + "_is_month_end"));
            if (includeIsBusinessDay) outCols.add(ensureCol(result, col + "_is_business_day"));

            for (int i = 0; i < result.rowCount(); i++) {
                ZonedDateTime zdt = parse(src.get(i));
                int p = 0;
                if (includeHour) {
                    result.set(i, outCols.get(p++), zdt == null ? null : zdt.getHour());
                }
                if (includeWeekday) {
                    result.set(i, outCols.get(p++), zdt == null ? null : zdt.getDayOfWeek().getValue() % 7);
                }
                if (includeDay) {
                    result.set(i, outCols.get(p++), zdt == null ? null : zdt.getDayOfMonth());
                }
                if (includeMonth) {
                    result.set(i, outCols.get(p++), zdt == null ? null : zdt.getMonthValue());
                }
                if (includeQuarter) {
                    result.set(i, outCols.get(p++), zdt == null ? null : (zdt.getMonthValue() - 1) / 3 + 1);
                }
                if (includeIsMonthEnd) {
                    result.set(i, outCols.get(p++), zdt == null ? null : zdt.toLocalDate().lengthOfMonth() == zdt.getDayOfMonth());
                }
                if (includeIsBusinessDay) {
                    result.set(i, outCols.get(p++), zdt == null ? null :
                        org.bytedeco.pytorch.dataframe.temporal.BusinessCalendar.weekendsOnly()
                            .isBusinessDay(zdt.toLocalDate()));
                }
            }
            if (dropOriginal && result.hasColumn(col)) result.removeColumn(col);
        }
        return result;
    }

    private String ensureCol(DataFrame df, String name) {
        String n = name;
        int k = 1;
        while (df.hasColumn(n)) n = name + "_" + (k++);
        df.addColumn(n, Column.DType.INT32);
        Column c = df.column(n);
        while (c.size() < df.rowCount()) c.add(null);
        for (int i = 0; i < df.rowCount(); i++) c.set(i, null);
        return n;
    }

    private ZonedDateTime parse(Object raw) {
        if (raw == null) return null;
        Object v = DataValues.unwrap(raw);
        if (v == null) return null;
        if (v instanceof java.util.Date date) {
            return ZonedDateTime.ofInstant(date.toInstant(), zone);
        }
        if (v instanceof Instant inst) {
            return ZonedDateTime.ofInstant(inst, zone);
        }
        if (v instanceof ZonedDateTime z) return z;
        if (v instanceof LocalDateTime ldt) return ldt.atZone(zone);
        if (v instanceof Number num) {
            long epoch = num.longValue();
            if (epoch < 100_000_000_000L) epoch *= 1000L;
            return ZonedDateTime.ofInstant(Instant.ofEpochMilli(epoch), zone);
        }
        String s = v.toString().trim();
        if (s.isEmpty()) return null;
        try {
            if (s.matches("^-?\\d+(\\.\\d+)?$")) {
                double d = Double.parseDouble(s);
                long epoch = (long) d;
                if (epoch < 100_000_000_000L) epoch *= 1000L;
                return ZonedDateTime.ofInstant(Instant.ofEpochMilli(epoch), zone);
            }
        } catch (Exception ignored) {}
        for (DateTimeFormatter fmt : FORMATS) {
            try {
                if (fmt == DateTimeFormatter.ISO_INSTANT) {
                    return ZonedDateTime.ofInstant(Instant.parse(s), zone);
                }
                try {
                    return ZonedDateTime.parse(s, fmt);
                } catch (DateTimeParseException e1) {
                    LocalDateTime ldt = LocalDateTime.parse(s, fmt);
                    return ldt.atZone(zone);
                }
            } catch (Exception ignored) {}
        }
        return null;
    }
}
