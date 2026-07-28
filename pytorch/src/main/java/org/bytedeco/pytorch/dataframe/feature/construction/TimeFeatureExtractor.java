package org.bytedeco.pytorch.dataframe.feature.construction;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;

/**
 * Extract calendar features from a datetime column (hour, weekday, optional day/month).
 * Accepts epoch-ms (Number/Long), ISO-8601 strings, or {@link java.util.Date}.
 */
public class TimeFeatureExtractor extends BaseTransformer {
    private static final long serialVersionUID = 1L;

    private boolean includeHour = true;
    private boolean includeWeekday = true;
    private boolean includeDay = false;
    private boolean includeMonth = false;
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

    public TimeFeatureExtractor(String... columns) {
        super(columns);
    }

    public TimeFeatureExtractor includeHour(boolean v) { this.includeHour = v; return this; }
    public TimeFeatureExtractor includeWeekday(boolean v) { this.includeWeekday = v; return this; }
    public TimeFeatureExtractor includeDay(boolean v) { this.includeDay = v; return this; }
    public TimeFeatureExtractor includeMonth(boolean v) { this.includeMonth = v; return this; }
    public TimeFeatureExtractor dropOriginal(boolean v) { this.dropOriginal = v; return this; }
    public TimeFeatureExtractor zone(ZoneId zone) { this.zone = zone == null ? ZoneId.systemDefault() : zone; return this; }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            // pick first column that looks temporal by name
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
            // heuristic: seconds vs millis
            if (epoch < 100_000_000_000L) epoch *= 1000L;
            return ZonedDateTime.ofInstant(Instant.ofEpochMilli(epoch), zone);
        }
        String s = v.toString().trim();
        if (s.isEmpty()) return null;
        // numeric string
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
