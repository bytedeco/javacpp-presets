package org.bytedeco.pytorch.dataframe.temporal;

import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.dtype.TimestampData;

import java.time.DayOfWeek;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Objects;

/**
 * Pure temporal operators used by {@code Expression.dt} and column-wise feature extractors.
 *
 * <p>All methods accept mixed cell values (Instant / LocalDateTime / LocalDate / Number epoch-ms /
 * ISO string / {@link TimestampData}) and return boxed primitives or temporal objects.
 * Null-in → null-out.
 */
public final class TemporalOps {

    private TemporalOps() {}

    public static final BusinessCalendar DEFAULT_CALENDAR = BusinessCalendar.weekendsOnly();

    // ---- coerce ----

    public static Instant toInstant(Object v, ZoneId zone) {
        if (v == null) return null;
        Object u = DataValues.unwrap(v);
        if (u == null) return null;
        ZoneId z = zone == null ? ZoneOffset.UTC : zone;
        if (u instanceof Instant in) return in;
        if (u instanceof ZonedDateTime zdt) return zdt.toInstant();
        if (u instanceof LocalDateTime ldt) return ldt.atZone(z).toInstant();
        if (u instanceof LocalDate ld) return ld.atStartOfDay(z).toInstant();
        if (u instanceof TimestampData td) return td.getInstant();
        if (u instanceof java.util.Date d) return d.toInstant();
        if (u instanceof Number n) {
            long epoch = n.longValue();
            if (Math.abs(epoch) < 100_000_000_000L) epoch *= 1000L; // seconds → ms
            return Instant.ofEpochMilli(epoch);
        }
        String s = u.toString().trim();
        if (s.isEmpty()) return null;
        try {
            if (s.matches("^-?\\d+$")) {
                long epoch = Long.parseLong(s);
                if (Math.abs(epoch) < 100_000_000_000L) epoch *= 1000L;
                return Instant.ofEpochMilli(epoch);
            }
            return Instant.parse(s);
        } catch (Exception e) {
            try {
                return LocalDateTime.parse(s).atZone(z).toInstant();
            } catch (Exception e2) {
                try {
                    return LocalDate.parse(s).atStartOfDay(z).toInstant();
                } catch (Exception e3) {
                    return null;
                }
            }
        }
    }

    public static LocalDateTime toLdt(Object v, ZoneId zone) {
        Instant in = toInstant(v, zone);
        if (in == null) return null;
        return LocalDateTime.ofInstant(in, zone == null ? ZoneOffset.UTC : zone);
    }

    public static LocalDate toLocalDate(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.toLocalDate();
    }

    // ---- component extractors (pandas/polars dt.*) ----

    public static Integer year(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.getYear();
    }

    public static Integer month(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.getMonthValue();
    }

    public static Integer day(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.getDayOfMonth();
    }

    public static Integer hour(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.getHour();
    }

    public static Integer minute(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.getMinute();
    }

    public static Integer second(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.getSecond();
    }

    /** ISO day of week 1=Mon … 7=Sun (pandas dayofweek is 0=Mon; we match ISO / polars). */
    public static Integer dayOfWeek(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        return d == null ? null : d.getDayOfWeek().getValue();
    }

    /** Quarter 1–4. */
    public static Integer quarter(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        return d == null ? null : (d.getMonthValue() - 1) / 3 + 1;
    }

    public static Integer dayOfYear(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        return d == null ? null : d.getDayOfYear();
    }

    public static Integer week(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return d.get(java.time.temporal.IsoFields.WEEK_OF_WEEK_BASED_YEAR);
    }

    public static Long epochMilli(Object v, ZoneId zone) {
        Instant in = toInstant(v, zone);
        return in == null ? null : in.toEpochMilli();
    }

    public static LocalDate toDate(Object v, ZoneId zone) {
        return toLocalDate(v, zone);
    }

    public static LocalTime toTime(Object v, ZoneId zone) {
        LocalDateTime ldt = toLdt(v, zone);
        return ldt == null ? null : ldt.toLocalTime();
    }

    // ---- calendar flags ----

    public static Boolean isWeekend(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        DayOfWeek dow = d.getDayOfWeek();
        return dow == DayOfWeek.SATURDAY || dow == DayOfWeek.SUNDAY;
    }

    public static Boolean isMonthEnd(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return d.getDayOfMonth() == d.lengthOfMonth();
    }

    public static Boolean isMonthStart(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return d.getDayOfMonth() == 1;
    }

    public static Boolean isQuarterEnd(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        int m = d.getMonthValue();
        return (m == 3 || m == 6 || m == 9 || m == 12) && d.getDayOfMonth() == d.lengthOfMonth();
    }

    public static Boolean isQuarterStart(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        int m = d.getMonthValue();
        return (m == 1 || m == 4 || m == 7 || m == 10) && d.getDayOfMonth() == 1;
    }

    public static Boolean isYearEnd(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return d.getMonthValue() == 12 && d.getDayOfMonth() == 31;
    }

    public static Boolean isYearStart(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return d.getMonthValue() == 1 && d.getDayOfMonth() == 1;
    }

    public static Boolean isLeapYear(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return d.isLeapYear();
    }

    // ---- business calendar ----

    public static Boolean isBusinessDay(Object v, ZoneId zone, BusinessCalendar cal) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return Objects.requireNonNullElse(cal, DEFAULT_CALENDAR).isBusinessDay(d);
    }

    public static Boolean isHoliday(Object v, ZoneId zone, BusinessCalendar cal) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return Objects.requireNonNullElse(cal, DEFAULT_CALENDAR).isHoliday(d);
    }

    public static LocalDate nextBusinessDay(Object v, ZoneId zone, BusinessCalendar cal) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return Objects.requireNonNullElse(cal, DEFAULT_CALENDAR).nextBusinessDay(d);
    }

    public static LocalDate previousBusinessDay(Object v, ZoneId zone, BusinessCalendar cal) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        return Objects.requireNonNullElse(cal, DEFAULT_CALENDAR).previousBusinessDay(d);
    }

    /**
     * Snap to business day: if already business day keep; else next business day
     * (pandas {@code to_busday} convention with roll='forward').
     */
    public static LocalDate toBusinessDay(Object v, ZoneId zone, BusinessCalendar cal) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        BusinessCalendar c = Objects.requireNonNullElse(cal, DEFAULT_CALENDAR);
        return c.isBusinessDay(d) ? d : c.nextBusinessDay(d);
    }

    public static Long businessDaysBetween(Object start, Object end, ZoneId zone, BusinessCalendar cal) {
        LocalDate a = toLocalDate(start, zone);
        LocalDate b = toLocalDate(end, zone);
        if (a == null || b == null) return null;
        return Objects.requireNonNullElse(cal, DEFAULT_CALENDAR).businessDaysBetween(a, b);
    }

    // ---- month/period boundaries ----

    public static LocalDate monthStart(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        return d == null ? null : d.withDayOfMonth(1);
    }

    public static LocalDate monthEnd(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        return d == null ? null : d.withDayOfMonth(d.lengthOfMonth());
    }

    public static LocalDate quarterStart(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        int q = (d.getMonthValue() - 1) / 3;
        return LocalDate.of(d.getYear(), q * 3 + 1, 1);
    }

    public static LocalDate quarterEnd(Object v, ZoneId zone) {
        LocalDate d = toLocalDate(v, zone);
        if (d == null) return null;
        int q = (d.getMonthValue() - 1) / 3;
        LocalDate start = LocalDate.of(d.getYear(), q * 3 + 1, 1).plusMonths(3).minusDays(1);
        return start;
    }
}
