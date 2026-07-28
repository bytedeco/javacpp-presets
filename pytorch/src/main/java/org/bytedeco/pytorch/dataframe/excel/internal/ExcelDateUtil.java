package org.bytedeco.pytorch.dataframe.excel.internal;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.Date;

/**
 * Excel serial-date helpers (1900 date system, including the legacy leap-day bug
 * so values match common spreadsheet tools for civil dates after 1900-02-28).
 */
public final class ExcelDateUtil {
    /** Excel epoch: 1899-12-30 in the 1900 system (accounts for the fake 1900-02-29). */
    private static final LocalDate EXCEL_EPOCH = LocalDate.of(1899, 12, 30);
    private static final double SECONDS_PER_DAY = 86400.0;

    private ExcelDateUtil() {}

    public static boolean isDateFormat(int numFmtId, String formatCode) {
        if (isBuiltInDateFmt(numFmtId)) return true;
        if (formatCode == null || formatCode.isEmpty()) return false;
        String f = formatCode.toLowerCase(java.util.Locale.ROOT);
        // strip bracketed sections and quoted literals roughly
        StringBuilder sb = new StringBuilder();
        boolean inQuote = false;
        for (int i = 0; i < f.length(); i++) {
            char c = f.charAt(i);
            if (c == '"') { inQuote = !inQuote; continue; }
            if (inQuote) continue;
            if (c == '[') {
                int end = f.indexOf(']', i);
                if (end >= 0) { i = end; continue; }
            }
            sb.append(c);
        }
        String s = sb.toString();
        // date-ish if has y/m/d or h:m without pure number formats
        boolean hasDate = s.indexOf('y') >= 0 || s.indexOf('d') >= 0
            || (s.indexOf('m') >= 0 && (s.indexOf('h') >= 0 || s.indexOf('s') >= 0 || s.indexOf('y') >= 0 || s.indexOf('d') >= 0));
        boolean hasTime = s.indexOf('h') >= 0 || s.indexOf('s') >= 0 || s.contains("am/pm") || s.contains("a/p");
        return hasDate || hasTime;
    }

    private static boolean isBuiltInDateFmt(int id) {
        // ECMA-376 built-in date/time format ids
        switch (id) {
            case 14: case 15: case 16: case 17:
            case 18: case 19: case 20: case 21: case 22:
            case 27: case 28: case 29: case 30: case 31:
            case 32: case 33: case 34: case 35: case 36:
            case 45: case 46: case 47:
            case 50: case 51: case 52: case 53: case 54:
            case 55: case 56: case 57: case 58:
                return true;
            default:
                return false;
        }
    }

    public static LocalDateTime fromSerial(double serial) {
        long days = (long) Math.floor(serial);
        double frac = serial - days;
        if (frac < 0) { // defensive
            days -= 1;
            frac += 1.0;
        }
        LocalDate date = EXCEL_EPOCH.plusDays(days);
        long nanos = Math.round(frac * SECONDS_PER_DAY * 1_000_000_000L);
        if (nanos >= 86_400_000_000_000L) {
            date = date.plusDays(1);
            nanos = 0;
        }
        LocalTime time = LocalTime.ofNanoOfDay(Math.max(0, Math.min(nanos, 86_399_999_999_999L)));
        return LocalDateTime.of(date, time);
    }

    public static double toSerial(LocalDate date) {
        return (double) java.time.temporal.ChronoUnit.DAYS.between(EXCEL_EPOCH, date);
    }

    public static double toSerial(LocalDateTime ldt) {
        double days = toSerial(ldt.toLocalDate());
        LocalTime t = ldt.toLocalTime();
        return days + t.toNanoOfDay() / (SECONDS_PER_DAY * 1_000_000_000.0);
    }

    public static double toSerial(Instant instant, ZoneId zone) {
        return toSerial(LocalDateTime.ofInstant(instant, zone));
    }

    public static double toSerial(Date date) {
        return toSerial(date.toInstant(), ZoneId.systemDefault());
    }
}
