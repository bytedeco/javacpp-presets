package org.bytedeco.pytorch.dataframe.temporal;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Business / trading calendar: holidays + weekend rules.
 *
 * <pre>
 *   BusinessCalendar cal = BusinessCalendar.weekendsOnly();
 *   BusinessCalendar nyse = BusinessCalendar.ofHolidays(nyseHolidays).weekend(DayOfWeek.SATURDAY, DayOfWeek.SUNDAY);
 *   boolean ok = cal.isBusinessDay(LocalDate.of(2024, 1, 2));
 * </pre>
 */
public interface BusinessCalendar {

    boolean isBusinessDay(LocalDate date);

    boolean isHoliday(LocalDate date);

    LocalDate nextBusinessDay(LocalDate date);

    LocalDate previousBusinessDay(LocalDate date);

    long businessDaysBetween(LocalDate start, LocalDate end);

    /** Weekend-only calendar (Sat/Sun off, no fixed holidays). */
    static BusinessCalendar weekendsOnly() {
        return new DefaultBusinessCalendar(Set.of(), Set.of(DayOfWeek.SATURDAY, DayOfWeek.SUNDAY));
    }

    /** Empty holidays + custom weekend days. */
    static BusinessCalendar ofWeekends(DayOfWeek... weekends) {
        Set<DayOfWeek> w = new HashSet<>();
        if (weekends != null) {
            Collections.addAll(w, weekends);
        }
        return new DefaultBusinessCalendar(Set.of(), w);
    }

    /** Holiday set with default Sat/Sun weekend. */
    static BusinessCalendar ofHolidays(Set<LocalDate> holidays) {
        return new DefaultBusinessCalendar(
                holidays == null ? Set.of() : new TreeSet<>(holidays),
                Set.of(DayOfWeek.SATURDAY, DayOfWeek.SUNDAY));
    }

    /** Full builder-style factory. */
    static BusinessCalendar of(Set<LocalDate> holidays, Set<DayOfWeek> weekend) {
        return new DefaultBusinessCalendar(
                holidays == null ? Set.of() : new TreeSet<>(holidays),
                weekend == null ? Set.of() : Set.copyOf(weekend));
    }

    /**
     * US Federal Reserve / bank holiday set for a given year (New Year, MLK, Presidents,
     * Memorial, Independence, Labor, Columbus, Veterans, Thanksgiving, Christmas).
     * Observed rules: if holiday falls on Sat → Fri; on Sun → Mon (standard US observance).
     */
    static BusinessCalendar usFederal(int year) {
        Set<LocalDate> h = new TreeSet<>();
        h.add(observe(LocalDate.of(year, 1, 1)));          // New Year
        h.add(nthWeekday(year, 1, DayOfWeek.MONDAY, 3));   // MLK 3rd Mon Jan
        h.add(nthWeekday(year, 2, DayOfWeek.MONDAY, 3));   // Presidents 3rd Mon Feb
        h.add(lastWeekday(year, 5, DayOfWeek.MONDAY));     // Memorial last Mon May
        h.add(observe(LocalDate.of(year, 7, 4)));          // Independence
        h.add(nthWeekday(year, 9, DayOfWeek.MONDAY, 1));   // Labor 1st Mon Sep
        h.add(nthWeekday(year, 10, DayOfWeek.MONDAY, 2));  // Columbus 2nd Mon Oct
        h.add(observe(LocalDate.of(year, 11, 11)));        // Veterans
        h.add(nthWeekday(year, 11, DayOfWeek.THURSDAY, 4)); // Thanksgiving
        h.add(observe(LocalDate.of(year, 12, 25)));        // Christmas
        return ofHolidays(h);
    }

    /** Observed holiday: Sat → previous Fri; Sun → next Mon. */
    static LocalDate observe(LocalDate d) {
        DayOfWeek dow = d.getDayOfWeek();
        if (dow == DayOfWeek.SATURDAY) return d.minusDays(1);
        if (dow == DayOfWeek.SUNDAY) return d.plusDays(1);
        return d;
    }

    static LocalDate nthWeekday(int year, int month, DayOfWeek dow, int n) {
        LocalDate d = LocalDate.of(year, month, 1);
        int count = 0;
        while (d.getMonthValue() == month) {
            if (d.getDayOfWeek() == dow) {
                count++;
                if (count == n) return d;
            }
            d = d.plusDays(1);
        }
        throw new IllegalArgumentException("nth weekday not found: " + year + "-" + month + " " + dow + " #" + n);
    }

    static LocalDate lastWeekday(int year, int month, DayOfWeek dow) {
        LocalDate d = LocalDate.of(year, month, 1).plusMonths(1).minusDays(1);
        while (d.getDayOfWeek() != dow) d = d.minusDays(1);
        return d;
    }

    /** Default implementation. */
    final class DefaultBusinessCalendar implements BusinessCalendar {
        private final Set<LocalDate> holidays;
        private final Set<DayOfWeek> weekend;

        DefaultBusinessCalendar(Set<LocalDate> holidays, Set<DayOfWeek> weekend) {
            this.holidays = holidays instanceof TreeSet ? holidays : new TreeSet<>(holidays);
            this.weekend = Set.copyOf(weekend);
        }

        @Override
        public boolean isHoliday(LocalDate date) {
            Objects.requireNonNull(date, "date");
            return holidays.contains(date);
        }

        @Override
        public boolean isBusinessDay(LocalDate date) {
            Objects.requireNonNull(date, "date");
            if (weekend.contains(date.getDayOfWeek())) return false;
            return !holidays.contains(date);
        }

        @Override
        public LocalDate nextBusinessDay(LocalDate date) {
            Objects.requireNonNull(date, "date");
            LocalDate d = date.plusDays(1);
            while (!isBusinessDay(d)) d = d.plusDays(1);
            return d;
        }

        @Override
        public LocalDate previousBusinessDay(LocalDate date) {
            Objects.requireNonNull(date, "date");
            LocalDate d = date.minusDays(1);
            while (!isBusinessDay(d)) d = d.minusDays(1);
            return d;
        }

        /**
         * Count of business days in half-open interval {@code [start, end)}.
         * Matches pandas {@code np.busday_count} convention when end &gt; start.
         * If end &lt; start, returns negative count of {@code [end, start)}.
         */
        @Override
        public long businessDaysBetween(LocalDate start, LocalDate end) {
            Objects.requireNonNull(start, "start");
            Objects.requireNonNull(end, "end");
            if (start.equals(end)) return 0L;
            boolean neg = end.isBefore(start);
            LocalDate a = neg ? end : start;
            LocalDate b = neg ? start : end;
            long n = 0;
            for (LocalDate d = a; d.isBefore(b); d = d.plusDays(1)) {
                if (isBusinessDay(d)) n++;
            }
            return neg ? -n : n;
        }

        public Set<LocalDate> holidays() {
            return Collections.unmodifiableSet(holidays);
        }

        public Set<DayOfWeek> weekend() {
            return weekend;
        }

        @Override
        public String toString() {
            return "BusinessCalendar{holidays=" + holidays.size() + ", weekend=" + weekend + "}";
        }
    }
}
