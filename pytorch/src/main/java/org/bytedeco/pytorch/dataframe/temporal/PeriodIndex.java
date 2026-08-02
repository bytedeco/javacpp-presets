package org.bytedeco.pytorch.dataframe.temporal;

import java.time.LocalDate;
import java.time.Period;
import java.time.YearMonth;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Pandas-style {@code PeriodIndex}: ordered sequence of calendar periods
 * (year / quarter / month / week / day).
 *
 * <p>Each element is represented as a half-open interval {@code [start, end)}
 * plus a {@link TimeUnit} frequency.
 */
public final class PeriodIndex {

    /** One period cell. */
    public record PeriodLabel(LocalDate start, LocalDate endExclusive, TimeUnit freq) {
        public PeriodLabel {
            Objects.requireNonNull(start, "start");
            Objects.requireNonNull(endExclusive, "endExclusive");
            Objects.requireNonNull(freq, "freq");
            if (!endExclusive.isAfter(start)) {
                throw new IllegalArgumentException("endExclusive must be after start: " + start + " / " + endExclusive);
            }
        }

        public boolean contains(LocalDate d) {
            return !d.isBefore(start) && d.isBefore(endExclusive);
        }

        public long lengthDays() {
            return java.time.temporal.ChronoUnit.DAYS.between(start, endExclusive);
        }

        @Override
        public String toString() {
            return switch (freq) {
                case YEAR -> start.getYear() + "";
                case QUARTER -> start.getYear() + "Q" + ((start.getMonthValue() - 1) / 3 + 1);
                case MONTH -> YearMonth.from(start).toString();
                case WEEK -> start + "/W";
                default -> start + "/" + endExclusive;
            };
        }
    }

    private final List<PeriodLabel> periods;
    private final TimeUnit freq;

    public PeriodIndex(List<PeriodLabel> periods, TimeUnit freq) {
        this.periods = List.copyOf(periods);
        this.freq = Objects.requireNonNull(freq, "freq");
    }

    public static PeriodIndex of(List<PeriodLabel> periods, TimeUnit freq) {
        return new PeriodIndex(periods, freq);
    }

    /** Monthly periods covering [startMonth, endMonth] inclusive. */
    public static PeriodIndex monthRange(YearMonth start, YearMonth end) {
        List<PeriodLabel> list = new ArrayList<>();
        YearMonth cur = start;
        while (!cur.isAfter(end)) {
            LocalDate s = cur.atDay(1);
            LocalDate e = cur.plusMonths(1).atDay(1);
            list.add(new PeriodLabel(s, e, TimeUnit.MONTH));
            cur = cur.plusMonths(1);
        }
        return new PeriodIndex(list, TimeUnit.MONTH);
    }

    /** Quarterly periods for years [startYear, endYear] inclusive. */
    public static PeriodIndex quarterRange(int startYear, int endYear) {
        List<PeriodLabel> list = new ArrayList<>();
        for (int y = startYear; y <= endYear; y++) {
            for (int q = 1; q <= 4; q++) {
                int m = (q - 1) * 3 + 1;
                LocalDate s = LocalDate.of(y, m, 1);
                LocalDate e = s.plusMonths(3);
                list.add(new PeriodLabel(s, e, TimeUnit.QUARTER));
            }
        }
        return new PeriodIndex(list, TimeUnit.QUARTER);
    }

    /** Yearly periods [startYear, endYear] inclusive. */
    public static PeriodIndex yearRange(int startYear, int endYear) {
        List<PeriodLabel> list = new ArrayList<>();
        for (int y = startYear; y <= endYear; y++) {
            list.add(new PeriodLabel(LocalDate.of(y, 1, 1), LocalDate.of(y + 1, 1, 1), TimeUnit.YEAR));
        }
        return new PeriodIndex(list, TimeUnit.YEAR);
    }

    /** Build from a list of LocalDate by flooring each to the given frequency. */
    public static PeriodIndex fromDates(List<LocalDate> dates, TimeUnit freq) {
        Objects.requireNonNull(dates, "dates");
        Objects.requireNonNull(freq, "freq");
        List<PeriodLabel> list = new ArrayList<>(dates.size());
        for (LocalDate d : dates) {
            list.add(floor(d, freq));
        }
        return new PeriodIndex(list, freq);
    }

    public static PeriodLabel floor(LocalDate d, TimeUnit freq) {
        Objects.requireNonNull(d, "d");
        return switch (freq) {
            case YEAR -> new PeriodLabel(LocalDate.of(d.getYear(), 1, 1),
                    LocalDate.of(d.getYear() + 1, 1, 1), TimeUnit.YEAR);
            case QUARTER -> {
                int q = (d.getMonthValue() - 1) / 3;
                LocalDate s = LocalDate.of(d.getYear(), q * 3 + 1, 1);
                yield new PeriodLabel(s, s.plusMonths(3), TimeUnit.QUARTER);
            }
            case MONTH -> {
                LocalDate s = d.withDayOfMonth(1);
                yield new PeriodLabel(s, s.plusMonths(1), TimeUnit.MONTH);
            }
            case WEEK -> {
                // ISO week: Monday start
                LocalDate s = d.minusDays((d.getDayOfWeek().getValue() + 6) % 7);
                yield new PeriodLabel(s, s.plusDays(7), TimeUnit.WEEK);
            }
            case DAY -> new PeriodLabel(d, d.plusDays(1), TimeUnit.DAY);
            default -> throw new IllegalArgumentException("PeriodIndex freq not supported: " + freq);
        };
    }

    public int size() {
        return periods.size();
    }

    public TimeUnit freq() {
        return freq;
    }

    public PeriodLabel get(int i) {
        return periods.get(i);
    }

    public List<PeriodLabel> periods() {
        return periods;
    }

    public List<LocalDate> starts() {
        List<LocalDate> out = new ArrayList<>(periods.size());
        for (PeriodLabel p : periods) out.add(p.start());
        return out;
    }

    /** Shift all periods by n frequency units. */
    public PeriodIndex shift(int n) {
        List<PeriodLabel> out = new ArrayList<>(periods.size());
        for (PeriodLabel p : periods) {
            LocalDate s = shiftDate(p.start(), freq, n);
            LocalDate e = shiftDate(p.endExclusive(), freq, n);
            out.add(new PeriodLabel(s, e, p.freq()));
        }
        return new PeriodIndex(out, freq);
    }

    private static LocalDate shiftDate(LocalDate d, TimeUnit freq, int n) {
        return switch (freq) {
            case YEAR -> d.plusYears(n);
            case QUARTER -> d.plusMonths(3L * n);
            case MONTH -> d.plusMonths(n);
            case WEEK -> d.plusWeeks(n);
            case DAY -> d.plusDays(n);
            default -> throw new IllegalArgumentException("cannot shift freq: " + freq);
        };
    }

    /** Convert period starts to a DateTimeIndex at UTC midnight. */
    public DateTimeIndex toDateTimeIndex() {
        List<java.time.Instant> instants = new ArrayList<>(periods.size());
        for (PeriodLabel p : periods) {
            instants.add(p.start().atStartOfDay().toInstant(java.time.ZoneOffset.UTC));
        }
        return DateTimeIndex.of(instants);
    }

    @Override
    public String toString() {
        return "PeriodIndex(size=" + periods.size() + ", freq=" + freq + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PeriodIndex that)) return false;
        return periods.equals(that.periods) && freq == that.freq;
    }

    @Override
    public int hashCode() {
        return Objects.hash(periods, freq);
    }
}
