package org.bytedeco.pytorch.dataframe.temporal;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Pandas-style {@code DateTimeIndex}: ordered sequence of instants with a zone.
 *
 * <p>Supports {@code loc}/{@code iloc}, reindex, asof, and merge helpers used by
 * temporal joins and resampling.
 */
public final class DateTimeIndex {

    private final List<Instant> instants;
    private final ZoneId zone;
    private final boolean sorted;

    public DateTimeIndex(List<Instant> instants, ZoneId zone) {
        Objects.requireNonNull(instants, "instants");
        this.zone = zone == null ? ZoneOffset.UTC : zone;
        this.instants = List.copyOf(instants);
        this.sorted = isSorted(this.instants);
    }

    public static DateTimeIndex of(List<Instant> instants) {
        return new DateTimeIndex(instants, ZoneOffset.UTC);
    }

    public static DateTimeIndex of(List<Instant> instants, ZoneId zone) {
        return new DateTimeIndex(instants, zone);
    }

    public static DateTimeIndex fromEpochMillis(long[] millis, ZoneId zone) {
        List<Instant> list = new ArrayList<>(millis.length);
        for (long m : millis) list.add(Instant.ofEpochMilli(m));
        return new DateTimeIndex(list, zone);
    }

    public static DateTimeIndex range(Instant start, Instant end, long stepMillis) {
        if (stepMillis <= 0) throw new IllegalArgumentException("stepMillis must be > 0");
        List<Instant> list = new ArrayList<>();
        for (long t = start.toEpochMilli(); t <= end.toEpochMilli(); t += stepMillis) {
            list.add(Instant.ofEpochMilli(t));
        }
        return new DateTimeIndex(list, ZoneOffset.UTC);
    }

    public int size() {
        return instants.size();
    }

    public boolean isEmpty() {
        return instants.isEmpty();
    }

    public ZoneId zone() {
        return zone;
    }

    public boolean isSorted() {
        return sorted;
    }

    public List<Instant> instants() {
        return instants;
    }

    public Instant get(int i) {
        return instants.get(i);
    }

    /** iloc: positional access. */
    public Instant iloc(int i) {
        return get(i);
    }

    /** loc by exact instant; empty if not present. */
    public Optional<Integer> loc(Instant instant) {
        for (int i = 0; i < instants.size(); i++) {
            if (instants.get(i).equals(instant)) return Optional.of(i);
        }
        return Optional.empty();
    }

    /** Epoch millis array (copy). */
    public long[] toEpochMillis() {
        long[] out = new long[instants.size()];
        for (int i = 0; i < instants.size(); i++) out[i] = instants.get(i).toEpochMilli();
        return out;
    }

    public List<ZonedDateTime> toZonedDateTime() {
        List<ZonedDateTime> out = new ArrayList<>(instants.size());
        for (Instant in : instants) out.add(in.atZone(zone));
        return out;
    }

    public List<LocalDate> toLocalDate() {
        List<LocalDate> out = new ArrayList<>(instants.size());
        for (Instant in : instants) out.add(in.atZone(zone).toLocalDate());
        return out;
    }

    public DateTimeIndex withZone(ZoneId newZone) {
        return new DateTimeIndex(instants, newZone);
    }

    /** Return a sorted copy (stable by instant). */
    public DateTimeIndex sorted() {
        if (sorted) return this;
        List<Instant> copy = new ArrayList<>(instants);
        copy.sort(Comparator.naturalOrder());
        return new DateTimeIndex(copy, zone);
    }

    /**
     * asof: largest index {@code i} with {@code instants[i] <= target}.
     * Returns -1 if all instants are after target. Requires sorted index for O(log n);
     * falls back to linear scan if unsorted.
     */
    public int asof(Instant target) {
        Objects.requireNonNull(target, "target");
        if (instants.isEmpty()) return -1;
        if (sorted) {
            int lo = 0, hi = instants.size() - 1, ans = -1;
            while (lo <= hi) {
                int mid = (lo + hi) >>> 1;
                Instant m = instants.get(mid);
                if (!m.isAfter(target)) {
                    ans = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            return ans;
        }
        int ans = -1;
        for (int i = 0; i < instants.size(); i++) {
            if (!instants.get(i).isAfter(target)) ans = i;
        }
        return ans;
    }

    /**
     * Reindex onto {@code other} labels: for each label in other, pick asof match
     * from this index (or -1 if none). Returns parallel int[] of source positions.
     */
    public int[] reindexAsof(DateTimeIndex other) {
        Objects.requireNonNull(other, "other");
        DateTimeIndex src = this.sorted ? this : this.sorted();
        int[] map = new int[other.size()];
        for (int i = 0; i < other.size(); i++) {
            map[i] = src.asof(other.get(i));
        }
        return map;
    }

    /** Slice [startInclusive, endExclusive) by position. */
    public DateTimeIndex slice(int start, int end) {
        return new DateTimeIndex(instants.subList(start, end), zone);
    }

    /** Filter instants in [start, end] inclusive. */
    public DateTimeIndex between(Instant start, Instant end) {
        List<Instant> out = new ArrayList<>();
        for (Instant in : instants) {
            if (!in.isBefore(start) && !in.isAfter(end)) out.add(in);
        }
        return new DateTimeIndex(out, zone);
    }

    private static boolean isSorted(List<Instant> list) {
        for (int i = 1; i < list.size(); i++) {
            if (list.get(i - 1).isAfter(list.get(i))) return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return "DateTimeIndex(size=" + instants.size() + ", zone=" + zone.getId()
                + ", sorted=" + sorted + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DateTimeIndex that)) return false;
        return instants.equals(that.instants) && zone.equals(that.zone);
    }

    @Override
    public int hashCode() {
        return Objects.hash(instants, zone);
    }
}
