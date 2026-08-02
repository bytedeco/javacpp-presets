package org.bytedeco.pytorch.dataframe.temporal;

import org.bytedeco.pytorch.dataframe.dtype.AbstractDataValue;

import java.time.Duration;
import java.util.Objects;

/**
 * Timedelta / duration cell value with fixed nanosecond precision storage.
 * Supports arithmetic (+, -, *, /) against other durations and scalars.
 *
 * <p>Aligned with pandas Timedelta and Arrow DurationType (nanosecond unit default).
 */
public final class DurationData extends AbstractDataValue {

    private static final long serialVersionUID = 1L;

    private final long nanos;

    public DurationData(long nanos) {
        this.nanos = nanos;
    }

    public static DurationData ofNanos(long nanos) {
        return new DurationData(nanos);
    }

    public static DurationData ofMillis(long millis) {
        return new DurationData(millis * 1_000_000L);
    }

    public static DurationData ofSeconds(long seconds) {
        return new DurationData(seconds * 1_000_000_000L);
    }

    public static DurationData of(Duration d) {
        Objects.requireNonNull(d, "duration");
        return new DurationData(d.toNanos());
    }

    public static DurationData parse(String s) {
        // ISO-8601 duration or plain millis number
        if (s == null || s.isBlank()) throw new IllegalArgumentException("empty duration");
        String t = s.trim();
        if (t.startsWith("P") || t.startsWith("p")) {
            return of(Duration.parse(t.toUpperCase()));
        }
        // "5s", "100ms", "1h", raw nanos number
        try {
            if (t.matches("^-?\\d+$")) return new DurationData(Long.parseLong(t));
            long mult = TimeUnit.parseMultiplier(t);
            TimeUnit u = TimeUnit.parse(t);
            if (u.isCalendarBased()) {
                throw new IllegalArgumentException("calendar unit not a fixed duration: " + t);
            }
            return new DurationData(mult * u.toNanos());
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("cannot parse duration: " + s, e);
        }
    }

    public long toNanos() {
        return nanos;
    }

    public long toMillis() {
        return nanos / 1_000_000L;
    }

    public long toSeconds() {
        return nanos / 1_000_000_000L;
    }

    public Duration toDuration() {
        return Duration.ofNanos(nanos);
    }

    public DurationData plus(DurationData other) {
        return new DurationData(Math.addExact(nanos, other.nanos));
    }

    public DurationData minus(DurationData other) {
        return new DurationData(Math.subtractExact(nanos, other.nanos));
    }

    public DurationData multiply(long factor) {
        return new DurationData(Math.multiplyExact(nanos, factor));
    }

    public DurationData multiply(double factor) {
        return new DurationData(Math.round(nanos * factor));
    }

    /** Integer division of durations → long quotient. */
    public long divide(DurationData other) {
        if (other.nanos == 0) throw new ArithmeticException("divide by zero duration");
        return nanos / other.nanos;
    }

    public DurationData divide(long divisor) {
        if (divisor == 0) throw new ArithmeticException("divide by zero");
        return new DurationData(nanos / divisor);
    }

    public DurationData abs() {
        return nanos < 0 ? new DurationData(-nanos) : this;
    }

    public DurationData negate() {
        return new DurationData(-nanos);
    }

    public int compareTo(DurationData other) {
        return Long.compare(nanos, other.nanos);
    }

    @Override
    public String getDataType() {
        return "DURATION";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow Duration nanoseconds
        return nanos;
    }

    @Override
    public String getShortDesc() {
        return nanos + "ns";
    }

    @Override
    public Number getNumericValue() {
        return nanos;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DurationData that)) return false;
        return nanos == that.nanos;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(nanos);
    }

    @Override
    public String toString() {
        return "DurationData[" + getShortDesc() + "]";
    }
}
