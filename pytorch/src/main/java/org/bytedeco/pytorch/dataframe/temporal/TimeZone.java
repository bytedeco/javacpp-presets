package org.bytedeco.pytorch.dataframe.temporal;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Objects;

/**
 * Timezone-aware conversion helpers for temporal columns.
 *
 * <p>Wraps {@link ZoneId} operations used by TimestampData and DateTimeIndex.
 * Does not reimplement the TZ database — delegates to {@code java.time}.
 */
public final class TimeZone {

    private TimeZone() {}

    public static final ZoneId UTC = ZoneOffset.UTC;

    public static ZoneId of(String id) {
        return ZoneId.of(Objects.requireNonNull(id, "id"));
    }

    /** Convert instant to ZonedDateTime in target zone. */
    public static ZonedDateTime atZone(Instant instant, ZoneId zone) {
        return Objects.requireNonNull(instant).atZone(zone == null ? UTC : zone);
    }

    /** Convert LocalDateTime assumed in {@code from} zone to Instant. */
    public static Instant toInstant(LocalDateTime ldt, ZoneId from) {
        return Objects.requireNonNull(ldt).atZone(from == null ? UTC : from).toInstant();
    }

    /**
     * Convert wall-clock LocalDateTime from one zone to another, preserving instant
     * (tz convert), not wall time.
     */
    public static LocalDateTime convert(LocalDateTime ldt, ZoneId from, ZoneId to) {
        Instant in = toInstant(ldt, from);
        return in.atZone(to == null ? UTC : to).toLocalDateTime();
    }

    /**
     * Localize naive LocalDateTime as if it were wall time in {@code zone}
     * (pandas {@code tz_localize}).
     */
    public static ZonedDateTime localize(LocalDateTime ldt, ZoneId zone) {
        return Objects.requireNonNull(ldt).atZone(zone == null ? UTC : zone);
    }

    /**
     * Strip zone, keep wall clock (pandas {@code tz_localize(None)} / {@code tz_convert} then naive).
     */
    public static LocalDateTime delocalize(ZonedDateTime zdt) {
        return Objects.requireNonNull(zdt).toLocalDateTime();
    }

    /** True if zone observes DST at the given instant. */
    public static boolean isDst(Instant instant, ZoneId zone) {
        ZoneId z = zone == null ? UTC : zone;
        return z.getRules().isDaylightSavings(Objects.requireNonNull(instant));
    }

    /** UTC offset seconds at instant. */
    public static int offsetSeconds(Instant instant, ZoneId zone) {
        ZoneId z = zone == null ? UTC : zone;
        return z.getRules().getOffset(Objects.requireNonNull(instant)).getTotalSeconds();
    }
}
