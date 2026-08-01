package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.PrecisionType;
import org.bytedeco.pytorch.dataframe.temporal.BusinessCalendar;
import org.bytedeco.pytorch.dataframe.temporal.TemporalOps;
import org.bytedeco.pytorch.dataframe.temporal.TimeZone;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.Objects;

/**
 * 时间戳容器（适配 Arrow TimestampType）
 * 支持时区、不同精度、业务日标记，对齐 Arrow Timestamp 标准。
 *
 * <p>Enterprise enhancements (2026-08): full ZoneId, business-day flag,
 * Arrow IPC / JSON serialization consistency helpers.
 */
public class TimestampData extends AbstractDataValue implements TemporalData {
    private static final long serialVersionUID = 1L;
    private static final DateTimeFormatter ISO_FORMAT = DateTimeFormatter.ISO_INSTANT.withZone(ZoneId.of("UTC"));

    // 底层存储：从1970-01-01T00:00:00Z开始的对应精度数值
    private long timestampValue;
    // 精度
    private PrecisionType precision;
    // 时区（Arrow Timestamp支持时区）
    private ZoneId timeZone;
    // 格式化的Instant对象
    private transient Instant instant;
    /** Optional business-day annotation (null = unknown / not evaluated). */
    private Boolean businessDay;

    // 构造：从Instant + 精度 + 时区
    public TimestampData(Instant instant, PrecisionType precision, ZoneId timeZone) {
        this.instant = Objects.requireNonNull(instant);
        this.precision = Objects.requireNonNull(precision);
        this.timeZone = Objects.requireNonNull(timeZone);
        this.businessDay = null;
        this.timestampValue = switch (precision) {
            case SECONDS -> instant.getEpochSecond();
            case MILLIS -> instant.toEpochMilli();
            case MICROS -> instant.getEpochSecond() * 1_000_000 + instant.getNano() / 1_000;
            case NANOS -> instant.getEpochSecond() * 1_000_000_000L + instant.getNano();
        };
    }

    // 构造：从数值 + 精度 + 时区（Arrow存储格式）
    public TimestampData(long timestampValue, PrecisionType precision, ZoneId timeZone) {
        this.precision = Objects.requireNonNull(precision);
        this.timeZone = Objects.requireNonNull(timeZone);
        this.timestampValue = timestampValue;
        this.businessDay = null;
        this.instant = switch (precision) {
            case SECONDS -> Instant.ofEpochSecond(timestampValue);
            case MILLIS -> Instant.ofEpochMilli(timestampValue);
            case MICROS -> Instant.ofEpochSecond(timestampValue / 1_000_000, (timestampValue % 1_000_000) * 1_000);
            case NANOS -> Instant.ofEpochSecond(timestampValue / 1_000_000_000L, timestampValue % 1_000_000_000L);
        };
    }

    /** Convenience: millis precision, UTC. */
    public static TimestampData ofEpochMilli(long millis) {
        return new TimestampData(millis, PrecisionType.MILLIS, ZoneId.of("UTC"));
    }

    public static TimestampData of(Instant instant) {
        return new TimestampData(instant, PrecisionType.NANOS, ZoneId.of("UTC"));
    }

    public static TimestampData of(Instant instant, ZoneId zone) {
        return new TimestampData(instant, PrecisionType.NANOS, zone == null ? ZoneId.of("UTC") : zone);
    }

    public Instant getInstant() {
        ensureInstant();
        return instant;
    }

    public long getTimestampValue() { return timestampValue; }

    public ZoneId getTimeZone() { return timeZone; }

    public PrecisionType precisionType() { return precision; }

    /** Nullable business-day flag; call {@link #withBusinessDay(BusinessCalendar)} to compute. */
    public Boolean getBusinessDay() { return businessDay; }

    public boolean isBusinessDay(BusinessCalendar cal) {
        LocalDate d = toLocalDate();
        return (cal == null ? TemporalOps.DEFAULT_CALENDAR : cal).isBusinessDay(d);
    }

    public TimestampData withZone(ZoneId newZone) {
        Objects.requireNonNull(newZone, "newZone");
        TimestampData t = new TimestampData(getInstant(), precision, newZone);
        t.businessDay = this.businessDay;
        return t;
    }

    /** Convert wall representation to another zone keeping the same instant. */
    public TimestampData convertZone(ZoneId newZone) {
        return withZone(newZone);
    }

    public TimestampData withBusinessDay(BusinessCalendar cal) {
        TimestampData t = new TimestampData(getInstant(), precision, timeZone);
        t.businessDay = t.isBusinessDay(cal);
        return t;
    }

    public TimestampData withBusinessDayFlag(Boolean flag) {
        TimestampData t = new TimestampData(getInstant(), precision, timeZone);
        t.businessDay = flag;
        return t;
    }

    public ZonedDateTime toZonedDateTime() {
        return TimeZone.atZone(getInstant(), timeZone);
    }

    public LocalDateTime toLocalDateTime() {
        return toZonedDateTime().toLocalDateTime();
    }

    public LocalDate toLocalDate() {
        return toZonedDateTime().toLocalDate();
    }

    public TimestampData plusMillis(long millis) {
        return new TimestampData(getInstant().plusMillis(millis), precision, timeZone)
                .withBusinessDayFlag(null);
    }

    public TimestampData plus(org.bytedeco.pytorch.dataframe.temporal.DurationData d) {
        Objects.requireNonNull(d, "duration");
        return new TimestampData(getInstant().plusNanos(d.toNanos()), precision, timeZone)
                .withBusinessDayFlag(null);
    }

    public org.bytedeco.pytorch.dataframe.temporal.DurationData until(TimestampData other) {
        Objects.requireNonNull(other, "other");
        long nanos = java.time.Duration.between(getInstant(), other.getInstant()).toNanos();
        return org.bytedeco.pytorch.dataframe.temporal.DurationData.ofNanos(nanos);
    }

    private void ensureInstant() {
        if (instant == null) {
            instant = switch (precision) {
                case SECONDS -> Instant.ofEpochSecond(timestampValue);
                case MILLIS -> Instant.ofEpochMilli(timestampValue);
                case MICROS -> Instant.ofEpochSecond(timestampValue / 1_000_000, (timestampValue % 1_000_000) * 1_000);
                case NANOS -> Instant.ofEpochSecond(timestampValue / 1_000_000_000L, timestampValue % 1_000_000_000L);
            };
        }
    }

    @Override
    public String getDataType() {
        return "TIMESTAMP";
    }

    @Override
    public Object toArrowCompatible() {
        return toArrowTimestampDesc();
    }

    @Override
    public String getShortDesc() {
        ensureInstant();
        return String.format("ts=%s, tz=%s, precision=%s, value=%d, biz=%s",
                ISO_FORMAT.format(instant), timeZone.getId(), precision, timestampValue, businessDay);
    }

    @Override
    public boolean isValid() {
        ensureInstant();
        return super.isValid() && instant != null && precision != null && timeZone != null;
    }

    @Override
    public String getPrecision() {
        return precision.name();
    }

    @Override
    public long toEpochMillis() {
        return getInstant().toEpochMilli();
    }

    public Map<String, Object> toArrowTimestampDesc() {
        Map<String, Object> desc = new java.util.LinkedHashMap<>();
        desc.put("value", timestampValue);
        desc.put("precision", precision.name().toLowerCase());
        desc.put("timezone", timeZone.getId());
        if (businessDay != null) desc.put("business_day", businessDay);
        return desc;
    }

    /** JSON-friendly map for cross-language serialization checks. */
    public Map<String, Object> toJsonMap() {
        Map<String, Object> m = toArrowTimestampDesc();
        m.put("iso", ISO_FORMAT.format(getInstant()));
        m.put("epoch_millis", toEpochMillis());
        return m;
    }

    @Override
    public String toString() {
        ensureInstant();
        return String.format("TimestampData[ts=%s, tz=%s, precision=%s, value=%d, biz=%s]",
                ISO_FORMAT.format(instant), timeZone.getId(), precision, timestampValue, businessDay);
    }

    @Override
    public Number getNumericValue() {
        return toEpochMillis();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TimestampData that)) return false;
        return timestampValue == that.timestampValue
                && precision == that.precision
                && Objects.equals(timeZone, that.timeZone)
                && Objects.equals(businessDay, that.businessDay);
    }

    @Override
    public int hashCode() {
        return Objects.hash(timestampValue, precision, timeZone, businessDay);
    }
}
