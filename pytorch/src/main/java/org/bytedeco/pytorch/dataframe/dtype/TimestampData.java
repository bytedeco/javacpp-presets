package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.PrecisionType;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.Objects;

/**
 * 时间戳容器（适配 Arrow TimestampType）
 * 支持时区、不同精度，对齐Arrow Timestamp标准
 */
public class TimestampData extends AbstractDataValue implements TemporalData{
    private static final long serialVersionUID = 1L;
    private static final DateTimeFormatter ISO_FORMAT = DateTimeFormatter.ISO_INSTANT.withZone(ZoneId.of("UTC"));

    // 精度枚举（对齐Arrow）
//    public enum Precision {
//        SECONDS, MILLIS, MICROS, NANOS
//    }

    // 底层存储：从1970-01-01T00:00:00Z开始的对应精度数值
    private long timestampValue;
    // 精度
    private PrecisionType precision;
    // 时区（Arrow Timestamp支持时区）
    private ZoneId timeZone;
    // 格式化的Instant对象
    private transient Instant instant;

    // 构造：从Instant + 精度 + 时区
    public TimestampData(Instant instant, PrecisionType precision, ZoneId timeZone) {
        this.instant = Objects.requireNonNull(instant);
        this.precision = Objects.requireNonNull(precision);
        this.timeZone = Objects.requireNonNull(timeZone);
        // 转换为对应精度的数值
        this.timestampValue = switch (precision) {
            case SECONDS -> instant.getEpochSecond();
            case MILLIS -> instant.toEpochMilli();
            case MICROS -> instant.getEpochSecond() * 1_000_000 + instant.getNano() / 1_000;
            case NANOS -> instant.getEpochSecond() * 1_000_000_000 + instant.getNano();
        };
    }

    // 构造：从数值 + 精度 + 时区（Arrow存储格式）
    public TimestampData(long timestampValue, PrecisionType precision, ZoneId timeZone) {
        this.precision = Objects.requireNonNull(precision);
        this.timeZone = Objects.requireNonNull(timeZone);
        this.timestampValue = timestampValue;
        // 转换为Instant
        this.instant = switch (precision) {
            case SECONDS -> Instant.ofEpochSecond(timestampValue);
            case MILLIS -> Instant.ofEpochMilli(timestampValue);
            case MICROS -> Instant.ofEpochSecond(timestampValue / 1_000_000, (timestampValue % 1_000_000) * 1_000);
            case NANOS -> Instant.ofEpochSecond(timestampValue / 1_000_000_000, timestampValue % 1_000_000_000);
        };
    }

    // Arrow适配：获取带时区的Timestamp描述
//    public Map<String, Object> toArrowTimestampDesc() {
//        Map<String, Object> desc = new java.util.HashMap<>();
//        desc.put("value", timestampValue);
//        desc.put("precision", precision.name().toLowerCase());
//        desc.put("timezone", timeZone.getId());
//        return desc;
//    }

    // Getter
    public Instant getInstant() { return instant; }
    public long getTimestampValue() { return timestampValue; }
//    public Precision getPrecision() { return precision; }
    public ZoneId getTimeZone() { return timeZone; }

    @Override
    public String getDataType() {
        return "TIMESTAMP";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回带时区的Timestamp描述
        return toArrowTimestampDesc();
    }

    @Override
    public String getShortDesc() {
        return String.format("ts=%s, tz=%s, precision=%s, value=%d",
                ISO_FORMAT.format(instant), timeZone.getId(), precision, timestampValue);
    }

    @Override
    public boolean isValid() {
        // 基础校验 + 时间戳专属校验：instant/precision/timeZone非空
        return super.isValid() && instant != null && precision != null && timeZone != null;
    }

    // ========== 实现 TemporalData 接口 ==========
    @Override
    public String getPrecision() {
        return precision.name();
    }

    @Override
    public long toEpochMillis() {
        // 转换为毫秒级时间戳
        return instant.toEpochMilli();
    }

    // ========== 原有方法（保留） ==========
    public Map<String, Object> toArrowTimestampDesc() {
        Map<String, Object> desc = new java.util.HashMap<>();
        desc.put("value", timestampValue);
        desc.put("precision", precision.name().toLowerCase());
        desc.put("timezone", timeZone.getId());
        return desc;
    }
    @Override
    public String toString() {
        return String.format("TimestampData[ts=%s, tz=%s, precision=%s, value=%d]",
                ISO_FORMAT.format(instant), timeZone.getId(), precision, timestampValue);
    }

    @Override
    public Number getNumericValue(){
        return null;
    }
}