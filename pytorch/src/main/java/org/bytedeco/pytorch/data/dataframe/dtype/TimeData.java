package org.bytedeco.pytorch.data.dataframe.dtype;

import org.bytedeco.pytorch.data.dataframe.enums.PrecisionType;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Objects;

/**
 * 时间数据容器（适配 Arrow TimeType（millis/micros/nanos））
 * 支持不同精度的时间存储，对齐Arrow Time类型标准
 */
public class TimeData extends AbstractDataValue implements TemporalData  {
    private static final long serialVersionUID = 1L;
    private static final DateTimeFormatter ISO_FORMAT = DateTimeFormatter.ISO_LOCAL_TIME;

    // 精度枚举（对齐Arrow TimeType）
//    public enum Precision {
//        MILLIS, MICROS, NANOS
//    }

    // 底层存储：对应精度的数值（如millis是从00:00开始的毫秒数）
    private long timeValue;
    // 精度
    private PrecisionType precision;
    // 格式化的时间对象
    private transient LocalTime localTime;

    // 构造：从LocalTime + 精度
    public TimeData(LocalTime localTime, PrecisionType precision) {
        this.localTime = Objects.requireNonNull(localTime);
        this.precision = Objects.requireNonNull(precision);
        // 转换为对应精度的数值
        this.timeValue = switch (precision) {
            case SECONDS -> localTime.toNanoOfDay() / 1_000_000_000;
            case MILLIS -> localTime.toNanoOfDay() / 1_000_000;
            case MICROS -> localTime.toNanoOfDay() / 1_000;
            case NANOS -> localTime.toNanoOfDay();

        };
    }

    // 构造：从时间数值 + 精度（Arrow存储格式）
    public TimeData(long timeValue, PrecisionType precision) {
        this.precision = Objects.requireNonNull(precision);
        this.timeValue = timeValue;
        // 转换为LocalTime
        long nanoOfDay = switch (precision) {
            case SECONDS -> timeValue * 1_000_000_000; // 秒转纳秒
            case MILLIS -> timeValue * 1_000_000;      // 毫秒转纳秒
            case MICROS -> timeValue * 1_000;          // 微秒转纳秒
            case NANOS -> timeValue;                   // 原始纳秒
        };
        this.localTime = LocalTime.ofNanoOfDay(nanoOfDay);
    }

    // Arrow适配：获取对应精度的数值
    public long toArrowTimeValue() {
        return timeValue;
    }

    // Getter
    public LocalTime getLocalTime() { return localTime; }
    public long getTimeValue() { return timeValue; }
//    public Precision getPrecision() { return precision; }

    @Override
    public String getDataType() {
        return "TIME";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回对应精度的数值
        return timeValue;
    }

    @Override
    public String getShortDesc() {
        return String.format("time=%s, precision=%s, value=%d",
                localTime.format(ISO_FORMAT), precision, timeValue);
    }

    // ========== 重写有效性校验 ==========
    @Override
    public boolean isValid() {
        // 基础校验 + 时间专属校验：localTime/precision非空，时间值非负
        return super.isValid() && localTime != null && precision != null && timeValue >= 0;
    }

    @Override
    public String getPrecision() {
        return precision.name();
    }

    @Override
    public long toEpochMillis() {
        // 转换为当天0点到该时间的毫秒数
        return localTime.toNanoOfDay() / 1_000_000;
    }
    @Override
    public String toString() {
        return String.format("TimeData[time=%s, precision=%s, value=%d]",
                localTime.format(ISO_FORMAT), precision, timeValue);
    }

    @Override
    public Number getNumericValue(){
        return null;
    }
}