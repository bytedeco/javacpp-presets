package org.bytedeco.pytorch.data.dataframe.dtype;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.Objects;

/**
 * 日期数据容器（适配 Arrow DateType（day-based））
 * 支持ISO 8601格式、Epoch天数（Arrow标准存储方式）
 */
public class DateData extends AbstractDataValue implements TemporalData  {
    private static final long serialVersionUID = 1L;
    private static final DateTimeFormatter ISO_FORMAT = DateTimeFormatter.ISO_LOCAL_DATE;
    // Arrow标准：从1970-01-01开始的天数（int32存储）
    private int epochDays;
    // 格式化的日期对象（缓存）
    private transient LocalDate localDate;

    // 构造：从LocalDate
    public DateData(LocalDate localDate) {
        this.localDate = Objects.requireNonNull(localDate);
        this.epochDays = (int) ChronoUnit.DAYS.between(LocalDate.of(1970, 1, 1), localDate);
    }

    // 构造：从Epoch天数（Arrow存储格式）
    public DateData(int epochDays) {
        this.epochDays = epochDays;
        this.localDate = LocalDate.of(1970, 1, 1).plusDays(epochDays);
    }

    // 构造：从ISO字符串（yyyy-MM-dd）
    public DateData(String isoDate) {
        this(LocalDate.parse(isoDate, ISO_FORMAT));
    }

    // Arrow适配：获取Epoch天数（Arrow DateType的底层存储）
    public int toArrowEpochDays() {
        return epochDays;
    }

    // 格式化为指定字符串
    public String format(DateTimeFormatter formatter) {
        return localDate.format(formatter);
    }

    // Getter
    public LocalDate getLocalDate() { return localDate; }
    public int getEpochDays() { return epochDays; }

    @Override
    public String toString() {
        return "DateData[date=" + localDate.format(ISO_FORMAT) + ", epochDays=" + epochDays + "]";
    }

    @Override
    public String getDataType() {
        return "DATE";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow DateType适配：返回epochDays（int32）
        return epochDays;
    }

    @Override
    public String getShortDesc() {
        return localDate.format(ISO_FORMAT) + " (epochDays=" + epochDays + ")";
    }

    // ========== 重写有效性校验 ==========
    @Override
    public boolean isValid() {
        return super.isValid() && localDate != null;
    }

    @Override
    public String getPrecision() {
        return "DAYS"; // 日期精度为天
    }

    @Override
    public long toEpochMillis() {
        // 转换为当天0点的毫秒时间戳
        return localDate.atStartOfDay().toInstant(java.time.ZoneOffset.UTC).toEpochMilli();
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    // ========== 原有方法（保留） ==========
//    public int toArrowEpochDays() {
//        return epochDays;
//    }

//    public String format(DateTimeFormatter formatter) {
//        return localDate.format(formatter);
//    }

}