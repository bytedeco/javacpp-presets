package org.bytedeco.pytorch.dataframe.dtype;

/**
 * 时间相关数据（日期/时间/时间戳）通用接口
 */
public interface TemporalData extends DataValue {

    /**
     * 获取时间精度（如MILLIS/MICROS/NANOS/SECONDS）
     * @return 精度字符串
     */
    String getPrecision();

    /**
     * 转换为标准时间戳（毫秒）
     * @return 毫秒级时间戳
     */
    long toEpochMillis();
}