package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;
import org.bytedeco.pytorch.dataframe.enums.PrecisionType;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 日志记录容器（适配 Arrow StructType，封装日志核心字段）
 * 包含日志级别、时间、内容、上下文等标准字段
 */
public class LogRecordData extends AbstractDataValue implements StructuredData {
    private static final long serialVersionUID = 1L;

    // 日志核心字段（对齐标准日志格式）
    private String level;        // INFO/WARN/ERROR/DEBUG
    private Instant timestamp;   // 日志产生时间
    private String message;      // 日志内容
    private String loggerName;   // 日志器名称
    private String threadName;   // 线程名
    private String stackTrace;   // 异常堆栈（可选）
    private Map<String, String> context; // 上下文参数（如traceId）

    // 基础构造
    public LogRecordData(String level, Instant timestamp, String message) {
        this.level = Objects.requireNonNull(level);
        this.timestamp = Objects.requireNonNull(timestamp);
        this.message = Objects.requireNonNull(message);
        this.context = new HashMap<>();
    }

    /**
     * 覆盖设置日志上下文（替换原有所有上下文）
     * @param context 新的上下文键值对
     */
    public void setContext(Map<?, ?> context) {
        // 清空原有上下文
        this.context.clear();

        // 空值判断：避免传入 null 导致空指针
        if (context == null || context.isEmpty()) {
            return;
        }

        // 遍历并转换为 String -> String 格式（统一类型，避免类型不一致问题）
        for (Map.Entry<?, ?> entry : context.entrySet()) {
            String key = entry.getKey() == null ? "" : entry.getKey().toString();
            String value = entry.getValue() == null ? "" : entry.getValue().toString();
            this.context.put(key, value);
        }
    }
    // 完整构造
    public LogRecordData(String level, Instant timestamp, String message, String loggerName, String threadName) {
        this(level, timestamp, message);
        this.loggerName = loggerName;
        this.threadName = threadName;
    }

    // 添加上下文参数
    public void addContext(String key, String value) {
        context.put(key, value);
    }

    // Arrow适配：转换为Arrow Struct（日志记录本质是结构化数据）
    public StructData toArrowStruct() {
        Map<String, ColumnType> fields = new HashMap<>();
        fields.put("level", ColumnType.STRING);
        fields.put("timestamp", ColumnType.TIMESTAMP);
        fields.put("message", ColumnType.STRING);
        fields.put("loggerName", ColumnType.STRING);
        fields.put("threadName", ColumnType.STRING);
        fields.put("stackTrace", ColumnType.STRING);
        fields.put("context", ColumnType.MAP_VIEW);

        StructData struct = new StructData("log_record", fields);
        struct.addField("level", ColumnType.STRING, level);
        struct.addField("timestamp", ColumnType.TIMESTAMP, new TimestampData(timestamp, PrecisionType.MILLIS, java.time.ZoneId.of("UTC")));
        struct.addField("message", ColumnType.STRING, message);
        struct.addField("loggerName", ColumnType.STRING, loggerName);
        struct.addField("threadName", ColumnType.STRING, threadName);
        struct.addField("stackTrace", ColumnType.STRING, stackTrace);
        struct.addField("context", ColumnType.MAP_VIEW, new MapViewData(new HashMap<>(context), ColumnType.STRING, ColumnType.STRING));

        return struct;
    }

    // Getter & Setter
    public String getLevel() { return level; }
    public void setLevel(String level) { this.level = level; }

    public void setTimestamp(Instant ts){
        this.timestamp = ts;
    }
    public Instant getTimestamp() { return timestamp; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public String getLoggerName() { return loggerName; }
    public void setLoggerName(String loggerName) { this.loggerName = loggerName; }

    public String getThreadName() { return threadName; }
    public void setThreadName(String threadName) { this.threadName = threadName; }

    public String getStackTrace() { return stackTrace; }
    public void setStackTrace(String stackTrace) { this.stackTrace = stackTrace; }
    public Map<String, String> getContext() { return new HashMap<>(context); }

    @Override
    public String getDataType() {
        return "LOG_RECORD";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：转换为Arrow Struct格式（复用原有toArrowStruct方法）
        return toArrowStruct();
    }

    @Override
    public String getShortDesc() {
        return String.format("level=%s, ts=%s, logger=%s, msg='%s', contextSize=%d",
                level, timestamp, loggerName,
                message.length() > 50 ? message.substring(0, 50) + "..." : message,
                context.size());
    }

    // ========== 重写有效性校验 ==========
    @Override
    public boolean isValid() {
        // 基础校验 + 日志专属校验：级别/时间/内容非空，级别为合法值
        return super.isValid()
                && level != null && !level.trim().isEmpty()
                && timestamp != null
                && message != null && !message.trim().isEmpty()
                && context != null
                && isLevelValid(level);
    }

    private boolean isLevelValid(String level) {
        return level.equalsIgnoreCase("INFO")
                || level.equalsIgnoreCase("WARN")
                || level.equalsIgnoreCase("ERROR")
                || level.equalsIgnoreCase("DEBUG")
                || level.equalsIgnoreCase("TRACE")
                || level.equalsIgnoreCase("FATAL");
    }

    @Override
    public int getSize() {
        // LogRecord大小：上下文参数数量 + 核心字段数量
        int coreFieldCount = 6; // level/timestamp/message/loggerName/threadName/stackTrace
        return coreFieldCount + context.size();
    }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("level", level);
        map.put("timestamp", timestamp);
        map.put("message", message);
        map.put("loggerName", loggerName);
        map.put("threadName", threadName);
        map.put("stackTrace", stackTrace);
        map.put("context", context);
        return map;
    }

    @Override
    public String toString() {
        return String.format("LogRecordData[level=%s, ts=%s, logger=%s, msg='%s', context=%s]",
                level, timestamp, loggerName, message, context);
    }

    @Override
    public Number getNumericValue(){
        return null;
    }
}