package org.bytedeco.pytorch.dataframe.json;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.data.json.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * DataFrame → JSON / JSONL writer (pandas-compatible orients).
 */
public final class JsonWriter {
    private JsonWriter() {}

    public static void write(DataFrame df, String path) throws IOException {
        write(df, Path.of(path), JsonOptions.defaults());
    }

    public static void write(DataFrame df, String path, JsonOptions options) throws IOException {
        write(df, Path.of(path), options);
    }

    public static void write(DataFrame df, Path path, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (opt.orient() == JsonOptions.Orient.LINES) {
            writeJsonl(df, path, opt);
            return;
        }
        JsonValue value = toJsonValue(df, opt);
        org.bytedeco.pytorch.data.json.JsonWriter.write(value, path, opt.toWriteOptions());
    }

    public static void write(DataFrame df, Writer writer, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (opt.orient() == JsonOptions.Orient.LINES) {
            writeJsonl(df, writer, opt);
            return;
        }
        JsonValue value = toJsonValue(df, opt);
        org.bytedeco.pytorch.data.json.JsonWriter.write(value, writer, opt.toWriteOptions());
    }

    public static void write(DataFrame df, OutputStream out, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        try (Writer w = new BufferedWriter(new OutputStreamWriter(out, opt.charset()))) {
            write(df, w, opt);
        }
    }

    public static String toString(DataFrame df) {
        return toString(df, JsonOptions.defaults());
    }

    public static String toString(DataFrame df, JsonOptions options) {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (opt.orient() == JsonOptions.Orient.LINES) {
            StringBuilder sb = new StringBuilder();
            try {
                writeJsonl(df, new StringWriter() {
                    @Override public void write(String str) { sb.append(str); }
                    @Override public void write(char[] cbuf, int off, int len) { sb.append(cbuf, off, len); }
                    @Override public void flush() {}
                    @Override public void close() {}
                }, opt);
            } catch (IOException e) {
                throw new JsonException("stringify failed", e);
            }
            return sb.toString();
        }
        return org.bytedeco.pytorch.data.json.JsonWriter.toString(toJsonValue(df, opt), opt.toWriteOptions());
    }

    public static void writeJsonl(DataFrame df, String path) throws IOException {
        writeJsonl(df, Path.of(path), JsonOptions.lines());
    }

    public static void writeJsonl(DataFrame df, Path path, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.lines() : options;
        try (Writer w = Files.newBufferedWriter(path, opt.charset())) {
            writeJsonl(df, w, opt);
        }
    }

    public static void writeJsonl(DataFrame df, Writer writer, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.lines() : options;
        JsonWriteOptions wopt = JsonWriteOptions.builder()
            .pretty(false)
            .nullHandling(opt.writeNulls()
                ? JsonWriteOptions.NullHandling.WRITE_NULL
                : JsonWriteOptions.NullHandling.OMIT)
            .charset(opt.charset())
            .build();
        int rows = df.rowCount();
        List<Column> cols = df.columns();
        for (int r = 0; r < rows; r++) {
            JsonValue obj = rowToObject(df, cols, r, opt);
            org.bytedeco.pytorch.data.json.JsonWriter.write(obj, writer, wopt);
            writer.write('\n');
        }
        writer.flush();
    }

    /** Convert DataFrame to a JsonValue tree according to orient. */
    public static JsonValue toJsonValue(DataFrame df, JsonOptions options) {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        switch (opt.orient()) {
            case RECORDS:
            case LINES:
                return toRecords(df, opt);
            case COLUMNS:
                return toColumns(df, opt);
            case VALUES:
                return toValues(df, opt);
            case INDEX:
                return toIndex(df, opt);
            case SPLIT:
                return toSplit(df, opt);
            case TABLE:
                return toTable(df, opt);
            default:
                return toRecords(df, opt);
        }
    }

    private static JsonValue toRecords(DataFrame df, JsonOptions opt) {
        JsonValue arr = JsonValue.array();
        List<Column> cols = df.columns();
        for (int r = 0; r < df.rowCount(); r++) {
            arr.add(rowToObject(df, cols, r, opt));
        }
        return arr;
    }

    private static JsonValue rowToObject(DataFrame df, List<Column> cols, int r, JsonOptions opt) {
        JsonValue obj = JsonValue.object();
        for (Column c : cols) {
            Object v = c.get(r);
            if (v == null && !opt.writeNulls()) continue;
            obj.put(c.name(), cellToJson(v, c.dtype(), opt));
        }
        return obj;
    }

    private static JsonValue toColumns(DataFrame df, JsonOptions opt) {
        JsonValue obj = JsonValue.object();
        for (Column c : df.columns()) {
            JsonValue arr = JsonValue.array();
            for (int r = 0; r < df.rowCount(); r++) {
                arr.add(cellToJson(c.get(r), c.dtype(), opt));
            }
            obj.put(c.name(), arr);
        }
        return obj;
    }

    private static JsonValue toValues(DataFrame df, JsonOptions opt) {
        JsonValue arr = JsonValue.array();
        List<Column> cols = df.columns();
        for (int r = 0; r < df.rowCount(); r++) {
            JsonValue row = JsonValue.array();
            for (Column c : cols) {
                row.add(cellToJson(c.get(r), c.dtype(), opt));
            }
            arr.add(row);
        }
        return arr;
    }

    private static JsonValue toIndex(DataFrame df, JsonOptions opt) {
        JsonValue obj = JsonValue.object();
        List<Column> cols = df.columns();
        for (int r = 0; r < df.rowCount(); r++) {
            obj.put(String.valueOf(r), rowToObject(df, cols, r, opt));
        }
        return obj;
    }

    private static JsonValue toSplit(DataFrame df, JsonOptions opt) {
        JsonValue root = JsonValue.object();
        JsonValue columns = JsonValue.array();
        for (Column c : df.columns()) columns.add(JsonValue.of(c.name()));
        root.put("columns", columns);

        JsonValue index = JsonValue.array();
        for (int r = 0; r < df.rowCount(); r++) index.add(JsonValue.of(r));
        root.put("index", index);

        root.put("data", toValues(df, opt));
        return root;
    }

    private static JsonValue toTable(DataFrame df, JsonOptions opt) {
        JsonValue root = JsonValue.object();
        JsonValue schema = JsonValue.object();
        JsonValue fields = JsonValue.array();
        for (Column c : df.columns()) {
            JsonValue f = JsonValue.object();
            f.put("name", c.name());
            f.put("type", dtypeToPandas(c.dtype()));
            fields.add(f);
        }
        schema.put("fields", fields);
        schema.put("pandas_version", "1.4.0");
        root.put("schema", schema);
        root.put("data", toRecords(df, opt));
        return root;
    }

    private static JsonValue cellToJson(Object v, Column.DType dtype, JsonOptions opt) {
        if (v == null) return JsonValue.NULL;
        switch (dtype) {
            case BOOLEAN:
                if (v instanceof Boolean) return JsonValue.of((Boolean) v);
                return JsonValue.of(Boolean.parseBoolean(String.valueOf(v)));
            case INT32:
            case INT64:
                if (v instanceof Number) return JsonValue.of(((Number) v).longValue());
                try { return JsonValue.of(Long.parseLong(String.valueOf(v))); }
                catch (Exception e) { return JsonValue.of(String.valueOf(v)); }
            case FLOAT32:
            case FLOAT64:
                if (v instanceof Number) {
                    double d = ((Number) v).doubleValue();
                    if (Double.isNaN(d) || Double.isInfinite(d)) return JsonValue.NULL;
                    return JsonValue.of(d);
                }
                try { return JsonValue.of(Double.parseDouble(String.valueOf(v))); }
                catch (Exception e) { return JsonValue.of(String.valueOf(v)); }
            case DATE:
                if (v instanceof LocalDate) {
                    return formatDate((LocalDate) v, opt);
                }
                return JsonValue.of(String.valueOf(v));
            case DATETIME:
                if (v instanceof LocalDateTime) {
                    return formatDateTime((LocalDateTime) v, opt);
                }
                return JsonValue.of(String.valueOf(v));
            case VECTOR:
            case EMBEDDING:
                if (v instanceof float[]) {
                    JsonValue arr = JsonValue.array();
                    for (float x : (float[]) v) arr.add(JsonValue.of(x));
                    return arr;
                }
                if (v instanceof double[]) {
                    JsonValue arr = JsonValue.array();
                    for (double x : (double[]) v) arr.add(JsonValue.of(x));
                    return arr;
                }
                if (v instanceof int[] || v instanceof long[] || v instanceof List) {
                    return JsonValue.fromJava(v);
                }
                return JsonValue.fromJava(v);
            case LIST:
                if (v instanceof int[] || v instanceof long[] || v instanceof float[]
                    || v instanceof double[] || v instanceof boolean[]
                    || v instanceof List || v instanceof Object[]) {
                    return JsonValue.fromJava(v);
                }
                if (v instanceof String) {
                    try { return org.bytedeco.pytorch.data.json.JsonParser.parse((String) v); }
                    catch (Exception e) { return JsonValue.of((String) v); }
                }
                return JsonValue.fromJava(v);
            case MAP:
            case STRUCT:
                if (v instanceof Map) return JsonValue.fromJava(v);
                if (v instanceof String) {
                    try { return org.bytedeco.pytorch.data.json.JsonParser.parse((String) v); }
                    catch (Exception e) { return JsonValue.of((String) v); }
                }
                return JsonValue.fromJava(v);
            case JSON:
                // try parse as JSON, else string
                if (v instanceof String) {
                    String s = (String) v;
                    try {
                        return JsonParser.parse(s);
                    } catch (Exception e) {
                        return JsonValue.of(s);
                    }
                }
                return JsonValue.fromJava(v);
            case STRING:
            default:
                // nested JSON stored as string — re-emit as structured if possible
                if (v instanceof String) {
                    String s = (String) v;
                    String t = s.trim();
                    if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
                        try { return JsonParser.parse(s); } catch (Exception ignored) {}
                    }
                    return JsonValue.of(s);
                }
                if (v instanceof Map || v instanceof List || v instanceof JsonValue) {
                    return JsonValue.fromJava(v);
                }
                return JsonValue.of(String.valueOf(v));
        }
    }

    private static JsonValue formatDate(LocalDate d, JsonOptions opt) {
        switch (opt.dateFormat()) {
            case EPOCH_MILLIS:
                return JsonValue.of(d.atStartOfDay().toInstant(ZoneOffset.UTC).toEpochMilli());
            case EPOCH_SECONDS:
                return JsonValue.of(d.atStartOfDay().toEpochSecond(ZoneOffset.UTC));
            case ISO:
            case STRING:
            default:
                return JsonValue.of(d.format(DateTimeFormatter.ISO_LOCAL_DATE));
        }
    }

    private static JsonValue formatDateTime(LocalDateTime d, JsonOptions opt) {
        switch (opt.dateFormat()) {
            case EPOCH_MILLIS:
                return JsonValue.of(d.toInstant(ZoneOffset.UTC).toEpochMilli());
            case EPOCH_SECONDS:
                return JsonValue.of(d.toEpochSecond(ZoneOffset.UTC));
            case ISO:
            case STRING:
            default:
                return JsonValue.of(d.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        }
    }

    private static String dtypeToPandas(Column.DType dt) {
        switch (dt) {
            case INT32:
            case INT64: return "integer";
            case FLOAT32:
            case FLOAT64: return "number";
            case BOOLEAN: return "boolean";
            case DATE: return "date";
            case DATETIME: return "datetime";
            default: return "string";
        }
    }
}
