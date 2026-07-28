package org.bytedeco.pytorch.dataframe.json;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.io.ComplexCellCodec;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.data.json.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;

/**
 * Production DataFrame JSON / JSONL reader.
 *
 * <p>Supports pandas-compatible orients ({@code records}, {@code columns},
 * {@code values}, {@code index}, {@code split}, {@code table}), nested
 * flattening, record-path extraction, schema inference, and streaming JSONL.
 */
public final class JsonReader {
    private JsonReader() {}

    // ---- public entry ----

    public static DataFrame read(String path) throws IOException {
        return read(Path.of(path), JsonOptions.defaults());
    }

    public static DataFrame read(String path, JsonOptions options) throws IOException {
        return read(Path.of(path), options);
    }

    public static DataFrame read(Path path) throws IOException {
        return read(path, JsonOptions.defaults());
    }

    public static DataFrame read(Path path, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (opt.orient() == JsonOptions.Orient.LINES || isProbablyJsonl(path, opt)) {
            return readJsonl(path, opt);
        }
        try (InputStream in = Files.newInputStream(path)) {
            return read(in, opt);
        }
    }

    public static DataFrame read(InputStream in, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (opt.orient() == JsonOptions.Orient.LINES) {
            return readJsonl(in, opt);
        }
        JsonValue root = JsonParser.parse(in, opt.toReadOptions());
        return fromJsonValue(root, opt);
    }

    public static DataFrame read(Reader reader, JsonOptions options) throws IOException {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (opt.orient() == JsonOptions.Orient.LINES) {
            return readJsonl(reader, opt);
        }
        JsonValue root = JsonParser.parse(reader, opt.toReadOptions());
        return fromJsonValue(root, opt);
    }

    public static DataFrame readJsonl(String path) throws IOException {
        return readJsonl(Path.of(path), JsonOptions.lines());
    }

    public static DataFrame readJsonl(String path, JsonOptions options) throws IOException {
        return readJsonl(Path.of(path), options == null ? JsonOptions.lines() : options);
    }

    public static DataFrame readJsonl(Path path, JsonOptions options) throws IOException {
        try (InputStream in = Files.newInputStream(path)) {
            return readJsonl(in, options);
        }
    }

    public static DataFrame readJsonl(InputStream in, JsonOptions options) throws IOException {
        JsonOptions opt = normalizeLines(options);
        // use parser entry that strips BOM + applies charset
        JsonValue arr = JsonParser.parseLines(in, opt.toReadOptions());
        List<Map<String, Object>> records = new ArrayList<>();
        if (arr != null && arr.isArray()) {
            for (int i = 0; i < arr.size(); i++) {
                Map<String, Object> rec = valueToRecord(arr.get(i), opt);
                if (rec != null) records.add(rec);
            }
        }
        return recordsToDataFrame(records, opt);
    }

    public static DataFrame readJsonl(Reader reader, JsonOptions options) throws IOException {
        JsonOptions opt = normalizeLines(options);
        List<Map<String, Object>> records = new ArrayList<>();
        JsonParser.parseLines(reader, opt.toReadOptions(), value -> {
            Map<String, Object> rec = valueToRecord(value, opt);
            if (rec != null) records.add(rec);
        });
        return recordsToDataFrame(records, opt);
    }

    /** Parse an in-memory JSON string into a DataFrame. */
    public static DataFrame readString(String json) {
        return readString(json, JsonOptions.defaults());
    }

    public static DataFrame readString(String json, JsonOptions options) {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (opt.orient() == JsonOptions.Orient.LINES) {
            try {
                return readJsonl(new StringReader(json), opt);
            } catch (IOException e) {
                throw new JsonException("JSONL parse failed", e);
            }
        }
        JsonValue root = JsonParser.parse(json, opt.toReadOptions());
        return fromJsonValue(root, opt);
    }

    /** Convert an already-parsed {@link JsonValue} into a DataFrame. */
    public static DataFrame fromJsonValue(JsonValue root, JsonOptions options) {
        JsonOptions opt = options == null ? JsonOptions.defaults() : options;
        if (root == null || root.isNull()) return DataFrame.create();

        // optional record path extraction
        if (opt.recordPath() != null && !opt.recordPath().isEmpty()) {
            JsonValue extracted = JsonPath.get(root, opt.recordPath());
            Map<String, Object> meta = extractMeta(root, opt);
            return fromRecordsArray(extracted, opt, meta);
        }

        switch (opt.orient()) {
            case RECORDS:
            case LINES:
                return fromRecordsOrient(root, opt);
            case COLUMNS:
                return fromColumnsOrient(root, opt);
            case VALUES:
                return fromValuesOrient(root, opt);
            case INDEX:
                return fromIndexOrient(root, opt);
            case SPLIT:
                return fromSplitOrient(root, opt);
            case TABLE:
                return fromTableOrient(root, opt);
            default:
                return fromRecordsOrient(root, opt);
        }
    }

    // ---- orients ----

    private static DataFrame fromRecordsOrient(JsonValue root, JsonOptions opt) {
        if (root.isArray()) {
            return fromRecordsArray(root, opt, null);
        }
        if (root.isObject()) {
            // single record → 1-row frame
            Map<String, Object> rec = valueToRecord(root, opt);
            List<Map<String, Object>> list = new ArrayList<>(1);
            if (rec != null) list.add(rec);
            return recordsToDataFrame(list, opt);
        }
        // scalar → single column
        DataFrame df = DataFrame.create();
        df.addColumn("value", inferDType(root.toJava()));
        df.addRow(convertLeaf(root, opt, "value"));
        return df;
    }

    private static DataFrame fromRecordsArray(JsonValue arr, JsonOptions opt, Map<String, Object> meta) {
        if (arr == null || arr.isNull()) return DataFrame.create();
        if (!arr.isArray()) {
            // wrap single object
            if (arr.isObject()) {
                JsonValue wrap = JsonValue.array(arr);
                return fromRecordsArray(wrap, opt, meta);
            }
            throw new JsonException("Expected array of records, got " + arr.type());
        }
        List<Map<String, Object>> records = new ArrayList<>(arr.size());
        int skip = Math.max(0, opt.skipRows());
        int max = opt.maxRows();
        for (int i = 0; i < arr.size(); i++) {
            if (i < skip) continue;
            if (max >= 0 && records.size() >= max) break;
            JsonValue item = arr.get(i);
            Map<String, Object> rec = valueToRecord(item, opt);
            if (rec == null) continue;
            if (meta != null && !meta.isEmpty()) {
                // meta keys do not overwrite record keys
                for (Map.Entry<String, Object> e : meta.entrySet()) {
                    rec.putIfAbsent(e.getKey(), e.getValue());
                }
            }
            records.add(rec);
        }
        return recordsToDataFrame(records, opt);
    }

    private static DataFrame fromColumnsOrient(JsonValue root, JsonOptions opt) {
        if (!root.isObject()) throw new JsonException("columns orient expects object");
        // determine row count
        int rows = 0;
        LinkedHashMap<String, List<Object>> cols = new LinkedHashMap<>();
        for (String key : root.keySet()) {
            JsonValue v = root.get(key);
            List<Object> data = new ArrayList<>();
            if (v.isArray()) {
                for (int i = 0; i < v.size(); i++) {
                    data.add(convertLeaf(v.get(i), opt, key));
                }
            } else {
                data.add(convertLeaf(v, opt, key));
            }
            rows = Math.max(rows, data.size());
            cols.put(key, data);
        }
        // pad
        for (List<Object> data : cols.values()) {
            while (data.size() < rows) data.add(null);
        }
        return columnsToDataFrame(cols, opt);
    }

    private static DataFrame fromValuesOrient(JsonValue root, JsonOptions opt) {
        if (!root.isArray()) throw new JsonException("values orient expects array of arrays");
        List<String> names = opt.columnNames();
        List<Map<String, Object>> records = new ArrayList<>();
        int skip = Math.max(0, opt.skipRows());
        int max = opt.maxRows();
        int colCount = 0;
        for (int i = 0; i < root.size(); i++) {
            if (i < skip) continue;
            if (max >= 0 && records.size() >= max) break;
            JsonValue row = root.get(i);
            if (!row.isArray()) {
                if (opt.strict()) throw new JsonException("values row is not array at " + i);
                continue;
            }
            colCount = Math.max(colCount, row.size());
            Map<String, Object> rec = new LinkedHashMap<>();
            for (int c = 0; c < row.size(); c++) {
                String name = (names != null && c < names.size()) ? names.get(c) : "col_" + c;
                rec.put(name, convertLeaf(row.get(c), opt, name));
            }
            records.add(rec);
        }
        if (names != null) {
            // ensure all named columns exist
            for (String n : names) {
                for (Map<String, Object> rec : records) rec.putIfAbsent(n, null);
            }
        } else if (colCount > 0) {
            for (int c = 0; c < colCount; c++) {
                String n = "col_" + c;
                for (Map<String, Object> rec : records) rec.putIfAbsent(n, null);
            }
        }
        return recordsToDataFrame(records, opt);
    }

    private static DataFrame fromIndexOrient(JsonValue root, JsonOptions opt) {
        if (!root.isObject()) throw new JsonException("index orient expects object");
        List<Map<String, Object>> records = new ArrayList<>();
        for (String idx : root.keySet()) {
            JsonValue row = root.get(idx);
            Map<String, Object> rec = valueToRecord(row, opt);
            if (rec == null) rec = new LinkedHashMap<>();
            rec.putIfAbsent("index", idx);
            records.add(rec);
            if (opt.maxRows() >= 0 && records.size() >= opt.maxRows()) break;
        }
        return recordsToDataFrame(records, opt);
    }

    private static DataFrame fromSplitOrient(JsonValue root, JsonOptions opt) {
        if (!root.isObject()) throw new JsonException("split orient expects object");
        List<String> columns = new ArrayList<>();
        if (root.has("columns") && root.get("columns").isArray()) {
            for (JsonValue c : root.get("columns").asArray()) {
                columns.add(String.valueOf(c.toJava()));
            }
        }
        if (opt.columnNames() != null) columns = new ArrayList<>(opt.columnNames());

        JsonValue data = root.has("data") ? root.get("data") : JsonValue.array();
        List<String> index = new ArrayList<>();
        if (root.has("index") && root.get("index").isArray()) {
            for (JsonValue ix : root.get("index").asArray()) {
                index.add(String.valueOf(ix.toJava()));
            }
        }

        List<Map<String, Object>> records = new ArrayList<>();
        if (data.isArray()) {
            for (int i = 0; i < data.size(); i++) {
                if (opt.maxRows() >= 0 && records.size() >= opt.maxRows()) break;
                JsonValue row = data.get(i);
                Map<String, Object> rec = new LinkedHashMap<>();
                if (!index.isEmpty() && i < index.size()) {
                    rec.put("index", index.get(i));
                }
                if (row.isArray()) {
                    for (int c = 0; c < row.size(); c++) {
                        String name = c < columns.size() ? columns.get(c) : "col_" + c;
                        rec.put(name, convertLeaf(row.get(c), opt, name));
                    }
                } else if (row.isObject()) {
                    Map<String, Object> m = valueToRecord(row, opt);
                    if (m != null) rec.putAll(m);
                }
                records.add(rec);
            }
        }
        return recordsToDataFrame(records, opt);
    }

    private static DataFrame fromTableOrient(JsonValue root, JsonOptions opt) {
        // pandas table: {schema: {fields:[{name,type},...]}, data:[...]}
        if (!root.isObject()) throw new JsonException("table orient expects object");
        Map<String, Column.DType> schema = new LinkedHashMap<>();
        List<String> names = new ArrayList<>();
        if (root.has("schema") && root.get("schema").isObject()) {
            JsonValue schemaNode = root.get("schema");
            if (schemaNode.has("fields") && schemaNode.get("fields").isArray()) {
                for (JsonValue f : schemaNode.get("fields").asArray()) {
                    if (!f.isObject()) continue;
                    String name = f.has("name") ? f.get("name").asString() : null;
                    if (name == null) continue;
                    String type = f.has("type") ? f.get("type").asString("string") : "string";
                    names.add(name);
                    schema.put(name, pandasTypeToDType(type));
                }
            }
        }
        JsonOptions.Builder b = JsonOptions.builder()
            .orient(JsonOptions.Orient.RECORDS)
            .inferSchema(schema.isEmpty())
            .strict(opt.strict())
            .flatten(opt.flatten())
            .flattenSeparator(opt.flattenSeparator())
            .keepNestedAsJson(opt.keepNestedAsJson())
            .maxRows(opt.maxRows())
            .skipRows(opt.skipRows())
            .writeNulls(opt.writeNulls())
            .pretty(opt.pretty())
            .charset(opt.charset());
        if (!schema.isEmpty()) b.schema(schema);
        if (!names.isEmpty()) b.columnNames(names);
        JsonOptions merged = b.build();

        JsonValue data = root.has("data") ? root.get("data") : JsonValue.array();
        return fromRecordsArray(data, merged, null);
    }

    // ---- record conversion ----

    private static Map<String, Object> valueToRecord(JsonValue value, JsonOptions opt) {
        if (value == null || value.isNull()) return null;
        if (value.isObject()) {
            if (opt.flatten()) {
                return flattenObject(value, "", opt.flattenSeparator(), opt);
            }
            Map<String, Object> rec = new LinkedHashMap<>();
            for (String k : value.keySet()) {
                JsonValue v = value.get(k);
                rec.put(k, convertCell(v, opt, k));
            }
            return rec;
        }
        // non-object line: wrap
        Map<String, Object> rec = new LinkedHashMap<>();
        rec.put("value", convertCell(value, opt, "value"));
        return rec;
    }

    private static Map<String, Object> flattenObject(JsonValue obj, String prefix, String sep, JsonOptions opt) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (!obj.isObject()) {
            String key = prefix.isEmpty() ? "value" : prefix;
            out.put(key, convertCell(obj, opt, key));
            return out;
        }
        for (String k : obj.keySet()) {
            String key = prefix.isEmpty() ? k : prefix + sep + k;
            JsonValue v = obj.get(k);
            if (v.isObject()) {
                out.putAll(flattenObject(v, key, sep, opt));
            } else if (v.isArray()) {
                if (opt.explodeArrays()) {
                    // store JSON string; explode handled later if needed
                    out.put(key, v.toString());
                } else if (opt.keepNestedAsJson()) {
                    out.put(key, v.toString());
                } else {
                    out.put(key, v.toJava());
                }
            } else {
                out.put(key, convertLeaf(v, opt, key));
            }
        }
        return out;
    }

    private static Object convertCell(JsonValue v, JsonOptions opt, String col) {
        if (v == null || v.isNull()) return null;
        if (v.isObject() || v.isArray()) {
            if (opt.keepNestedAsJson()) return v.toString();
            return v.toJava();
        }
        return convertLeaf(v, opt, col);
    }

    private static Object convertLeaf(JsonValue v, JsonOptions opt, String col) {
        if (v == null || v.isNull()) return null;
        switch (v.type()) {
            case BOOLEAN: return v.asBoolean();
            case STRING: {
                String s = v.asString();
                if (opt.isNullToken(s)) return null;
                // schema-forced conversion happens later
                return s;
            }
            case NUMBER: {
                if (v.isIntegralNumber()) {
                    try {
                        long lv = v.asLong();
                        if (lv >= Integer.MIN_VALUE && lv <= Integer.MAX_VALUE) return (int) lv;
                        return lv;
                    } catch (Exception e) {
                        return v.asDouble();
                    }
                }
                return v.asDouble();
            }
            default:
                return v.toJava();
        }
    }

    private static Map<String, Object> extractMeta(JsonValue root, JsonOptions opt) {
        Map<String, Object> meta = new LinkedHashMap<>();
        if (opt.metaPaths() == null) return meta;
        for (String p : opt.metaPaths()) {
            try {
                JsonValue v = JsonPath.get(root, p);
                String key = p.contains(".") ? p.substring(p.lastIndexOf('.') + 1) : p;
                key = key.replaceAll("\\[\\d+\\]", "");
                if (key.startsWith("[\"")) key = key.replaceAll("[\\[\\]\"]", "");
                meta.put(key.isEmpty() ? p : key, convertCell(v, opt, key));
            } catch (Exception ignored) {
                // missing meta is fine
            }
        }
        return meta;
    }

    // ---- build DataFrame ----

    private static DataFrame recordsToDataFrame(List<Map<String, Object>> records, JsonOptions opt) {
        if (records == null || records.isEmpty()) {
            DataFrame empty = DataFrame.create();
            if (opt.columnNames() != null) {
                for (String n : opt.columnNames()) {
                    Column.DType dt = dtypeFor(n, null, opt);
                    empty.addColumn(n, dt);
                }
            }
            return empty;
        }

        // union of keys, stable order: first-seen across records
        LinkedHashSet<String> keys = new LinkedHashSet<>();
        if (opt.columnNames() != null) keys.addAll(opt.columnNames());
        for (Map<String, Object> rec : records) {
            if (rec != null) keys.addAll(rec.keySet());
        }

        // infer dtypes
        Map<String, Column.DType> dtypes = new LinkedHashMap<>();
        int sample = Math.min(opt.inferSampleSize(), records.size());
        for (String k : keys) {
            if (opt.schema() != null && opt.schema().containsKey(k)) {
                dtypes.put(k, opt.schema().get(k));
            } else if (!opt.inferSchema()) {
                dtypes.put(k, Column.DType.STRING);
            } else {
                dtypes.put(k, inferColumnType(records, k, sample, opt));
            }
        }

        DataFrame df = DataFrame.create();
        for (String k : keys) {
            df.addColumn(k, dtypes.get(k));
        }

        for (Map<String, Object> rec : records) {
            Object[] row = new Object[keys.size()];
            int i = 0;
            for (String k : keys) {
                Object raw = rec == null ? null : rec.get(k);
                row[i++] = coerce(raw, dtypes.get(k), opt);
            }
            try {
                df.addRow(row);
            } catch (Exception e) {
                int idx = df.addEmptyRow();
                int c = 0;
                for (String k : keys) {
                    df.set(idx, k, row[c++]);
                }
            }
        }
        return df;
    }

    private static DataFrame columnsToDataFrame(Map<String, List<Object>> cols, JsonOptions opt) {
        DataFrame df = DataFrame.create();
        int rows = 0;
        for (List<Object> data : cols.values()) rows = Math.max(rows, data.size());

        for (Map.Entry<String, List<Object>> e : cols.entrySet()) {
            Column.DType dt = dtypeFor(e.getKey(), e.getValue(), opt);
            df.addColumn(e.getKey(), dt);
        }
        for (int r = 0; r < rows; r++) {
            if (opt.maxRows() >= 0 && r >= opt.maxRows()) break;
            Object[] row = new Object[cols.size()];
            int c = 0;
            for (Map.Entry<String, List<Object>> e : cols.entrySet()) {
                List<Object> data = e.getValue();
                Object raw = r < data.size() ? data.get(r) : null;
                row[c++] = coerce(raw, df.column(e.getKey()).dtype(), opt);
            }
            df.addRow(row);
        }
        return df;
    }

    private static Column.DType dtypeFor(String name, List<Object> sample, JsonOptions opt) {
        if (opt.schema() != null && opt.schema().containsKey(name)) return opt.schema().get(name);
        if (!opt.inferSchema() || sample == null) return Column.DType.STRING;
        Column.DType best = Column.DType.STRING;
        boolean canBool = true, canInt = true, canLong = true, canDouble = true, canDate = true, canDt = true;
        int nonNull = 0;
        for (Object v : sample) {
            if (v == null) continue;
            nonNull++;
            if (!(v instanceof Boolean) && !isBoolString(v)) canBool = false;
            if (!(v instanceof Integer) && !(v instanceof Short) && !(v instanceof Byte) && !isIntString(v))
                canInt = false;
            if (!(v instanceof Long) && !(v instanceof Integer) && !(v instanceof Short) && !(v instanceof Byte)
                && !isLongString(v)) canLong = false;
            if (!(v instanceof Number) && !isDoubleString(v)) canDouble = false;
            if (!(v instanceof LocalDate) && !isDateString(v)) canDate = false;
            if (!(v instanceof LocalDateTime) && !isDateTimeString(v)) canDt = false;
        }
        if (nonNull == 0) return Column.DType.STRING;
        if (canBool) best = Column.DType.BOOLEAN;
        else if (canInt) best = Column.DType.INT32;
        else if (canLong) best = Column.DType.INT64;
        else if (canDouble) best = Column.DType.FLOAT64;
        else if (canDt) best = Column.DType.DATETIME;
        else if (canDate) best = Column.DType.DATE;
        else best = Column.DType.STRING;
        return best;
    }

    private static Column.DType inferColumnType(List<Map<String, Object>> records, String key,
                                                 int sample, JsonOptions opt) {
        List<Object> vals = new ArrayList<>(sample);
        for (int i = 0; i < sample; i++) {
            Map<String, Object> rec = records.get(i);
            vals.add(rec == null ? null : rec.get(key));
        }
        return dtypeFor(key, vals, opt);
    }

    private static Object coerce(Object raw, Column.DType dtype, JsonOptions opt) {
        if (raw == null) return null;
        if (raw instanceof String && opt.isNullToken((String) raw)) return null;
        try {
            switch (dtype) {
                case BOOLEAN:
                    if (raw instanceof Boolean) return raw;
                    if (raw instanceof Number) return ((Number) raw).doubleValue() != 0;
                    String bs = String.valueOf(raw).trim();
                    if ("1".equals(bs) || "true".equalsIgnoreCase(bs) || "yes".equalsIgnoreCase(bs)) return true;
                    if ("0".equals(bs) || "false".equalsIgnoreCase(bs) || "no".equalsIgnoreCase(bs)) return false;
                    return Boolean.parseBoolean(bs);
                case INT32:
                    if (raw instanceof Integer) return raw;
                    if (raw instanceof Number) return ((Number) raw).intValue();
                    return Integer.parseInt(String.valueOf(raw).trim());
                case INT64:
                    if (raw instanceof Long) return raw;
                    if (raw instanceof Number) return ((Number) raw).longValue();
                    return Long.parseLong(String.valueOf(raw).trim());
                case FLOAT32:
                    if (raw instanceof Float) return raw;
                    if (raw instanceof Number) return ((Number) raw).floatValue();
                    return Float.parseFloat(String.valueOf(raw).trim());
                case FLOAT64:
                    if (raw instanceof Double) return raw;
                    if (raw instanceof Number) return ((Number) raw).doubleValue();
                    return Double.parseDouble(String.valueOf(raw).trim());
                case DATE:
                    if (raw instanceof LocalDate) return raw;
                    if (raw instanceof Number) {
                        long epoch = ((Number) raw).longValue();
                        if (opt.dateUnit() == 1) {
                            return Instant.ofEpochMilli(epoch).atZone(ZoneOffset.UTC).toLocalDate();
                        }
                        return Instant.ofEpochSecond(epoch).atZone(ZoneOffset.UTC).toLocalDate();
                    }
                    return parseDate(String.valueOf(raw));
                case DATETIME:
                    if (raw instanceof LocalDateTime) return raw;
                    if (raw instanceof Number) {
                        long epoch = ((Number) raw).longValue();
                        if (opt.dateFormat() == JsonOptions.DateFormat.EPOCH_SECONDS || opt.dateUnit() == 0) {
                            return LocalDateTime.ofInstant(Instant.ofEpochSecond(epoch), ZoneOffset.UTC);
                        }
                        return LocalDateTime.ofInstant(Instant.ofEpochMilli(epoch), ZoneOffset.UTC);
                    }
                    return parseDateTime(String.valueOf(raw));
                case VECTOR:
                case EMBEDDING:
                    return ComplexCellCodec.coerceComplex(raw, Column.DType.VECTOR);
                case LIST:
                    return ComplexCellCodec.coerceComplex(raw, Column.DType.LIST);
                case MAP:
                case STRUCT:
                    return ComplexCellCodec.coerceComplex(raw, dtype);
                case JSON:
                    if (raw instanceof Map || raw instanceof List || raw instanceof JsonValue) {
                        return raw instanceof JsonValue ? ((JsonValue) raw).toJava() : raw;
                    }
                    if (raw instanceof String) {
                        try { return JsonParser.parse((String) raw).toJava(); }
                        catch (Exception e) { return raw; }
                    }
                    return raw;
                case STRING:
                default:
                    if (raw instanceof String) return raw;
                    if (raw instanceof Map || raw instanceof List || raw.getClass().isArray()) {
                        // Keep structured values when present; stringify only for pure STRING columns
                        if (dtype == Column.DType.STRING) {
                            return Json.stringify(JsonValue.fromJava(raw));
                        }
                        return raw;
                    }
                    return String.valueOf(raw);
            }
        } catch (Exception e) {
            if (opt.strict()) {
                throw new JsonException("Cannot coerce value '" + raw + "' to " + dtype, e);
            }
            return raw instanceof String ? raw : String.valueOf(raw);
        }
    }

    // ---- type helpers ----

    private static Column.DType inferDType(Object v) {
        if (v == null) return Column.DType.STRING;
        if (v instanceof Boolean) return Column.DType.BOOLEAN;
        if (v instanceof Integer || v instanceof Short || v instanceof Byte) return Column.DType.INT32;
        if (v instanceof Long) return Column.DType.INT64;
        if (v instanceof Float) return Column.DType.FLOAT32;
        if (v instanceof Double || v instanceof Number) return Column.DType.FLOAT64;
        if (v instanceof LocalDate) return Column.DType.DATE;
        if (v instanceof LocalDateTime) return Column.DType.DATETIME;
        if (v instanceof float[] || v instanceof double[]) return Column.DType.VECTOR;
        if (v instanceof int[] || v instanceof long[] || v instanceof boolean[]) return Column.DType.LIST;
        if (v instanceof Map) return Column.DType.MAP;
        if (v instanceof List) return ComplexCellCodec.inferComplex(v);
        if (v instanceof JsonValue) return ComplexCellCodec.inferComplex(v);
        return Column.DType.STRING;
    }

    private static Column.DType pandasTypeToDType(String type) {
        if (type == null) return Column.DType.STRING;
        String t = type.toLowerCase(Locale.ROOT);
        switch (t) {
            case "integer": case "int": case "int64": case "int32": return Column.DType.INT64;
            case "number": case "float": case "float64": case "double": return Column.DType.FLOAT64;
            case "boolean": case "bool": return Column.DType.BOOLEAN;
            case "datetime": case "datetime64": case "timestamp": return Column.DType.DATETIME;
            case "date": return Column.DType.DATE;
            case "string": case "str": case "unicode": default: return Column.DType.STRING;
        }
    }

    private static boolean isBoolString(Object v) {
        if (!(v instanceof String)) return false;
        String s = ((String) v).trim();
        return "true".equalsIgnoreCase(s) || "false".equalsIgnoreCase(s)
            || "yes".equalsIgnoreCase(s) || "no".equalsIgnoreCase(s)
            || "1".equals(s) || "0".equals(s);
    }

    private static boolean isIntString(Object v) {
        if (!(v instanceof String)) return v instanceof Integer || v instanceof Short || v instanceof Byte;
        try { Integer.parseInt(((String) v).trim()); return true; } catch (Exception e) { return false; }
    }

    private static boolean isLongString(Object v) {
        if (!(v instanceof String)) return v instanceof Number && !(v instanceof Double) && !(v instanceof Float);
        try { Long.parseLong(((String) v).trim()); return true; } catch (Exception e) { return false; }
    }

    private static boolean isDoubleString(Object v) {
        if (!(v instanceof String)) return v instanceof Number;
        try { Double.parseDouble(((String) v).trim()); return true; } catch (Exception e) { return false; }
    }

    private static final DateTimeFormatter[] DATE_FMTS = {
        DateTimeFormatter.ISO_LOCAL_DATE,
        DateTimeFormatter.ofPattern("yyyy/MM/dd"),
        DateTimeFormatter.ofPattern("MM/dd/yyyy")
    };
    private static final DateTimeFormatter[] DATETIME_FMTS = {
        DateTimeFormatter.ISO_LOCAL_DATE_TIME,
        DateTimeFormatter.ISO_OFFSET_DATE_TIME,
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
        DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss")
    };

    private static boolean isDateString(Object v) {
        if (!(v instanceof String)) return false;
        String s = ((String) v).trim();
        for (DateTimeFormatter f : DATE_FMTS) {
            try { LocalDate.parse(s, f); return true; } catch (DateTimeParseException ignored) {}
        }
        return false;
    }

    private static boolean isDateTimeString(Object v) {
        if (!(v instanceof String)) return false;
        String s = ((String) v).trim();
        for (DateTimeFormatter f : DATETIME_FMTS) {
            try { LocalDateTime.parse(s, f); return true; } catch (DateTimeParseException ignored) {}
            try {
                return java.time.OffsetDateTime.parse(s, f) != null;
            } catch (Exception ignored) {}
        }
        return false;
    }

    private static LocalDate parseDate(String s) {
        for (DateTimeFormatter f : DATE_FMTS) {
            try { return LocalDate.parse(s.trim(), f); } catch (DateTimeParseException ignored) {}
        }
        throw new JsonException("Cannot parse date: " + s);
    }

    private static LocalDateTime parseDateTime(String s) {
        String t = s.trim();
        for (DateTimeFormatter f : DATETIME_FMTS) {
            try { return LocalDateTime.parse(t, f); } catch (DateTimeParseException ignored) {}
            try {
                return java.time.OffsetDateTime.parse(t).toLocalDateTime();
            } catch (Exception ignored) {}
        }
        // date only → start of day
        try { return parseDate(t).atStartOfDay(); } catch (Exception ignored) {}
        throw new JsonException("Cannot parse datetime: " + s);
    }

    private static JsonOptions normalizeLines(JsonOptions options) {
        JsonOptions opt = options == null ? JsonOptions.lines() : options;
        if (opt.orient() != JsonOptions.Orient.LINES) {
            return JsonOptions.builder()
                .orient(JsonOptions.Orient.LINES)
                .flatten(opt.flatten())
                .flattenSeparator(opt.flattenSeparator())
                .inferSchema(opt.inferSchema())
                .inferSampleSize(opt.inferSampleSize())
                .strict(opt.strict())
                .maxRows(opt.maxRows())
                .skipRows(opt.skipRows())
                .columnNames(opt.columnNames())
                .schema(opt.schema())
                .keepNestedAsJson(opt.keepNestedAsJson())
                .explodeArrays(opt.explodeArrays())
                .pretty(opt.pretty())
                .writeNulls(opt.writeNulls())
                .charset(opt.charset())
                .stripBom(opt.stripBom())
                .allowComments(opt.allowComments())
                .allowTrailingCommas(opt.allowTrailingCommas())
                .allowMultiLineJsonl(opt.allowMultiLineJsonl())
                .linesCommentPrefix(opt.linesCommentPrefix())
                .build();
        }
        return opt;
    }

    /** Sniff first non-empty line: if whole file is one JSON value starting with [ or {, not lines. */
    private static boolean isProbablyJsonl(Path path, JsonOptions opt) {
        if (opt.orient() == JsonOptions.Orient.LINES) return true;
        if (opt.orient() != JsonOptions.Orient.RECORDS) return false;
        String name = path.getFileName() == null ? "" : path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".jsonl") || name.endsWith(".ndjson") || name.endsWith(".jsonlines")) return true;
        // peek: if multiple top-level values on separate lines, treat as jsonl
        try (BufferedReader br = Files.newBufferedReader(path, opt.charset())) {
            int nonEmpty = 0;
            boolean sawObjectLine = false;
            String line;
            while ((line = br.readLine()) != null && nonEmpty < 5) {
                String t = line.trim();
                if (t.isEmpty()) continue;
                nonEmpty++;
                if (t.startsWith("{") && t.endsWith("}")) sawObjectLine = true;
                if (nonEmpty == 1 && (t.startsWith("[") || (t.startsWith("{") && !t.endsWith("}")))) {
                    // classic JSON array / multi-line object
                    return false;
                }
            }
            return sawObjectLine && nonEmpty >= 2;
        } catch (Exception e) {
            return false;
        }
    }
}
