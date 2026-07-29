package org.bytedeco.pytorch.utils.kafka;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Offline Kafka dump I/O — no broker required.
 *
 * <p>Formats used across Flink / Spark / console-consumer / internal replay tools:
 * <ul>
 *   <li><b>JSONL</b> — one JSON object per line (default dump)</li>
 *   <li><b>JSON array</b> — small fixtures</li>
 *   <li><b>CSV</b> — flat columns (optional metadata)</li>
 *   <li><b>Console</b> — {@code key\tvalue} or {@code partition-offset\tkey\tvalue}</li>
 *   <li><b>Binary</b> — length-prefixed records ({@code int32 len + bytes})</li>
 * </ul>
 *
 * <pre>{@code
 * KafkaFile.writeJsonl(df, Path.of("feature_dump.jsonl"));
 * DataFrame back = KafkaFile.readJsonl(Path.of("feature_dump.jsonl"));
 * }</pre>
 */
public final class KafkaFile {

    private KafkaFile() {}

    // ── JSONL ────────────────────────────────────────────────────────────────

    public static void writeJsonl(DataFrame df, Path path) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(path, "path");
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                for (Map<String, Object> row : df.toRecords()) {
                    w.write(Json.encode(row));
                    w.newLine();
                }
            }
        } catch (IOException e) {
            throw new KafkaException("writeJsonl failed: " + e.getMessage(), e, "writeJsonl", null);
        }
    }

    public static void writeJsonl(List<KafkaRecord> records, Path path, boolean includeMetadata) {
        Objects.requireNonNull(path, "path");
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                if (records == null) return;
                for (KafkaRecord r : records) {
                    w.write(Json.encode(r.toRowMap(includeMetadata)));
                    w.newLine();
                }
            }
        } catch (IOException e) {
            throw new KafkaException("writeJsonl failed: " + e.getMessage(), e, "writeJsonl", null);
        }
    }

    public static DataFrame readJsonl(Path path) {
        Objects.requireNonNull(path, "path");
        List<Map<String, Object>> rows = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            long lineNo = 0;
            while ((line = r.readLine()) != null) {
                lineNo++;
                String t = line.trim();
                if (t.isEmpty() || t.startsWith("#")) continue;
                try {
                    Object v = Json.decode(t);
                    if (v instanceof Map<?, ?> m) {
                        Map<String, Object> row = new LinkedHashMap<>();
                        for (Map.Entry<?, ?> e : m.entrySet()) {
                            if (e.getKey() != null) row.put(String.valueOf(e.getKey()), e.getValue());
                        }
                        rows.add(row);
                    } else {
                        Map<String, Object> wrap = new LinkedHashMap<>();
                        wrap.put("value", v);
                        rows.add(wrap);
                    }
                } catch (Exception ex) {
                    throw new KafkaException("readJsonl parse error at line " + lineNo + ": " + ex.getMessage(),
                            ex, "readJsonl", null);
                }
            }
        } catch (KafkaException e) {
            throw e;
        } catch (IOException e) {
            throw new KafkaException("readJsonl failed: " + e.getMessage(), e, "readJsonl", null);
        }
        return DataFrame.fromRecords(rows);
    }

    // ── JSON array ───────────────────────────────────────────────────────────

    public static void writeJsonArray(DataFrame df, Path path) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(path, "path");
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            Files.writeString(path, Json.encode(df.toRecords()), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new KafkaException("writeJsonArray failed: " + e.getMessage(), e, "writeJsonArray", null);
        }
    }

    @SuppressWarnings("unchecked")
    public static DataFrame readJsonArray(Path path) {
        Objects.requireNonNull(path, "path");
        try {
            String text = Files.readString(path, StandardCharsets.UTF_8).trim();
            if (text.isEmpty()) return DataFrame.create();
            Object v = Json.decode(text);
            if (v instanceof List<?> list) {
                List<Map<String, Object>> rows = new ArrayList<>(list.size());
                for (Object o : list) {
                    if (o instanceof Map<?, ?> m) {
                        Map<String, Object> row = new LinkedHashMap<>();
                        for (Map.Entry<?, ?> e : m.entrySet()) {
                            if (e.getKey() != null) row.put(String.valueOf(e.getKey()), e.getValue());
                        }
                        rows.add(row);
                    } else {
                        Map<String, Object> wrap = new LinkedHashMap<>();
                        wrap.put("value", o);
                        rows.add(wrap);
                    }
                }
                return DataFrame.fromRecords(rows);
            }
            if (v instanceof Map<?, ?> m) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    if (e.getKey() != null) row.put(String.valueOf(e.getKey()), e.getValue());
                }
                return DataFrame.fromRecords(List.of(row));
            }
            throw new KafkaException("expected JSON array or object", null, "readJsonArray", null);
        } catch (KafkaException e) {
            throw e;
        } catch (Exception e) {
            throw new KafkaException("readJsonArray failed: " + e.getMessage(), e, "readJsonArray", null);
        }
    }

    // ── CSV ──────────────────────────────────────────────────────────────────

    public static void writeCsv(DataFrame df, Path path) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(path, "path");
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            List<String> cols = new ArrayList<>();
            for (int i = 0; i < df.columnCount(); i++) {
                cols.add(df.column(i).name());
            }
            try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                // header
                for (int i = 0; i < cols.size(); i++) {
                    if (i > 0) w.write(',');
                    w.write(csvEscape(cols.get(i)));
                }
                w.newLine();
                int n = df.rowCount();
                for (int r = 0; r < n; r++) {
                    for (int c = 0; c < cols.size(); c++) {
                        if (c > 0) w.write(',');
                        Object v = df.get(r, cols.get(c));
                        w.write(csvEscape(v == null ? "" : String.valueOf(v)));
                    }
                    w.newLine();
                }
            }
        } catch (IOException e) {
            throw new KafkaException("writeCsv failed: " + e.getMessage(), e, "writeCsv", null);
        }
    }

    public static DataFrame readCsv(Path path) {
        Objects.requireNonNull(path, "path");
        try (BufferedReader r = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = r.readLine();
            if (header == null || header.isBlank()) return DataFrame.create();
            List<String> cols = splitCsv(header);
            List<Map<String, Object>> rows = new ArrayList<>();
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isBlank()) continue;
                List<String> vals = splitCsv(line);
                Map<String, Object> row = new LinkedHashMap<>();
                for (int i = 0; i < cols.size(); i++) {
                    String v = i < vals.size() ? vals.get(i) : null;
                    row.put(cols.get(i), coerce(v));
                }
                rows.add(row);
            }
            return DataFrame.fromRecords(rows);
        } catch (IOException e) {
            throw new KafkaException("readCsv failed: " + e.getMessage(), e, "readCsv", null);
        }
    }

    // ── console-consumer style ───────────────────────────────────────────────

    /**
     * Write console format: {@code key\tjsonValue} per line.
     * If {@code withMeta}, prefix {@code partition-offset\t}.
     */
    public static void writeConsole(List<KafkaRecord> records, Path path, boolean withMeta) {
        Objects.requireNonNull(path, "path");
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                if (records == null) return;
                for (KafkaRecord rec : records) {
                    if (withMeta) {
                        w.write(String.valueOf(rec.partition() == null ? 0 : rec.partition()));
                        w.write('-');
                        w.write(String.valueOf(rec.offset() == null ? 0 : rec.offset()));
                        w.write('\t');
                    }
                    w.write(rec.key() == null ? "" : rec.key());
                    w.write('\t');
                    Object val = rec.value();
                    if (val instanceof String s) w.write(s);
                    else if (val instanceof byte[] b) w.write(new String(b, StandardCharsets.UTF_8));
                    else w.write(Json.encode(val));
                    w.newLine();
                }
            }
        } catch (IOException e) {
            throw new KafkaException("writeConsole failed: " + e.getMessage(), e, "writeConsole", null);
        }
    }

    public static void writeConsole(DataFrame df, Path path, String keyColumn) {
        Objects.requireNonNull(df, "df");
        List<KafkaRecord> records = new ArrayList<>(df.rowCount());
        for (Map<String, Object> row : df.toRecords()) {
            String key = null;
            if (keyColumn != null && row.get(keyColumn) != null) {
                key = String.valueOf(row.get(keyColumn));
            }
            records.add(KafkaRecord.builder().key(key).value(row).build());
        }
        writeConsole(records, path, false);
    }

    /**
     * Read console dump lines. Supports:
     * <ul>
     *   <li>{@code key\tvalue}</li>
     *   <li>{@code partition-offset\tkey\tvalue}</li>
     *   <li>value-only lines (no tab)</li>
     * </ul>
     */
    public static DataFrame readConsole(Path path) {
        Objects.requireNonNull(path, "path");
        List<Map<String, Object>> rows = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String key = null;
                String value;
                Integer partition = null;
                Long offset = null;
                String[] parts = line.split("\t", 3);
                if (parts.length == 1) {
                    value = parts[0];
                } else if (parts.length == 2) {
                    key = parts[0].isEmpty() ? null : parts[0];
                    value = parts[1];
                } else {
                    // partition-offset \t key \t value  OR key \t value with extra tabs in value
                    String metaOrKey = parts[0];
                    if (metaOrKey.matches("\\d+-\\d+")) {
                        int dash = metaOrKey.indexOf('-');
                        partition = Integer.parseInt(metaOrKey.substring(0, dash));
                        offset = Long.parseLong(metaOrKey.substring(dash + 1));
                        key = parts[1].isEmpty() ? null : parts[1];
                        value = parts[2];
                    } else {
                        key = metaOrKey.isEmpty() ? null : metaOrKey;
                        value = parts[1] + "\t" + parts[2];
                    }
                }
                Map<String, Object> row = KafkaSerde.decodeToMap(value, KafkaOptions.ValueFormat.JSON);
                if (key != null) row.putIfAbsent("__key", key);
                if (partition != null) row.put("__partition", partition.longValue());
                if (offset != null) row.put("__offset", offset);
                rows.add(row);
            }
        } catch (IOException e) {
            throw new KafkaException("readConsole failed: " + e.getMessage(), e, "readConsole", null);
        }
        return DataFrame.fromRecords(rows);
    }

    // ── binary length-prefixed ───────────────────────────────────────────────

    /**
     * Binary dump: for each record write {@code int32 BE keyLen + keyBytes + int32 BE valLen + valBytes}.
     * keyLen/valLen = -1 means null.
     */
    public static void writeBinary(List<KafkaRecord> records, Path path) {
        Objects.requireNonNull(path, "path");
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            try (DataOutputStream out = new DataOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(path)))) {
                if (records == null) return;
                for (KafkaRecord r : records) {
                    byte[] key = KafkaSerde.encodeKey(r.key());
                    byte[] val = KafkaSerde.encodeValue(r.value());
                    writeBytes(out, key);
                    writeBytes(out, val);
                }
            }
        } catch (IOException e) {
            throw new KafkaException("writeBinary failed: " + e.getMessage(), e, "writeBinary", null);
        }
    }

    public static void writeBinary(DataFrame df, Path path, String keyColumn) {
        Objects.requireNonNull(df, "df");
        List<KafkaRecord> records = new ArrayList<>(df.rowCount());
        for (Map<String, Object> row : df.toRecords()) {
            String key = null;
            if (keyColumn != null && row.get(keyColumn) != null) {
                key = String.valueOf(row.get(keyColumn));
            }
            records.add(KafkaRecord.builder().key(key).value(row).build());
        }
        writeBinary(records, path);
    }

    public static DataFrame readBinary(Path path) {
        Objects.requireNonNull(path, "path");
        List<Map<String, Object>> rows = new ArrayList<>();
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(path)))) {
            while (in.available() > 0 || peekHasMore(in)) {
                byte[] key = readBytes(in);
                byte[] val = readBytes(in);
                Map<String, Object> row = KafkaSerde.decodeToMap(val, KafkaOptions.ValueFormat.JSON);
                if (key != null) row.putIfAbsent("__key", KafkaSerde.decodeKey(key));
                rows.add(row);
            }
        } catch (IOException e) {
            // EOF is normal end
            if (!(e instanceof java.io.EOFException)) {
                throw new KafkaException("readBinary failed: " + e.getMessage(), e, "readBinary", null);
            }
        }
        return DataFrame.fromRecords(rows);
    }

    /** Auto-detect by extension: .jsonl, .json, .csv, .console/.txt, .bin/.dump */
    public static DataFrame read(Path path) {
        Objects.requireNonNull(path, "path");
        String name = path.getFileName().toString().toLowerCase();
        if (name.endsWith(".jsonl") || name.endsWith(".ndjson")) return readJsonl(path);
        if (name.endsWith(".json")) return readJsonArray(path);
        if (name.endsWith(".csv")) return readCsv(path);
        if (name.endsWith(".bin") || name.endsWith(".dump") || name.endsWith(".kfk")) return readBinary(path);
        if (name.endsWith(".console") || name.endsWith(".txt") || name.endsWith(".log")) return readConsole(path);
        // default JSONL
        return readJsonl(path);
    }

    public static void write(DataFrame df, Path path) {
        Objects.requireNonNull(path, "path");
        String name = path.getFileName().toString().toLowerCase();
        if (name.endsWith(".jsonl") || name.endsWith(".ndjson")) {
            writeJsonl(df, path);
        } else if (name.endsWith(".json")) {
            writeJsonArray(df, path);
        } else if (name.endsWith(".csv")) {
            writeCsv(df, path);
        } else if (name.endsWith(".bin") || name.endsWith(".dump") || name.endsWith(".kfk")) {
            writeBinary(df, path, null);
        } else if (name.endsWith(".console") || name.endsWith(".txt") || name.endsWith(".log")) {
            writeConsole(df, path, null);
        } else {
            writeJsonl(df, path);
        }
    }

    // ── internals ────────────────────────────────────────────────────────────

    private static void writeBytes(DataOutputStream out, byte[] data) throws IOException {
        if (data == null) {
            out.writeInt(-1);
        } else {
            out.writeInt(data.length);
            out.write(data);
        }
    }

    private static byte[] readBytes(DataInputStream in) throws IOException {
        int len = in.readInt();
        if (len < 0) return null;
        byte[] buf = new byte[len];
        in.readFully(buf);
        return buf;
    }

    private static boolean peekHasMore(DataInputStream in) {
        try {
            in.mark(1);
            int b = in.read();
            if (b < 0) return false;
            in.reset();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private static String csvEscape(String s) {
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0 && s.indexOf('\r') < 0) {
            return s;
        }
        return '"' + s.replace("\"", "\"\"") + '"';
    }

    private static List<String> splitCsv(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    cur.append(c);
                }
            } else if (c == '"') {
                inQuotes = true;
            } else if (c == ',') {
                out.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        out.add(cur.toString());
        return out;
    }

    private static Object coerce(String v) {
        if (v == null || v.isEmpty()) return v;
        try {
            if (v.indexOf('.') >= 0 || v.indexOf('e') >= 0 || v.indexOf('E') >= 0) {
                return Double.parseDouble(v);
            }
            long l = Long.parseLong(v);
            if (l >= Integer.MIN_VALUE && l <= Integer.MAX_VALUE) return (int) l;
            return l;
        } catch (NumberFormatException e) {
            if ("true".equalsIgnoreCase(v)) return Boolean.TRUE;
            if ("false".equalsIgnoreCase(v)) return Boolean.FALSE;
            return v;
        }
    }
}
