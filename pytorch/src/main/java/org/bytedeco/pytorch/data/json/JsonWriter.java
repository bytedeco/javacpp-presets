package org.bytedeco.pytorch.data.json;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * RFC 8259 JSON writer with pretty-print, HTML escaping, key ordering, JSONL.
 */
public final class JsonWriter {
    private JsonWriter() {}

    // ---- string ----

    public static String toString(JsonValue value) {
        return toString(value, JsonWriteOptions.compact());
    }

    public static String toString(JsonValue value, JsonWriteOptions options) {
        StringBuilder sb = new StringBuilder(256);
        try {
            write(value, sb, options == null ? JsonWriteOptions.compact() : options, 0);
        } catch (IOException e) {
            throw new JsonException("write failed", e);
        }
        return sb.toString();
    }

    public static String toPrettyString(JsonValue value) {
        return toString(value, JsonWriteOptions.prettyMode());
    }

    // ---- files / streams ----

    public static void write(JsonValue value, Path path) throws IOException {
        write(value, path, JsonWriteOptions.defaults());
    }

    public static void write(JsonValue value, Path path, JsonWriteOptions options) throws IOException {
        JsonWriteOptions opt = options == null ? JsonWriteOptions.defaults() : options;
        try (OutputStream out = Files.newOutputStream(path)) {
            write(value, out, opt);
        }
    }

    public static void write(JsonValue value, String path) throws IOException {
        write(value, Path.of(path), JsonWriteOptions.defaults());
    }

    public static void write(JsonValue value, String path, JsonWriteOptions options) throws IOException {
        write(value, Path.of(path), options);
    }

    public static void write(JsonValue value, OutputStream out, JsonWriteOptions options) throws IOException {
        JsonWriteOptions opt = options == null ? JsonWriteOptions.defaults() : options;
        Charset cs = opt.charset();
        if (opt.writeBom() && cs.equals(java.nio.charset.StandardCharsets.UTF_8)) {
            out.write(new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
        }
        try (Writer w = new BufferedWriter(new OutputStreamWriter(out, cs))) {
            write(value, w, opt);
        }
    }

    public static void write(JsonValue value, Writer writer, JsonWriteOptions options) throws IOException {
        JsonWriteOptions opt = options == null ? JsonWriteOptions.defaults() : options;
        write(value, writer, opt, 0);
        writer.flush();
    }

    // ---- JSONL ----

    public static void writeLines(Iterable<JsonValue> values, Path path) throws IOException {
        writeLines(values, path, JsonWriteOptions.compact());
    }

    public static void writeLines(Iterable<JsonValue> values, Path path, JsonWriteOptions options)
            throws IOException {
        JsonWriteOptions opt = options == null ? JsonWriteOptions.compact() : options;
        // JSONL is always compact one-value-per-line
        JsonWriteOptions lineOpt = JsonWriteOptions.builder()
            .charset(opt.charset())
            .pretty(false)
            .escapeNonAscii(opt.escapeNonAscii())
            .escapeHtml(opt.escapeHtml())
            .escapeSolidus(opt.escapeSolidus())
            .orderKeys(opt.orderKeys())
            .nullHandling(opt.nullHandling())
            .nanAsNull(opt.nanAsNull())
            .maxDepth(opt.maxDepth())
            .writeBom(opt.writeBom())
            .lineSeparator(opt.lineSeparator())
            .build();
        try (OutputStream out = Files.newOutputStream(path);
             Writer w = new BufferedWriter(new OutputStreamWriter(out, lineOpt.charset()))) {
            if (lineOpt.writeBom() && lineOpt.charset().equals(java.nio.charset.StandardCharsets.UTF_8)) {
                out.write(new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
            }
            writeLines(values, w, lineOpt);
        }
    }

    public static void writeLines(Iterable<JsonValue> values, Writer writer, JsonWriteOptions options)
            throws IOException {
        JsonWriteOptions opt = options == null ? JsonWriteOptions.compact() : options;
        String sep = opt.lineSeparator() == null ? "\n" : opt.lineSeparator();
        boolean first = true;
        for (JsonValue v : values) {
            if (!first) writer.write(sep);
            first = false;
            write(v == null ? JsonValue.NULL : v, writer, opt, 0);
        }
        if (!first) writer.write(sep);
        writer.flush();
    }

    // ---- core writer ----

    private static void write(JsonValue value, Appendable out, JsonWriteOptions opt, int depth)
            throws IOException {
        if (depth > opt.maxDepth()) {
            throw new JsonException("Nesting depth exceeds maxDepth=" + opt.maxDepth());
        }
        if (value == null || value.isNull()) {
            out.append("null");
            return;
        }
        switch (value.type()) {
            case BOOLEAN:
                out.append(value.asBoolean() ? "true" : "false");
                break;
            case NUMBER: {
                String lex = value.numberLex();
                if ("NaN".equals(lex) || "Infinity".equals(lex) || "-Infinity".equals(lex)) {
                    if (opt.nanAsNull()) out.append("null");
                    else throw new JsonException("Cannot serialize non-finite number: " + lex);
                } else {
                    out.append(lex);
                }
                break;
            }
            case STRING:
                writeString(value.asString(), out, opt);
                break;
            case ARRAY:
                writeArray(value, out, opt, depth);
                break;
            case OBJECT:
                writeObject(value, out, opt, depth);
                break;
            default:
                out.append("null");
        }
    }

    private static void writeArray(JsonValue arr, Appendable out, JsonWriteOptions opt, int depth)
            throws IOException {
        out.append('[');
        int n = arr.size();
        if (n == 0) {
            out.append(']');
            return;
        }
        boolean pretty = opt.pretty();
        String ind = pretty ? indentOf(opt, depth + 1) : "";
        String ind0 = pretty ? indentOf(opt, depth) : "";
        String nl = pretty ? opt.lineSeparator() : "";
        for (int i = 0; i < n; i++) {
            if (i > 0) out.append(',');
            if (pretty) {
                out.append(nl).append(ind);
            }
            write(arr.get(i), out, opt, depth + 1);
        }
        if (pretty) out.append(nl).append(ind0);
        out.append(']');
    }

    private static void writeObject(JsonValue obj, Appendable out, JsonWriteOptions opt, int depth)
            throws IOException {
        out.append('{');
        Collection<String> keys = obj.keySet();
        if (opt.orderKeys()) {
            List<String> sorted = new ArrayList<>(keys);
            Collections.sort(sorted);
            keys = sorted;
        }
        boolean pretty = opt.pretty();
        String ind = pretty ? indentOf(opt, depth + 1) : "";
        String ind0 = pretty ? indentOf(opt, depth) : "";
        String nl = pretty ? opt.lineSeparator() : "";
        boolean first = true;
        for (String key : keys) {
            JsonValue v = obj.get(key);
            if (v == null || v.isNull()) {
                if (opt.nullHandling() == JsonWriteOptions.NullHandling.OMIT) continue;
            }
            if (!first) out.append(',');
            first = false;
            if (pretty) out.append(nl).append(ind);
            writeString(key, out, opt);
            out.append(pretty ? ": " : ":");
            write(v == null ? JsonValue.NULL : v, out, opt, depth + 1);
        }
        if (!first && pretty) out.append(nl).append(ind0);
        out.append('}');
    }

    private static String indentOf(JsonWriteOptions opt, int depth) {
        String unit = opt.indent() == null ? "  " : opt.indent();
        if (depth <= 0 || unit.isEmpty()) return "";
        StringBuilder sb = new StringBuilder(unit.length() * depth);
        for (int i = 0; i < depth; i++) sb.append(unit);
        return sb.toString();
    }

    static void writeString(String s, Appendable out, JsonWriteOptions opt) throws IOException {
        out.append('"');
        if (s == null) {
            out.append('"');
            return;
        }
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            switch (cp) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\b': out.append("\\b"); break;
                case '\f': out.append("\\f"); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                case '/':
                    if (opt.escapeSolidus()) out.append("\\/");
                    else out.append('/');
                    break;
                case '<':
                    if (opt.escapeHtml()) out.append("\\u003c");
                    else out.append('<');
                    break;
                case '>':
                    if (opt.escapeHtml()) out.append("\\u003e");
                    else out.append('>');
                    break;
                case '&':
                    if (opt.escapeHtml()) out.append("\\u0026");
                    else out.append('&');
                    break;
                default:
                    if (cp < 0x20 || (opt.escapeNonAscii() && cp > 0x7F)
                        || (cp >= 0x7F && cp <= 0x9F)) {
                        if (cp <= 0xFFFF) {
                            out.append(String.format("\\u%04x", cp));
                        } else {
                            // surrogate pair
                            char[] pair = Character.toChars(cp);
                            out.append(String.format("\\u%04x\\u%04x", (int) pair[0], (int) pair[1]));
                        }
                    } else if (cp <= 0xFFFF) {
                        out.append((char) cp);
                    } else {
                        out.append(new String(Character.toChars(cp)));
                    }
            }
        }
        out.append('"');
    }
}
