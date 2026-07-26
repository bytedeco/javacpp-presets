package org.bytedeco.pytorch.data.json;
import java.io.*;
import java.nio.file.Path;
import java.util.*;

/**
 * Façade for the pure-Java JSON module.
 *
 * <pre>
 *   JsonValue v = Json.parse("{\"a\":1}");
 *   long a = v.at("a").asLong();
 *   String s = Json.stringify(v, true);
 *   List&lt;JsonValue&gt; lines = Json.parseJsonl(path);
 * </pre>
 */
public final class Json {
    private Json() {}

    // ---- parse ----

    public static JsonValue parse(String text) {
        return JsonParser.parse(text);
    }

    public static JsonValue parse(String text, JsonReadOptions options) {
        return JsonParser.parse(text, options);
    }

    public static JsonValue parse(Reader reader) throws IOException {
        return JsonParser.parse(reader);
    }

    public static JsonValue parse(Reader reader, JsonReadOptions options) throws IOException {
        return JsonParser.parse(reader, options);
    }

    public static JsonValue parse(InputStream in) throws IOException {
        return JsonParser.parse(in);
    }

    public static JsonValue parse(InputStream in, JsonReadOptions options) throws IOException {
        return JsonParser.parse(in, options);
    }

    public static JsonValue parse(Path path) throws IOException {
        return JsonParser.parse(path);
    }

    public static JsonValue parse(Path path, JsonReadOptions options) throws IOException {
        return JsonParser.parse(path, options);
    }

    public static JsonValue parseFile(String path) throws IOException {
        return JsonParser.parseFile(path);
    }

    public static JsonValue parseFile(String path, JsonReadOptions options) throws IOException {
        return JsonParser.parseFile(path, options);
    }

    // ---- JSONL ----

    public static JsonValue parseJsonl(String path) throws IOException {
        return JsonParser.parseLinesFile(path);
    }

    public static JsonValue parseJsonl(String path, JsonReadOptions options) throws IOException {
        return JsonParser.parseLinesFile(path, options);
    }

    public static JsonValue parseJsonl(Path path) throws IOException {
        return JsonParser.parseLines(path, JsonReadOptions.defaults());
    }

    public static JsonValue parseJsonl(Path path, JsonReadOptions options) throws IOException {
        return JsonParser.parseLines(path, options);
    }

    public static JsonValue parseJsonl(Reader reader) throws IOException {
        return JsonParser.parseLines(reader);
    }

    public static JsonValue parseJsonl(Reader reader, JsonReadOptions options) throws IOException {
        return JsonParser.parseLines(reader, options);
    }

    public static void forEachJsonl(String path, JsonParser.JsonValueConsumer consumer) throws IOException {
        try (Reader r = new BufferedReader(new FileReader(path))) {
            JsonParser.parseLines(r, JsonReadOptions.defaults(), consumer);
        }
    }

    public static void forEachJsonl(String path, JsonReadOptions options,
                                    JsonParser.JsonValueConsumer consumer) throws IOException {
        try (Reader r = new InputStreamReader(
                new java.io.FileInputStream(path),
                options == null ? java.nio.charset.StandardCharsets.UTF_8 : options.charset())) {
            JsonParser.parseLines(r, options, consumer);
        }
    }

    // ---- stringify / write ----

    public static String stringify(JsonValue value) {
        return JsonWriter.toString(value);
    }

    public static String stringify(JsonValue value, boolean pretty) {
        return JsonWriter.toString(value, pretty ? JsonWriteOptions.prettyMode() : JsonWriteOptions.compact());
    }

    public static String stringify(JsonValue value, JsonWriteOptions options) {
        return JsonWriter.toString(value, options);
    }

    public static String stringify(Object javaValue) {
        return stringify(JsonValue.fromJava(javaValue));
    }

    public static String stringify(Object javaValue, boolean pretty) {
        return stringify(JsonValue.fromJava(javaValue), pretty);
    }

    public static void write(JsonValue value, String path) throws IOException {
        JsonWriter.write(value, path);
    }

    public static void write(JsonValue value, String path, JsonWriteOptions options) throws IOException {
        JsonWriter.write(value, path, options);
    }

    public static void write(JsonValue value, Path path) throws IOException {
        JsonWriter.write(value, path);
    }

    public static void write(JsonValue value, Path path, JsonWriteOptions options) throws IOException {
        JsonWriter.write(value, path, options);
    }

    public static void writeJsonl(Iterable<JsonValue> values, String path) throws IOException {
        JsonWriter.writeLines(values, Path.of(path));
    }

    public static void writeJsonl(Iterable<JsonValue> values, String path, JsonWriteOptions options)
            throws IOException {
        JsonWriter.writeLines(values, Path.of(path), options);
    }

    public static void writeJsonl(Iterable<JsonValue> values, Path path) throws IOException {
        JsonWriter.writeLines(values, path);
    }

    public static void writeJsonl(Iterable<JsonValue> values, Path path, JsonWriteOptions options)
            throws IOException {
        JsonWriter.writeLines(values, path, options);
    }

    // ---- builders ----

    public static JsonValue obj() { return JsonValue.object(); }

    public static JsonValue obj(Object... keyValues) {
        if (keyValues.length % 2 != 0) {
            throw new JsonException("obj() requires even number of key/value arguments");
        }
        JsonValue o = JsonValue.object();
        for (int i = 0; i < keyValues.length; i += 2) {
            String k = String.valueOf(keyValues[i]);
            o.put(k, JsonValue.fromJava(keyValues[i + 1]));
        }
        return o;
    }

    public static JsonValue arr(Object... items) {
        JsonValue a = JsonValue.array();
        if (items != null) for (Object o : items) a.add(JsonValue.fromJava(o));
        return a;
    }

    // ---- path / flatten ----

    public static JsonValue at(JsonValue root, String path) {
        return JsonPath.get(root, path);
    }

    public static Map<String, Object> flatten(JsonValue value) {
        return JsonPath.flatten(value);
    }

    // ---- validate ----

    public static boolean isValid(String text) {
        try {
            parse(text, JsonReadOptions.strictMode());
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public static boolean isValid(String text, JsonReadOptions options) {
        try {
            parse(text, options);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /** Merge two objects (shallow). Right overwrites left on key conflict. */
    public static JsonValue merge(JsonValue left, JsonValue right) {
        if (left == null || left.isNull()) return right == null ? JsonValue.NULL : right;
        if (right == null || right.isNull()) return left;
        if (!left.isObject() || !right.isObject()) {
            throw new JsonException("merge requires two objects");
        }
        JsonValue out = JsonValue.object();
        for (String k : left.keySet()) out.put(k, left.get(k));
        for (String k : right.keySet()) out.put(k, right.get(k));
        return out;
    }

    /** Deep merge objects; arrays are replaced (not concatenated). */
    public static JsonValue deepMerge(JsonValue left, JsonValue right) {
        if (left == null || left.isNull()) return right == null ? JsonValue.NULL : right;
        if (right == null || right.isNull()) return left;
        if (!left.isObject() || !right.isObject()) return right;
        JsonValue out = JsonValue.object();
        Set<String> keys = new LinkedHashSet<>();
        keys.addAll(left.keySet());
        keys.addAll(right.keySet());
        for (String k : keys) {
            if (left.has(k) && right.has(k)
                && left.get(k).isObject() && right.get(k).isObject()) {
                out.put(k, deepMerge(left.get(k), right.get(k)));
            } else if (right.has(k)) {
                out.put(k, right.get(k));
            } else {
                out.put(k, left.get(k));
            }
        }
        return out;
    }
}
