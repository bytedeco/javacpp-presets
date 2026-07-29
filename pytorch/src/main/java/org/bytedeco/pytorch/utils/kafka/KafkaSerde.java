package org.bytedeco.pytorch.utils.kafka;

import org.bytedeco.pytorch.utils.json.Json;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Encode / decode Kafka keys and values without Schema Registry.
 *
 * <p>Formats:
 * <ul>
 *   <li>{@link KafkaOptions.ValueFormat#JSON} — UTF-8 JSON object/array (default for DataFrame rows)</li>
 *   <li>{@link KafkaOptions.ValueFormat#STRING} — raw UTF-8 text</li>
 *   <li>{@link KafkaOptions.ValueFormat#BYTES} — opaque binary (Base64 when forced through text paths)</li>
 *   <li>{@link KafkaOptions.ValueFormat#CSV_ROW} — comma-joined values (order = column order)</li>
 * </ul>
 *
 * <p>Uses the project-local {@link Json} codec (no Jackson/Gson dependency on this path).
 */
public final class KafkaSerde {

    private KafkaSerde() {}

    // ── key ──────────────────────────────────────────────────────────────────

    public static byte[] encodeKey(String key) {
        if (key == null) return null;
        return key.getBytes(StandardCharsets.UTF_8);
    }

    public static String decodeKey(byte[] key) {
        if (key == null) return null;
        return new String(key, StandardCharsets.UTF_8);
    }

    // ── value ────────────────────────────────────────────────────────────────

    public static byte[] encodeValue(Object value) {
        return encodeValue(value, KafkaOptions.ValueFormat.JSON);
    }

    @SuppressWarnings("unchecked")
    public static byte[] encodeValue(Object value, KafkaOptions.ValueFormat format) {
        if (value == null) return null;
        if (value instanceof byte[] b) return b;
        KafkaOptions.ValueFormat fmt = format == null ? KafkaOptions.ValueFormat.JSON : format;
        return switch (fmt) {
            case BYTES -> {
                if (value instanceof byte[] bb) yield bb;
                if (value instanceof String s) {
                    // allow base64-tagged or raw
                    yield s.getBytes(StandardCharsets.UTF_8);
                }
                yield Json.encode(value).getBytes(StandardCharsets.UTF_8);
            }
            case STRING -> String.valueOf(value).getBytes(StandardCharsets.UTF_8);
            case CSV_ROW -> {
                if (value instanceof Map<?, ?> m) {
                    StringBuilder sb = new StringBuilder();
                    boolean first = true;
                    for (Object v : m.values()) {
                        if (!first) sb.append(',');
                        first = false;
                        sb.append(csvEscape(v == null ? "" : String.valueOf(v)));
                    }
                    yield sb.toString().getBytes(StandardCharsets.UTF_8);
                }
                if (value instanceof List<?> list) {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < list.size(); i++) {
                        if (i > 0) sb.append(',');
                        Object v = list.get(i);
                        sb.append(csvEscape(v == null ? "" : String.valueOf(v)));
                    }
                    yield sb.toString().getBytes(StandardCharsets.UTF_8);
                }
                yield String.valueOf(value).getBytes(StandardCharsets.UTF_8);
            }
            case JSON, JSONL_ROW -> {
                if (value instanceof String s) {
                    String t = s.trim();
                    // already JSON text
                    if ((t.startsWith("{") && t.endsWith("}"))
                            || (t.startsWith("[") && t.endsWith("]"))) {
                        yield s.getBytes(StandardCharsets.UTF_8);
                    }
                    yield Json.encode(s).getBytes(StandardCharsets.UTF_8);
                }
                yield Json.encode(value).getBytes(StandardCharsets.UTF_8);
            }
        };
    }

    public static Object decodeToObject(Object raw) {
        return decodeToObject(raw, KafkaOptions.ValueFormat.JSON);
    }

    public static Object decodeToObject(Object raw, KafkaOptions.ValueFormat format) {
        if (raw == null) return null;
        if (raw instanceof Map || raw instanceof List) return raw;
        KafkaOptions.ValueFormat fmt = format == null ? KafkaOptions.ValueFormat.JSON : format;
        String text;
        if (raw instanceof byte[] b) {
            if (fmt == KafkaOptions.ValueFormat.BYTES) return b;
            text = new String(b, StandardCharsets.UTF_8);
        } else {
            text = String.valueOf(raw);
        }
        return switch (fmt) {
            case BYTES -> text.getBytes(StandardCharsets.UTF_8);
            case STRING, CSV_ROW -> text;
            case JSON, JSONL_ROW -> {
                String t = text.trim();
                if (t.isEmpty()) yield null;
                try {
                    yield Json.decode(t);
                } catch (Exception e) {
                    // non-JSON payload → keep as string (common for console dumps)
                    yield text;
                }
            }
        };
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> decodeToMap(Object raw, KafkaOptions.ValueFormat format) {
        Object v = decodeToObject(raw, format);
        if (v == null) return new LinkedHashMap<>();
        if (v instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (e.getKey() != null) out.put(String.valueOf(e.getKey()), e.getValue());
            }
            return out;
        }
        Map<String, Object> wrap = new LinkedHashMap<>();
        wrap.put("value", v);
        return wrap;
    }

    public static String encodeHeadersJson(Map<String, String> headers) {
        if (headers == null || headers.isEmpty()) return "{}";
        return Json.encode(headers);
    }

    @SuppressWarnings("unchecked")
    public static Map<String, String> decodeHeadersJson(String json) {
        if (json == null || json.isBlank()) return Map.of();
        try {
            Object v = Json.decode(json);
            if (!(v instanceof Map<?, ?> m)) return Map.of();
            Map<String, String> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (e.getKey() != null) {
                    out.put(String.valueOf(e.getKey()),
                            e.getValue() == null ? null : String.valueOf(e.getValue()));
                }
            }
            return out;
        } catch (Exception e) {
            return Map.of();
        }
    }

    public static String toBase64(byte[] bytes) {
        if (bytes == null) return null;
        return Base64.getEncoder().encodeToString(bytes);
    }

    public static byte[] fromBase64(String b64) {
        if (b64 == null) return null;
        return Base64.getDecoder().decode(b64);
    }

    private static String csvEscape(String s) {
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0) return s;
        return '"' + s.replace("\"", "\"\"") + '"';
    }
}
