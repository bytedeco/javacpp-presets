/*
 * Serialize feature values for Redis hash fields / SQLite TEXT columns.
 * Format: type-tagged strings so round-trip preserves numbers, lists, embeddings.
 */
package org.bytedeco.pytorch.feature.store;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Value codec shared by Redis / SQLite online adapters. */
public final class FeatureValueCodec {

    private FeatureValueCodec() {}

    public static String encode(Object v) {
        if (v == null) return "null:";
        if (v instanceof Boolean) return "b:" + ((Boolean) v ? "1" : "0");
        if (v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte) {
            return "i:" + ((Number) v).longValue();
        }
        if (v instanceof Double || v instanceof Float) {
            return "f:" + ((Number) v).doubleValue();
        }
        if (v instanceof float[]) {
            return "e:" + Base64.getEncoder().encodeToString(floatsToBytes((float[]) v));
        }
        if (v instanceof double[]) {
            double[] d = (double[]) v;
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return "e:" + Base64.getEncoder().encodeToString(floatsToBytes(f));
        }
        if (v instanceof long[]) {
            return "l:" + joinLongs((long[]) v);
        }
        if (v instanceof int[]) {
            int[] a = (int[]) v;
            long[] l = new long[a.length];
            for (int i = 0; i < a.length; i++) l[i] = a[i];
            return "l:" + joinLongs(l);
        }
        if (v instanceof List) {
            // encode as JSON-ish array of numbers/strings
            return "j:" + listToJson((List<?>) v);
        }
        if (v instanceof Map) {
            return "m:" + mapToJson((Map<?, ?>) v);
        }
        return "s:" + escape(String.valueOf(v));
    }

    public static Object decode(String raw) {
        if (raw == null) return null;
        if (raw.equals("null:") || raw.equals("null")) return null;
        int colon = raw.indexOf(':');
        if (colon < 0) return raw;
        String tag = raw.substring(0, colon);
        String body = raw.substring(colon + 1);
        switch (tag) {
            case "b":
                return "1".equals(body) || "true".equalsIgnoreCase(body);
            case "i":
                try { return Long.parseLong(body); } catch (NumberFormatException e) { return 0L; }
            case "f":
                try { return Double.parseDouble(body); } catch (NumberFormatException e) { return 0.0; }
            case "e":
                try {
                    return bytesToFloats(Base64.getDecoder().decode(body));
                } catch (Exception e) {
                    return new float[0];
                }
            case "l":
                return parseLongs(body);
            case "j":
                return parseJsonArray(body);
            case "m":
                return parseJsonObject(body);
            case "s":
                return unescape(body);
            default:
                return raw;
        }
    }

    public static Map<String, String> encodeMap(Map<String, Object> values) {
        Map<String, String> out = new LinkedHashMap<>();
        if (values == null) return out;
        for (Map.Entry<String, Object> e : values.entrySet()) {
            out.put(e.getKey(), encode(e.getValue()));
        }
        return out;
    }

    public static Map<String, Object> decodeMap(Map<String, String> encoded) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (encoded == null) return out;
        for (Map.Entry<String, String> e : encoded.entrySet()) {
            // skip internal meta fields starting with _
            out.put(e.getKey(), decode(e.getValue()));
        }
        return out;
    }

    public static byte[] floatsToBytes(float[] v) {
        if (v == null) return new byte[0];
        ByteBuffer buf = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) buf.putFloat(x);
        return buf.array();
    }

    public static float[] bytesToFloats(byte[] bytes) {
        if (bytes == null || bytes.length < 4) return new float[0];
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[bytes.length / 4];
        for (int i = 0; i < out.length; i++) out[i] = buf.getFloat();
        return out;
    }

    private static String joinLongs(long[] a) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(a[i]);
        }
        return sb.toString();
    }

    private static long[] parseLongs(String body) {
        if (body == null || body.isEmpty()) return new long[0];
        String[] parts = body.split(",");
        long[] out = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try { out[i] = Long.parseLong(parts[i].trim()); }
            catch (NumberFormatException e) { out[i] = 0L; }
        }
        return out;
    }

    private static String listToJson(List<?> list) {
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for (Object o : list) {
            if (!first) sb.append(',');
            first = false;
            if (o == null) sb.append("null");
            else if (o instanceof Number || o instanceof Boolean) sb.append(o);
            else sb.append('"').append(escape(String.valueOf(o))).append('"');
        }
        return sb.append(']').toString();
    }

    private static String mapToJson(Map<?, ?> map) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<?, ?> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(escape(String.valueOf(e.getKey()))).append("\":");
            Object v = e.getValue();
            if (v == null) sb.append("null");
            else if (v instanceof Number || v instanceof Boolean) sb.append(v);
            else sb.append('"').append(escape(String.valueOf(v))).append('"');
        }
        return sb.append('}').toString();
    }

    private static List<Object> parseJsonArray(String body) {
        List<Object> out = new ArrayList<>();
        if (body == null) return out;
        String s = body.trim();
        if (s.startsWith("[")) s = s.substring(1);
        if (s.endsWith("]")) s = s.substring(0, s.length() - 1);
        if (s.isBlank()) return out;
        for (String part : splitTop(s, ',')) {
            out.add(parseScalar(part.trim()));
        }
        return out;
    }

    private static Map<String, Object> parseJsonObject(String body) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (body == null) return out;
        String s = body.trim();
        if (s.startsWith("{")) s = s.substring(1);
        if (s.endsWith("}")) s = s.substring(0, s.length() - 1);
        if (s.isBlank()) return out;
        for (String part : splitTop(s, ',')) {
            int c = indexOfColon(part);
            if (c < 0) continue;
            String k = unquote(part.substring(0, c).trim());
            Object v = parseScalar(part.substring(c + 1).trim());
            out.put(k, v);
        }
        return out;
    }

    private static Object parseScalar(String raw) {
        if (raw == null || raw.equals("null")) return null;
        if (raw.equals("true")) return Boolean.TRUE;
        if (raw.equals("false")) return Boolean.FALSE;
        if (raw.startsWith("\"") && raw.endsWith("\"")) return unquote(raw);
        try {
            if (raw.contains(".") || raw.toLowerCase(Locale.ROOT).contains("e")) {
                return Double.parseDouble(raw);
            }
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            return raw;
        }
    }

    private static List<String> splitTop(String s, char sep) {
        List<String> parts = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQ = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQ = !inQ;
            if (c == sep && !inQ) {
                parts.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        if (cur.length() > 0) parts.add(cur.toString());
        return parts;
    }

    private static int indexOfColon(String s) {
        boolean inQ = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQ = !inQ;
            if (c == ':' && !inQ) return i;
        }
        return -1;
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    private static String unescape(String s) {
        return s.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
    }

    private static String unquote(String s) {
        s = s.trim();
        if (s.startsWith("\"") && s.endsWith("\"") && s.length() >= 2) {
            return unescape(s.substring(1, s.length() - 1));
        }
        return s;
    }
}
