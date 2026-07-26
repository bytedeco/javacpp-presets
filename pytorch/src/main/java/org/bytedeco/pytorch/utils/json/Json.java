package org.bytedeco.pytorch.utils.json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON encode / decode used by Visdom / WandB / SwanLab clients.
 * No external dependencies; supports maps, lists, arrays, numbers, strings, booleans, null, base64 bytes.
 */
public final class Json {
    private Json() {}

    // =========================================================================
    // Encode
    // =========================================================================

    public static String encode(Object value) {
        StringBuilder sb = new StringBuilder(256);
        write(sb, value);
        return sb.toString();
    }

    public static void write(StringBuilder sb, Object value) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof String str) {
            sb.append('"').append(escape(str)).append('"');
        } else if (value instanceof Boolean b) {
            sb.append(b ? "true" : "false");
        } else if (value instanceof Number n) {
            double d = n.doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                sb.append("null");
            } else if (n instanceof Float || n instanceof Double) {
                sb.append(d);
            } else {
                sb.append(n.toString());
            }
        } else if (value instanceof Map<?, ?> map) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (e.getValue() == SKIP) continue;
                if (!first) sb.append(',');
                first = false;
                sb.append('"').append(escape(String.valueOf(e.getKey()))).append('"').append(':');
                write(sb, e.getValue());
            }
            sb.append('}');
        } else if (value instanceof Collection<?> col) {
            sb.append('[');
            boolean first = true;
            for (Object o : col) {
                if (!first) sb.append(',');
                first = false;
                write(sb, o);
            }
            sb.append(']');
        } else if (value instanceof double[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                double d = arr[i];
                if (Double.isNaN(d) || Double.isInfinite(d)) sb.append("null");
                else sb.append(d);
            }
            sb.append(']');
        } else if (value instanceof float[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                float d = arr[i];
                if (Float.isNaN(d) || Float.isInfinite(d)) sb.append("null");
                else sb.append(d);
            }
            sb.append(']');
        } else if (value instanceof int[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(arr[i]);
            }
            sb.append(']');
        } else if (value instanceof long[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(arr[i]);
            }
            sb.append(']');
        } else if (value instanceof byte[] arr) {
            sb.append('"').append(Base64.getEncoder().encodeToString(arr)).append('"');
        } else if (value instanceof boolean[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(arr[i] ? "true" : "false");
            }
            sb.append(']');
        } else if (value instanceof double[][] arr2) {
            sb.append('[');
            for (int i = 0; i < arr2.length; i++) {
                if (i > 0) sb.append(',');
                write(sb, arr2[i]);
            }
            sb.append(']');
        } else if (value instanceof float[][] arr2) {
            sb.append('[');
            for (int i = 0; i < arr2.length; i++) {
                if (i > 0) sb.append(',');
                write(sb, arr2[i]);
            }
            sb.append(']');
        } else if (value instanceof int[][] arr2) {
            sb.append('[');
            for (int i = 0; i < arr2.length; i++) {
                if (i > 0) sb.append(',');
                write(sb, arr2[i]);
            }
            sb.append(']');
        } else if (value instanceof String[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                write(sb, arr[i]);
            }
            sb.append(']');
        } else if (value instanceof Object[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                write(sb, arr[i]);
            }
            sb.append(']');
        } else if (value instanceof CharSequence cs) {
            sb.append('"').append(escape(cs.toString())).append('"');
        } else {
            sb.append('"').append(escape(String.valueOf(value))).append('"');
        }
    }

    /** Sentinel: skip this map entry when encoding (used for optional null fields). */
    public static final Object SKIP = new Object();

    public static String escape(String s) {
        if (s == null) return "";
        StringBuilder out = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\' -> out.append("\\\\");
                case '"' -> out.append("\\\"");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.toString();
    }

    // =========================================================================
    // Decode (lightweight — enough for API responses)
    // =========================================================================

    public static Object decode(String json) throws IOException {
        if (json == null) return null;
        Parser p = new Parser(json.trim());
        Object v = p.parseValue();
        p.skipWs();
        return v;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> decodeObject(String json) throws IOException {
        Object v = decode(json);
        if (v == null) return new LinkedHashMap<>();
        if (v instanceof Map) return (Map<String, Object>) v;
        throw new IOException("expected JSON object, got " + v.getClass().getSimpleName());
    }

    @SuppressWarnings("unchecked")
    public static List<Object> decodeArray(String json) throws IOException {
        Object v = decode(json);
        if (v == null) return new ArrayList<>();
        if (v instanceof List) return (List<Object>) v;
        throw new IOException("expected JSON array, got " + v.getClass().getSimpleName());
    }

    private static final class Parser {
        final String s;
        int i;

        Parser(String s) { this.s = s; }

        void skipWs() {
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') i++;
                else break;
            }
        }

        char peek() throws IOException {
            skipWs();
            if (i >= s.length()) throw new IOException("unexpected end of JSON");
            return s.charAt(i);
        }

        char next() throws IOException {
            skipWs();
            if (i >= s.length()) throw new IOException("unexpected end of JSON");
            return s.charAt(i++);
        }

        Object parseValue() throws IOException {
            char c = peek();
            return switch (c) {
                case '{' -> parseObject();
                case '[' -> parseArray();
                case '"' -> parseString();
                case 't' -> parseLiteral("true", Boolean.TRUE);
                case 'f' -> parseLiteral("false", Boolean.FALSE);
                case 'n' -> parseLiteral("null", null);
                case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' -> parseNumber();
                default -> throw new IOException("unexpected char '" + c + "' at " + i);
            };
        }

        Map<String, Object> parseObject() throws IOException {
            next(); // {
            Map<String, Object> map = new LinkedHashMap<>();
            skipWs();
            if (peek() == '}') { next(); return map; }
            while (true) {
                String key = parseString();
                if (next() != ':') throw new IOException("expected ':' after key");
                map.put(key, parseValue());
                char c = next();
                if (c == '}') break;
                if (c != ',') throw new IOException("expected ',' or '}' in object");
            }
            return map;
        }

        List<Object> parseArray() throws IOException {
            next(); // [
            List<Object> list = new ArrayList<>();
            skipWs();
            if (peek() == ']') { next(); return list; }
            while (true) {
                list.add(parseValue());
                char c = next();
                if (c == ']') break;
                if (c != ',') throw new IOException("expected ',' or ']' in array");
            }
            return list;
        }

        String parseString() throws IOException {
            if (next() != '"') throw new IOException("expected '\"'");
            StringBuilder sb = new StringBuilder();
            while (i < s.length()) {
                char c = s.charAt(i++);
                if (c == '"') return sb.toString();
                if (c == '\\') {
                    if (i >= s.length()) throw new IOException("bad escape");
                    char e = s.charAt(i++);
                    switch (e) {
                        case '"', '\\', '/' -> sb.append(e);
                        case 'b' -> sb.append('\b');
                        case 'f' -> sb.append('\f');
                        case 'n' -> sb.append('\n');
                        case 'r' -> sb.append('\r');
                        case 't' -> sb.append('\t');
                        case 'u' -> {
                            if (i + 4 > s.length()) throw new IOException("bad unicode escape");
                            int code = Integer.parseInt(s.substring(i, i + 4), 16);
                            sb.append((char) code);
                            i += 4;
                        }
                        default -> throw new IOException("bad escape \\" + e);
                    }
                } else {
                    sb.append(c);
                }
            }
            throw new IOException("unterminated string");
        }

        Object parseLiteral(String lit, Object value) throws IOException {
            if (!s.startsWith(lit, i)) throw new IOException("expected " + lit);
            i += lit.length();
            return value;
        }

        Number parseNumber() throws IOException {
            int start = i;
            if (s.charAt(i) == '-') i++;
            while (i < s.length() && Character.isDigit(s.charAt(i))) i++;
            boolean isFloat = false;
            if (i < s.length() && s.charAt(i) == '.') {
                isFloat = true;
                i++;
                while (i < s.length() && Character.isDigit(s.charAt(i))) i++;
            }
            if (i < s.length() && (s.charAt(i) == 'e' || s.charAt(i) == 'E')) {
                isFloat = true;
                i++;
                if (i < s.length() && (s.charAt(i) == '+' || s.charAt(i) == '-')) i++;
                while (i < s.length() && Character.isDigit(s.charAt(i))) i++;
            }
            String num = s.substring(start, i);
            if (isFloat) return Double.parseDouble(num);
            try {
                long l = Long.parseLong(num);
                if (l >= Integer.MIN_VALUE && l <= Integer.MAX_VALUE) return (int) l;
                return l;
            } catch (NumberFormatException e) {
                return Double.parseDouble(num);
            }
        }
    }
}
