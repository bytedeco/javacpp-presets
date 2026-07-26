package org.bytedeco.pytorch.data.json;

import java.util.ArrayList;
import java.util.List;

/**
 * Lightweight JSONPath-like accessor supporting:
 * <ul>
 *   <li>{@code .key} / {@code key} — object field</li>
 *   <li>{@code [0]} — array index (negative from end)</li>
 *   <li>{@code ["key.with.dots"]} — quoted key</li>
 *   <li>chained: {@code a.b[0].c}</li>
 * </ul>
 */
public final class JsonPath {
    private JsonPath() {}

    public static JsonValue get(JsonValue root, String path) {
        if (root == null) return JsonValue.NULL;
        if (path == null || path.isEmpty() || "$".equals(path)) return root;
        String p = path.startsWith("$.") ? path.substring(2)
            : path.startsWith("$") ? path.substring(1) : path;
        if (p.startsWith(".")) p = p.substring(1);

        JsonValue cur = root;
        int i = 0;
        while (i < p.length()) {
            char c = p.charAt(i);
            if (c == '.') {
                i++;
                continue;
            }
            if (c == '[') {
                int close = findMatchingBracket(p, i);
                String inside = p.substring(i + 1, close).trim();
                cur = applyBracket(cur, inside);
                i = close + 1;
            } else {
                int end = i;
                while (end < p.length()) {
                    char ch = p.charAt(end);
                    if (ch == '.' || ch == '[') break;
                    end++;
                }
                String key = p.substring(i, end);
                if (key.isEmpty()) {
                    i = end;
                    continue;
                }
                if (!cur.isObject() || !cur.has(key)) {
                    throw new JsonException("Path not found: " + path + " (at key '" + key + "')");
                }
                cur = cur.get(key);
                i = end;
            }
        }
        return cur;
    }

    public static boolean exists(JsonValue root, String path) {
        try {
            get(root, path);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Flatten a nested object/array into dotted keys.
     * Arrays become {@code key[0]}, {@code key[1]}, ...
     * Leaf values are returned as Java objects via {@link JsonValue#toJava()}.
     */
    public static java.util.LinkedHashMap<String, Object> flatten(JsonValue value) {
        java.util.LinkedHashMap<String, Object> out = new java.util.LinkedHashMap<>();
        flattenInto(value, "", out);
        return out;
    }

    private static void flattenInto(JsonValue v, String prefix, java.util.Map<String, Object> out) {
        if (v == null || v.isNull()) {
            if (!prefix.isEmpty()) out.put(prefix, null);
            return;
        }
        if (v.isObject()) {
            if (v.isEmpty() && !prefix.isEmpty()) {
                out.put(prefix, new java.util.LinkedHashMap<>());
                return;
            }
            for (String k : v.keySet()) {
                String next = prefix.isEmpty() ? k : prefix + "." + k;
                flattenInto(v.get(k), next, out);
            }
        } else if (v.isArray()) {
            if (v.isEmpty() && !prefix.isEmpty()) {
                out.put(prefix, new java.util.ArrayList<>());
                return;
            }
            for (int i = 0; i < v.size(); i++) {
                String next = prefix + "[" + i + "]";
                flattenInto(v.get(i), next, out);
            }
        } else {
            out.put(prefix.isEmpty() ? "_value" : prefix, v.toJava());
        }
    }

    private static JsonValue applyBracket(JsonValue cur, String inside) {
        if (inside.isEmpty()) throw new JsonException("Empty bracket in path");
        if (inside.charAt(0) == '"' || inside.charAt(0) == '\'') {
            char q = inside.charAt(0);
            if (inside.length() < 2 || inside.charAt(inside.length() - 1) != q) {
                throw new JsonException("Unclosed quoted key: " + inside);
            }
            String key = inside.substring(1, inside.length() - 1);
            if (!cur.isObject() || !cur.has(key)) {
                throw new JsonException("Missing key: " + key);
            }
            return cur.get(key);
        }
        // numeric index
        int idx;
        try {
            idx = Integer.parseInt(inside);
        } catch (NumberFormatException e) {
            // treat as bare key
            if (!cur.isObject() || !cur.has(inside)) {
                throw new JsonException("Missing key: " + inside);
            }
            return cur.get(inside);
        }
        if (!cur.isArray()) throw new JsonException("Not an array for index " + idx);
        if (idx < 0) idx = cur.size() + idx;
        if (idx < 0 || idx >= cur.size()) {
            throw new JsonException("Array index out of bounds: " + idx + " size=" + cur.size());
        }
        return cur.get(idx);
    }

    private static int findMatchingBracket(String p, int open) {
        int i = open + 1;
        boolean inStr = false;
        char q = 0;
        while (i < p.length()) {
            char c = p.charAt(i);
            if (inStr) {
                if (c == '\\') { i += 2; continue; }
                if (c == q) inStr = false;
            } else {
                if (c == '"' || c == '\'') { inStr = true; q = c; }
                else if (c == ']') return i;
            }
            i++;
        }
        throw new JsonException("Unclosed '[' in path: " + p);
    }

    /** Split path into segments for diagnostics. */
    public static List<String> segments(String path) {
        List<String> out = new ArrayList<>();
        if (path == null || path.isEmpty()) return out;
        String p = path.startsWith("$.") ? path.substring(2)
            : path.startsWith("$") ? path.substring(1) : path;
        if (p.startsWith(".")) p = p.substring(1);
        int i = 0;
        while (i < p.length()) {
            char c = p.charAt(i);
            if (c == '.') { i++; continue; }
            if (c == '[') {
                int close = findMatchingBracket(p, i);
                out.add(p.substring(i, close + 1));
                i = close + 1;
            } else {
                int end = i;
                while (end < p.length() && p.charAt(end) != '.' && p.charAt(end) != '[') end++;
                out.add(p.substring(i, end));
                i = end;
            }
        }
        return out;
    }
}
