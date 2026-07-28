package org.bytedeco.pytorch.dataframe.dtype;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * JSON cell value (Arrow Utf8-compatible). Pure-Java storage without Jackson.
 */
public class JsonData extends AbstractDataValue implements StructuredData {
    private static final long serialVersionUID = 1L;

    private String rawJson;
    private transient Map<String, Object> parsedMap;

    public JsonData(String rawJson) {
        setRawJson(rawJson);
    }

    public JsonData(Map<String, Object> dataMap) {
        if (dataMap == null) throw new IllegalArgumentException("map must not be null");
        this.parsedMap = new LinkedHashMap<>(dataMap);
        this.rawJson = simpleMapToJson(dataMap);
    }

    public void setRawJson(String rawJson) {
        if (rawJson == null || rawJson.trim().isEmpty()) {
            throw new IllegalArgumentException("JSON string must not be empty");
        }
        this.rawJson = rawJson;
        this.parsedMap = simpleParseObject(rawJson);
    }

    public Map<String, Object> getAsMap() {
        if (parsedMap == null && rawJson != null) parsedMap = simpleParseObject(rawJson);
        return parsedMap == null ? Collections.emptyMap() : parsedMap;
    }

    public String getRawJson() { return rawJson; }

    @Override
    public String getDataType() { return "JSON"; }

    @Override
    public Object toArrowCompatible() { return rawJson; }

    @Override
    public String getShortDesc() {
        String preview = rawJson == null ? "" :
            (rawJson.length() > 50 ? rawJson.substring(0, 50) + "..." : rawJson);
        return String.format("len=%d, content='%s'", rawJson == null ? 0 : rawJson.length(), preview);
    }

    @Override
    public boolean isValid() {
        return super.isValid() && rawJson != null && !rawJson.trim().isEmpty();
    }

    @Override
    public int getSize() {
        return getAsMap().size();
    }

    @Override
    public Map<String, Object> toMap() {
        return getAsMap();
    }

    @Override
    public Number getNumericValue() { return null; }

    // ---- minimal JSON object helpers (string/number/bool/null leaves) ----

    private static String simpleMapToJson(Map<String, Object> map) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(escape(e.getKey())).append('"').append(':');
            Object v = e.getValue();
            if (v == null) sb.append("null");
            else if (v instanceof Number || v instanceof Boolean) sb.append(v);
            else sb.append('"').append(escape(String.valueOf(v))).append('"');
        }
        return sb.append('}').toString();
    }

    /** Best-effort flat object parser; stores raw string values for nested content. */
    private static Map<String, Object> simpleParseObject(String json) {
        Map<String, Object> out = new LinkedHashMap<>();
        String s = json.trim();
        if (!s.startsWith("{") || !s.endsWith("}")) {
            // not a flat object — store as single entry
            out.put("_raw", s);
            return out;
        }
        s = s.substring(1, s.length() - 1).trim();
        if (s.isEmpty()) return out;
        int i = 0;
        while (i < s.length()) {
            while (i < s.length() && (s.charAt(i) == ',' || Character.isWhitespace(s.charAt(i)))) i++;
            if (i >= s.length()) break;
            if (s.charAt(i) != '"') break;
            int k1 = ++i;
            while (i < s.length() && s.charAt(i) != '"') {
                if (s.charAt(i) == '\\') i++;
                i++;
            }
            String key = unescape(s.substring(k1, i));
            i++; // skip closing quote
            while (i < s.length() && s.charAt(i) != ':') i++;
            if (i >= s.length()) break;
            i++; // skip :
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
            if (i >= s.length()) break;
            Object val;
            char c = s.charAt(i);
            if (c == '"') {
                int v1 = ++i;
                while (i < s.length() && s.charAt(i) != '"') {
                    if (s.charAt(i) == '\\') i++;
                    i++;
                }
                val = unescape(s.substring(v1, Math.min(i, s.length())));
                if (i < s.length()) i++;
            } else if (c == '{' || c == '[') {
                int depth = 0; int start = i;
                do {
                    char ch = s.charAt(i);
                    if (ch == '{' || ch == '[') depth++;
                    else if (ch == '}' || ch == ']') depth--;
                    i++;
                } while (i < s.length() && depth > 0);
                val = s.substring(start, i);
            } else {
                int start = i;
                while (i < s.length() && s.charAt(i) != ',') i++;
                String tok = s.substring(start, i).trim();
                if ("null".equals(tok)) val = null;
                else if ("true".equals(tok)) val = Boolean.TRUE;
                else if ("false".equals(tok)) val = Boolean.FALSE;
                else {
                    try {
                        if (tok.contains(".") || tok.contains("e") || tok.contains("E"))
                            val = Double.parseDouble(tok);
                        else val = Long.parseLong(tok);
                    } catch (Exception e) {
                        val = tok;
                    }
                }
            }
            out.put(key, val);
        }
        return out;
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String unescape(String s) {
        return s.replace("\\\"", "\"").replace("\\\\", "\\");
    }
}
