package org.bytedeco.pytorch.dataframe.io;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.data.json.JsonParser;
import org.bytedeco.pytorch.data.json.JsonValue;
import org.bytedeco.pytorch.data.json.JsonWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Shared encode / decode for complex DataFrame cells across tabular formats
 * (CSV, Excel, HDF5 string columns, Avro/Arrow/ORC string fallbacks, JSON).
 *
 * <p>Canonical text form is JSON:
 * <ul>
 *   <li>{@code LIST} / nested sequences → JSON array</li>
 *   <li>{@code VECTOR} / {@code EMBEDDING} → JSON array of numbers (or dense {@code float[]})</li>
 *   <li>{@code MAP} / {@code STRUCT} → JSON object</li>
 *   <li>{@code JSON} → structured JSON or string</li>
 * </ul>
 *
 * <p>Binary formats with native nested types (Parquet LIST/MAP, Arrow List/Map/Struct,
 * Avro array/map/record) should prefer their native encodings and only use this codec
 * for text fallbacks or when a cell must be round-tripped as a string.
 */
public final class ComplexCellCodec {
    private ComplexCellCodec() {}

    /** True if dtype is a nested / structured column type. */
    public static boolean isComplex(Column.DType dt) {
        if (dt == null) return false;
        return switch (dt) {
            case LIST, MAP, STRUCT, VECTOR, EMBEDDING, JSON,
                 TENSOR, GRAPH, POINT_CLOUD, IMAGE, AUDIO, VIDEO, BINARY -> true;
            default -> false;
        };
    }

    /** True if dtype is list-like (including vector / embedding). */
    public static boolean isListLike(Column.DType dt) {
        return dt == Column.DType.LIST
            || dt == Column.DType.VECTOR
            || dt == Column.DType.EMBEDDING;
    }

    /** True if dtype is dict / struct-like. */
    public static boolean isMapLike(Column.DType dt) {
        return dt == Column.DType.MAP || dt == Column.DType.STRUCT;
    }

    /**
     * Infer a complex-aware dtype from a Java cell value.
     * Prefer this over {@link IoTypeCoercion#inferFromObject} when nested cells are expected.
     */
    public static Column.DType inferComplex(Object v) {
        if (v == null) return Column.DType.STRING;
        if (v instanceof float[] || v instanceof double[]) return Column.DType.VECTOR;
        if (v instanceof int[] || v instanceof long[] || v instanceof boolean[]) return Column.DType.LIST;
        if (v instanceof byte[]) return Column.DType.BINARY;
        if (v instanceof Map) return Column.DType.MAP;
        if (v instanceof List || v instanceof Collection) {
            // homogeneous float list → VECTOR; else LIST
            Collection<?> c = (Collection<?>) v;
            if (c.isEmpty()) return Column.DType.LIST;
            boolean allFloat = true;
            boolean allNumber = true;
            for (Object o : c) {
                if (o == null) continue;
                if (!(o instanceof Number)) {
                    allNumber = false;
                    allFloat = false;
                    break;
                }
                if (!(o instanceof Float) && !(o instanceof Double)) allFloat = false;
            }
            if (allFloat && allNumber) return Column.DType.VECTOR;
            return Column.DType.LIST;
        }
        if (v instanceof JsonValue jv) {
            if (jv.isArray()) return Column.DType.LIST;
            if (jv.isObject()) return Column.DType.MAP;
            return Column.DType.JSON;
        }
        if (v instanceof CharSequence) {
            String s = v.toString().trim();
            if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
                try {
                    JsonValue j = JsonParser.parse(s);
                    if (j.isArray()) {
                        // numeric array → VECTOR; else LIST
                        if (j.size() > 0) {
                            boolean allNum = true;
                            for (int i = 0; i < j.size(); i++) {
                                JsonValue el = j.get(i);
                                if (el == null || el.isNull()) continue;
                                if (!el.isNumber()) { allNum = false; break; }
                            }
                            if (allNum) return Column.DType.VECTOR;
                        }
                        return Column.DType.LIST;
                    }
                    if (j.isObject()) return Column.DType.MAP;
                    return Column.DType.JSON;
                } catch (Exception ignored) { /* plain string */ }
            }
            return Column.DType.STRING;
        }
        // Scalars — do NOT call IoTypeCoercion.inferFromObject (would re-enter for nested).
        if (v instanceof Boolean) return Column.DType.BOOLEAN;
        if (v instanceof Byte || v instanceof Short || v instanceof Integer) return Column.DType.INT32;
        if (v instanceof Long) return Column.DType.INT64;
        if (v instanceof Float) return Column.DType.FLOAT32;
        if (v instanceof Double || v instanceof Number) return Column.DType.FLOAT64;
        return Column.DType.STRING;
    }

    /**
     * Encode a cell to its canonical JSON text form (for CSV / Excel / string columns).
     * Scalars that are not complex are returned via {@code String.valueOf}.
     */
    public static String encodeText(Object val) {
        if (val == null) return null;
        if (val instanceof String s) {
            // already text — if it looks like JSON keep as-is
            return s;
        }
        if (val instanceof JsonValue jv) {
            return JsonWriter.toString(jv);
        }
        // Fast paths for primitive arrays (avoid any JsonWriter edge cases / toString leaks)
        if (val instanceof float[] a) {
            StringBuilder sb = new StringBuilder(a.length * 8 + 2).append('[');
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (val instanceof double[] a) {
            StringBuilder sb = new StringBuilder(a.length * 8 + 2).append('[');
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (val instanceof int[] a) {
            StringBuilder sb = new StringBuilder(a.length * 4 + 2).append('[');
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (val instanceof long[] a) {
            StringBuilder sb = new StringBuilder(a.length * 4 + 2).append('[');
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (val instanceof boolean[] a) {
            StringBuilder sb = new StringBuilder(a.length * 6 + 2).append('[');
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        try {
            return JsonWriter.toString(JsonValue.fromJava(val));
        } catch (Exception e) {
            // last resort — never emit Object.toString for arrays
            if (val.getClass().isArray()) {
                return JsonWriter.toString(JsonValue.fromJava(toJsonFriendly(val)));
            }
            return String.valueOf(val);
        }
    }

    /**
     * Encode for formats that always want structured JSON (even for scalars that
     * are already stringified numbers etc.). Null → null.
     */
    public static String encodeJsonText(Object val) {
        if (val == null) return null;
        if (val instanceof JsonValue jv) return JsonWriter.toString(jv);
        if (val instanceof String s) {
            String t = s.trim();
            if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
                try {
                    JsonParser.parse(s); // validate
                    return s;
                } catch (Exception ignored) { /* wrap as JSON string */ }
            }
            return JsonWriter.toString(JsonValue.of(s));
        }
        return JsonWriter.toString(JsonValue.fromJava(val));
    }

    /**
     * Decode text into a Java cell for the given complex dtype.
     * Returns null for null/blank input.
     */
    public static Object decodeText(String text, Column.DType dtype) {
        if (text == null) return null;
        String t = text.trim();
        if (t.isEmpty() || "null".equalsIgnoreCase(t)) return null;

        switch (dtype) {
            case VECTOR:
            case EMBEDDING:
                return decodeVector(t);
            case LIST:
                return decodeList(t);
            case MAP:
            case STRUCT:
                return decodeMap(t);
            case JSON:
                return decodeJson(t);
            case BINARY:
                // hex or base-ish: leave as UTF-8 bytes of the text
                return t.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            default:
                // try structured parse when text looks nested
                if ((t.startsWith("[") && t.endsWith("]")) || (t.startsWith("{") && t.endsWith("}"))) {
                    try {
                        return JsonParser.parse(t).toJava();
                    } catch (Exception ignored) { /* fall through */ }
                }
                return text;
        }
    }

    /**
     * Coerce an arbitrary Java value (already materialised) into the target complex dtype.
     * Used by Avro / Arrow / Pickle after native nested decode.
     */
    @SuppressWarnings("unchecked")
    public static Object coerceComplex(Object raw, Column.DType dtype) {
        if (raw == null || dtype == null) return raw;
        switch (dtype) {
            case VECTOR:
            case EMBEDDING:
                if (raw instanceof float[]) return raw;
                if (raw instanceof double[]) {
                    double[] d = (double[]) raw;
                    float[] f = new float[d.length];
                    for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
                    return f;
                }
                if (raw instanceof int[]) {
                    int[] a = (int[]) raw;
                    float[] f = new float[a.length];
                    for (int i = 0; i < a.length; i++) f[i] = a[i];
                    return f;
                }
                if (raw instanceof long[]) {
                    long[] a = (long[]) raw;
                    float[] f = new float[a.length];
                    for (int i = 0; i < a.length; i++) f[i] = a[i];
                    return f;
                }
                if (raw instanceof List) {
                    List<?> list = (List<?>) raw;
                    float[] f = new float[list.size()];
                    for (int i = 0; i < list.size(); i++) {
                        Object o = list.get(i);
                        f[i] = o == null ? Float.NaN : ((Number) o).floatValue();
                    }
                    return f;
                }
                if (raw instanceof CharSequence) return decodeVector(raw.toString());
                return raw;
            case LIST:
                if (raw instanceof List) return densifyList((List<?>) raw);
                if (raw instanceof int[] || raw instanceof long[]
                    || raw instanceof float[] || raw instanceof double[]
                    || raw instanceof boolean[] || raw instanceof Object[]) {
                    return densifyArray(raw);
                }
                if (raw instanceof CharSequence) return decodeList(raw.toString());
                if (raw instanceof Collection) return densifyList(new ArrayList<>((Collection<?>) raw));
                // single scalar → length-1 list
                List<Object> one = new ArrayList<>(1);
                one.add(raw);
                return one;
            case MAP:
            case STRUCT:
                if (raw instanceof Map) return raw;
                if (raw instanceof CharSequence) return decodeMap(raw.toString());
                if (raw instanceof JsonValue jv && jv.isObject()) return jv.toJava();
                return raw;
            case JSON:
                if (raw instanceof JsonValue) return ((JsonValue) raw).toJava();
                if (raw instanceof Map || raw instanceof List) return raw;
                if (raw instanceof CharSequence) return decodeJson(raw.toString());
                return raw;
            default:
                return raw;
        }
    }

    /** Convert a Java cell to a JSON-friendly structure (List / Map / Number / String). */
    public static Object toJsonFriendly(Object val) {
        if (val == null) return null;
        if (val instanceof JsonValue) return ((JsonValue) val).toJava();
        if (val instanceof float[] a) {
            List<Double> out = new ArrayList<>(a.length);
            for (float x : a) out.add((double) x);
            return out;
        }
        if (val instanceof double[] a) {
            List<Double> out = new ArrayList<>(a.length);
            for (double x : a) out.add(x);
            return out;
        }
        if (val instanceof int[] a) {
            List<Integer> out = new ArrayList<>(a.length);
            for (int x : a) out.add(x);
            return out;
        }
        if (val instanceof long[] a) {
            List<Long> out = new ArrayList<>(a.length);
            for (long x : a) out.add(x);
            return out;
        }
        if (val instanceof boolean[] a) {
            List<Boolean> out = new ArrayList<>(a.length);
            for (boolean x : a) out.add(x);
            return out;
        }
        if (val instanceof byte[]) {
            // keep as list of ints for JSON portability
            byte[] a = (byte[]) val;
            List<Integer> out = new ArrayList<>(a.length);
            for (byte x : a) out.add(x & 0xff);
            return out;
        }
        if (val instanceof Object[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (Object o : a) out.add(toJsonFriendly(o));
            return out;
        }
        if (val instanceof Collection<?> c) {
            List<Object> out = new ArrayList<>(c.size());
            for (Object o : c) out.add(toJsonFriendly(o));
            return out;
        }
        if (val instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                out.put(String.valueOf(e.getKey()), toJsonFriendly(e.getValue()));
            }
            return out;
        }
        return val;
    }

    // ---- internals ----------------------------------------------------------

    private static Object decodeVector(String t) {
        // Reject classic Java array toString leaks like "[F@1a2b3c4"
        if (t.matches("^\\[[A-Z]@?[0-9a-fA-F]+$") || t.matches("^\\[F@[0-9a-fA-F]+$")) {
            return new float[0];
        }
        if (t.startsWith("[")) {
            try {
                Object java = JsonParser.parse(t).toJava();
                return coerceComplex(java, Column.DType.VECTOR);
            } catch (Exception e) {
                try {
                    return IoTypeCoercion.parseVector(t);
                } catch (Exception e2) {
                    return new float[0];
                }
            }
        }
        try {
            return IoTypeCoercion.parseVector(t);
        } catch (Exception e) {
            return new float[0];
        }
    }

    private static Object decodeList(String t) {
        if (t.startsWith("[")) {
            try {
                Object java = JsonParser.parse(t).toJava();
                return coerceComplex(java, Column.DType.LIST);
            } catch (Exception ignored) { /* fall through */ }
        }
        // comma-separated fallback
        String body = t;
        if (body.startsWith("[") && body.endsWith("]")) {
            body = body.substring(1, body.length() - 1).trim();
        }
        if (body.isEmpty()) return new ArrayList<>();
        String[] parts = body.split(",");
        List<Object> out = new ArrayList<>(parts.length);
        for (String p : parts) {
            String s = p.trim();
            if (s.isEmpty() || "null".equalsIgnoreCase(s)) {
                out.add(null);
            } else if (IoTypeCoercion.isLong(s)) {
                out.add(Long.parseLong(s));
            } else if (IoTypeCoercion.isDouble(s)) {
                out.add(Double.parseDouble(s));
            } else if (IoTypeCoercion.isBoolean(s)) {
                out.add(IoTypeCoercion.parseBoolean(s));
            } else {
                // strip surrounding quotes
                if (s.length() >= 2 && ((s.startsWith("\"") && s.endsWith("\""))
                    || (s.startsWith("'") && s.endsWith("'")))) {
                    s = s.substring(1, s.length() - 1);
                }
                out.add(s);
            }
        }
        return densifyList(out);
    }

    private static Object decodeMap(String t) {
        if (t.startsWith("{")) {
            try {
                Object java = JsonParser.parse(t).toJava();
                if (java instanceof Map) return java;
            } catch (Exception ignored) { /* fall through */ }
        }
        // empty / invalid → empty map
        return new LinkedHashMap<String, Object>();
    }

    private static Object decodeJson(String t) {
        try {
            return JsonParser.parse(t).toJava();
        } catch (Exception e) {
            return t;
        }
    }

    /** Densify homogeneous numeric List → primitive array; else keep List. */
    public static Object densifyList(List<?> elems) {
        if (elems == null) return null;
        if (elems.isEmpty()) return new ArrayList<>();
        boolean allLong = true, allInt = true, allFloat = true, allDouble = true, allBool = true;
        for (Object o : elems) {
            if (o == null) {
                allLong = allInt = allFloat = allDouble = allBool = false;
                break;
            }
            if (!(o instanceof Number) && !(o instanceof Boolean)) {
                allLong = allInt = allFloat = allDouble = allBool = false;
                break;
            }
            if (o instanceof Boolean) {
                allLong = allInt = allFloat = allDouble = false;
            } else {
                allBool = false;
                if (!(o instanceof Long) && !(o instanceof Integer)
                    && !(o instanceof Short) && !(o instanceof Byte)) allLong = false;
                if (!(o instanceof Integer) && !(o instanceof Short) && !(o instanceof Byte)) allInt = false;
                if (!(o instanceof Float)) allFloat = false;
                if (!(o instanceof Double) && !(o instanceof Float)) allDouble = false;
            }
        }
        if (allInt) {
            int[] a = new int[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).intValue();
            return a;
        }
        if (allLong) {
            long[] a = new long[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).longValue();
            return a;
        }
        if (allFloat) {
            float[] a = new float[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).floatValue();
            return a;
        }
        if (allDouble) {
            double[] a = new double[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).doubleValue();
            return a;
        }
        if (allBool) {
            boolean[] a = new boolean[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = (Boolean) elems.get(i);
            return a;
        }
        // recursive densify nested lists; keep as List
        List<Object> out = new ArrayList<>(elems.size());
        for (Object o : elems) {
            if (o instanceof List) out.add(densifyList((List<?>) o));
            else if (o instanceof Map) out.add(o);
            else out.add(o);
        }
        return out;
    }

    private static Object densifyArray(Object arr) {
        if (arr instanceof int[] || arr instanceof long[]
            || arr instanceof float[] || arr instanceof double[]
            || arr instanceof boolean[]) {
            return arr;
        }
        if (arr instanceof Object[] a) {
            return densifyList(Arrays.asList(a));
        }
        return arr;
    }

    /** Flatten a list-like cell into a List of boxed scalars (for Avro array / Arrow list write). */
    public static List<Object> asObjectList(Object cell) {
        if (cell == null) return null;
        if (cell instanceof List) {
            List<?> src = (List<?>) cell;
            List<Object> out = new ArrayList<>(src.size());
            out.addAll(src);
            return out;
        }
        if (cell instanceof int[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (int x : a) out.add(x);
            return out;
        }
        if (cell instanceof long[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (long x : a) out.add(x);
            return out;
        }
        if (cell instanceof float[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (float x : a) out.add(x);
            return out;
        }
        if (cell instanceof double[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (double x : a) out.add(x);
            return out;
        }
        if (cell instanceof boolean[] a) {
            List<Object> out = new ArrayList<>(a.length);
            for (boolean x : a) out.add(x);
            return out;
        }
        if (cell instanceof Object[] a) {
            return new ArrayList<>(Arrays.asList(a));
        }
        if (cell instanceof Collection<?> c) {
            return new ArrayList<>(c);
        }
        if (cell instanceof CharSequence) {
            Object decoded = decodeList(cell.toString());
            return asObjectList(decoded);
        }
        List<Object> one = new ArrayList<>(1);
        one.add(cell);
        return one;
    }

    /** Flatten a map-like cell into Map&lt;String,Object&gt;. */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> asStringMap(Object cell) {
        if (cell == null) return null;
        if (cell instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) cell;
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                out.put(String.valueOf(e.getKey()), e.getValue());
            }
            return out;
        }
        if (cell instanceof CharSequence) {
            Object d = decodeMap(cell.toString());
            if (d instanceof Map) return (Map<String, Object>) d;
        }
        if (cell instanceof JsonValue jv && jv.isObject()) {
            Object j = jv.toJava();
            if (j instanceof Map) return (Map<String, Object>) j;
        }
        return new LinkedHashMap<>();
    }
}
