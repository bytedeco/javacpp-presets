package org.bytedeco.pytorch.data.json;
import java.io.Serializable;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

/**
 * Immutable-ish JSON value AST (RFC 8259). Pure Java, no external deps.
 *
 * <p>Types: {@code NULL}, {@code BOOLEAN}, {@code NUMBER}, {@code STRING},
 * {@code ARRAY}, {@code OBJECT}. Numbers keep their original lexical form so
 * integers stay integers and large values do not lose precision.
 */
public final class JsonValue implements Serializable, Iterable<JsonValue> {
    private static final long serialVersionUID = 1L;

    public enum Type { NULL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT }

    public static final JsonValue NULL = new JsonValue(Type.NULL, null, null, null, null);
    public static final JsonValue TRUE = new JsonValue(Type.BOOLEAN, Boolean.TRUE, null, null, null);
    public static final JsonValue FALSE = new JsonValue(Type.BOOLEAN, Boolean.FALSE, null, null, null);

    private final Type type;
    private final Object scalar;              // Boolean / String / NumberToken
    private final List<JsonValue> array;      // for ARRAY
    private final LinkedHashMap<String, JsonValue> object; // for OBJECT
    private final String numberLex;           // original number text

    private JsonValue(Type type, Object scalar, String numberLex,
                      List<JsonValue> array, LinkedHashMap<String, JsonValue> object) {
        this.type = type;
        this.scalar = scalar;
        this.numberLex = numberLex;
        this.array = array;
        this.object = object;
    }

    // ---- factories ----

    public static JsonValue ofNull() { return NULL; }

    public static JsonValue of(boolean v) { return v ? TRUE : FALSE; }

    public static JsonValue of(String s) {
        if (s == null) return NULL;
        return new JsonValue(Type.STRING, s, null, null, null);
    }

    public static JsonValue of(long v) {
        return new JsonValue(Type.NUMBER, null, Long.toString(v), null, null);
    }

    public static JsonValue of(int v) {
        return new JsonValue(Type.NUMBER, null, Integer.toString(v), null, null);
    }

    public static JsonValue of(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            throw new JsonException("JSON number cannot be NaN or Infinity");
        }
        // Prefer compact representation without scientific noise when possible
        String lex = Double.toString(v);
        return new JsonValue(Type.NUMBER, null, lex, null, null);
    }

    public static JsonValue of(float v) {
        return of((double) v);
    }

    public static JsonValue of(Number n) {
        if (n == null) return NULL;
        if (n instanceof BigDecimal) return ofNumberLex(((BigDecimal) n).toPlainString());
        if (n instanceof BigInteger) return ofNumberLex(n.toString());
        if (n instanceof Double || n instanceof Float) return of(n.doubleValue());
        if (n instanceof Long || n instanceof Integer || n instanceof Short || n instanceof Byte)
            return of(n.longValue());
        return ofNumberLex(n.toString());
    }

    /** Create a NUMBER from raw lexical form (must be a valid JSON number). */
    public static JsonValue ofNumberLex(String lex) {
        if (lex == null || lex.isEmpty()) throw new JsonException("empty number");
        return new JsonValue(Type.NUMBER, null, lex, null, null);
    }

    public static JsonValue array() {
        return new JsonValue(Type.ARRAY, null, null, new ArrayList<>(), null);
    }

    public static JsonValue array(JsonValue... items) {
        List<JsonValue> list = new ArrayList<>(items.length);
        for (JsonValue v : items) list.add(v == null ? NULL : v);
        return new JsonValue(Type.ARRAY, null, null, list, null);
    }

    public static JsonValue arrayOf(Collection<?> items) {
        List<JsonValue> list = new ArrayList<>(items == null ? 0 : items.size());
        if (items != null) {
            for (Object o : items) {
                if (o instanceof JsonValue) list.add((JsonValue) o);
                else list.add(fromJava(o));
            }
        }
        return new JsonValue(Type.ARRAY, null, null, list, null);
    }

    public static JsonValue object() {
        return new JsonValue(Type.OBJECT, null, null, null, new LinkedHashMap<>());
    }

    public static JsonValue object(Map<String, ?> map) {
        LinkedHashMap<String, JsonValue> m = new LinkedHashMap<>();
        if (map != null) {
            for (Map.Entry<String, ?> e : map.entrySet()) {
                m.put(e.getKey(), fromJava(e.getValue()));
            }
        }
        return new JsonValue(Type.OBJECT, null, null, null, m);
    }

    /**
     * Convert arbitrary Java value to JsonValue.
     * Supports null, Boolean, Number, String, CharSequence, Map, Collection, arrays,
     * JsonValue, Enum (name), and falls back to {@code String.valueOf}.
     */
    @SuppressWarnings("unchecked")
    public static JsonValue fromJava(Object o) {
        if (o == null) return NULL;
        if (o instanceof JsonValue) return (JsonValue) o;
        if (o instanceof Boolean) return of((Boolean) o);
        if (o instanceof Number) return of((Number) o);
        if (o instanceof CharSequence) return of(o.toString());
        if (o instanceof Character) return of(String.valueOf(o));
        if (o instanceof Enum) return of(((Enum<?>) o).name());
        if (o instanceof Map) return object((Map<String, ?>) o);
        if (o instanceof Collection) return arrayOf((Collection<?>) o);
        if (o instanceof Object[]) return arrayOf(Arrays.asList((Object[]) o));
        if (o instanceof int[]) {
            int[] a = (int[]) o;
            List<JsonValue> list = new ArrayList<>(a.length);
            for (int v : a) list.add(of(v));
            return new JsonValue(Type.ARRAY, null, null, list, null);
        }
        if (o instanceof long[]) {
            long[] a = (long[]) o;
            List<JsonValue> list = new ArrayList<>(a.length);
            for (long v : a) list.add(of(v));
            return new JsonValue(Type.ARRAY, null, null, list, null);
        }
        if (o instanceof double[]) {
            double[] a = (double[]) o;
            List<JsonValue> list = new ArrayList<>(a.length);
            for (double v : a) list.add(of(v));
            return new JsonValue(Type.ARRAY, null, null, list, null);
        }
        if (o instanceof float[]) {
            float[] a = (float[]) o;
            List<JsonValue> list = new ArrayList<>(a.length);
            for (float v : a) list.add(of(v));
            return new JsonValue(Type.ARRAY, null, null, list, null);
        }
        if (o instanceof boolean[]) {
            boolean[] a = (boolean[]) o;
            List<JsonValue> list = new ArrayList<>(a.length);
            for (boolean v : a) list.add(of(v));
            return new JsonValue(Type.ARRAY, null, null, list, null);
        }
        if (o instanceof byte[]) {
            // encode as base64-ish hex array of numbers? prefer int list of unsigned bytes
            byte[] a = (byte[]) o;
            List<JsonValue> list = new ArrayList<>(a.length);
            for (byte v : a) list.add(of(v & 0xFF));
            return new JsonValue(Type.ARRAY, null, null, list, null);
        }
        // temporal / common types → string
        if (o instanceof java.time.temporal.Temporal || o instanceof java.util.Date
            || o instanceof java.util.UUID) {
            return of(String.valueOf(o));
        }
        return of(String.valueOf(o));
    }

    // ---- type queries ----

    public Type type() { return type; }
    public boolean isNull() { return type == Type.NULL; }
    public boolean isBoolean() { return type == Type.BOOLEAN; }
    public boolean isNumber() { return type == Type.NUMBER; }
    public boolean isString() { return type == Type.STRING; }
    public boolean isArray() { return type == Type.ARRAY; }
    public boolean isObject() { return type == Type.OBJECT; }
    public boolean isPrimitive() {
        return type == Type.NULL || type == Type.BOOLEAN || type == Type.NUMBER || type == Type.STRING;
    }

    // ---- scalar accessors ----

    public boolean asBoolean() {
        require(Type.BOOLEAN);
        return (Boolean) scalar;
    }

    public boolean asBoolean(boolean defaultValue) {
        if (type == Type.NULL) return defaultValue;
        if (type == Type.BOOLEAN) return (Boolean) scalar;
        if (type == Type.NUMBER) return asDouble() != 0.0;
        if (type == Type.STRING) {
            String s = ((String) scalar).trim();
            if ("true".equalsIgnoreCase(s) || "1".equals(s) || "yes".equalsIgnoreCase(s)) return true;
            if ("false".equalsIgnoreCase(s) || "0".equals(s) || "no".equalsIgnoreCase(s)) return false;
        }
        return defaultValue;
    }

    public String asString() {
        if (type == Type.NULL) return null;
        if (type == Type.STRING) return (String) scalar;
        if (type == Type.NUMBER) return numberLex;
        if (type == Type.BOOLEAN) return String.valueOf(scalar);
        throw new JsonException("Not a string-compatible value: " + type);
    }

    public String asString(String defaultValue) {
        if (type == Type.NULL) return defaultValue;
        try { return asString(); } catch (JsonException e) { return defaultValue; }
    }

    public String numberLex() {
        require(Type.NUMBER);
        return numberLex;
    }

    public boolean isIntegralNumber() {
        if (type != Type.NUMBER) return false;
        return numberLex.indexOf('.') < 0 && numberLex.indexOf('e') < 0 && numberLex.indexOf('E') < 0;
    }

    public long asLong() {
        require(Type.NUMBER);
        try {
            if (isIntegralNumber()) return Long.parseLong(numberLex);
            return new BigDecimal(numberLex).longValueExact();
        } catch (Exception e) {
            throw new JsonException("Cannot convert number to long: " + numberLex, e);
        }
    }

    public long asLong(long defaultValue) {
        if (type == Type.NULL) return defaultValue;
        if (type != Type.NUMBER) return defaultValue;
        try { return asLong(); } catch (Exception e) { return defaultValue; }
    }

    public int asInt() {
        long v = asLong();
        if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE)
            throw new JsonException("Number out of int range: " + numberLex);
        return (int) v;
    }

    public double asDouble() {
        require(Type.NUMBER);
        try {
            return Double.parseDouble(numberLex);
        } catch (Exception e) {
            throw new JsonException("Cannot convert number to double: " + numberLex, e);
        }
    }

    public double asDouble(double defaultValue) {
        if (type == Type.NULL) return defaultValue;
        if (type != Type.NUMBER) return defaultValue;
        try { return asDouble(); } catch (Exception e) { return defaultValue; }
    }

    public BigDecimal asBigDecimal() {
        require(Type.NUMBER);
        return new BigDecimal(numberLex);
    }

    public Number asNumber() {
        require(Type.NUMBER);
        if (isIntegralNumber()) {
            try {
                long v = Long.parseLong(numberLex);
                if (v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE) return (int) v;
                return v;
            } catch (NumberFormatException e) {
                return new BigInteger(numberLex);
            }
        }
        double d = Double.parseDouble(numberLex);
        // prefer Double; BigDecimal when precision matters and lex is long
        if (numberLex.length() > 18) return new BigDecimal(numberLex);
        return d;
    }

    // ---- array accessors ----

    public int size() {
        if (type == Type.ARRAY) return array.size();
        if (type == Type.OBJECT) return object.size();
        return 0;
    }

    public boolean isEmpty() { return size() == 0; }

    public JsonValue get(int index) {
        require(Type.ARRAY);
        return array.get(index);
    }

    public JsonValue get(int index, JsonValue defaultValue) {
        if (type != Type.ARRAY || index < 0 || index >= array.size()) return defaultValue;
        return array.get(index);
    }

    public List<JsonValue> asArray() {
        require(Type.ARRAY);
        return Collections.unmodifiableList(array);
    }

    public JsonValue add(JsonValue v) {
        require(Type.ARRAY);
        array.add(v == null ? NULL : v);
        return this;
    }

    public JsonValue addAll(Collection<JsonValue> vs) {
        require(Type.ARRAY);
        if (vs != null) for (JsonValue v : vs) array.add(v == null ? NULL : v);
        return this;
    }

    // ---- object accessors ----

    public JsonValue get(String key) {
        require(Type.OBJECT);
        JsonValue v = object.get(key);
        if (v == null) throw new JsonException("Missing key: " + key);
        return v;
    }

    public JsonValue get(String key, JsonValue defaultValue) {
        if (type != Type.OBJECT) return defaultValue;
        JsonValue v = object.get(key);
        return v == null ? defaultValue : v;
    }

    public boolean has(String key) {
        return type == Type.OBJECT && object.containsKey(key);
    }

    public JsonValue put(String key, JsonValue value) {
        require(Type.OBJECT);
        if (key == null) throw new JsonException("object key must not be null");
        object.put(key, value == null ? NULL : value);
        return this;
    }

    public JsonValue put(String key, Object javaValue) {
        return put(key, fromJava(javaValue));
    }

    public JsonValue remove(String key) {
        require(Type.OBJECT);
        object.remove(key);
        return this;
    }

    public Set<String> keySet() {
        require(Type.OBJECT);
        return Collections.unmodifiableSet(object.keySet());
    }

    public Map<String, JsonValue> asObject() {
        require(Type.OBJECT);
        return Collections.unmodifiableMap(object);
    }

    public LinkedHashMap<String, JsonValue> asMutableObject() {
        require(Type.OBJECT);
        return object;
    }

    // ---- path access (dot / bracket) ----

    /**
     * Resolve a simple path: {@code a.b[0].c}. Returns NULL-ish missing as null.
     * Supports {@code .} for object keys and {@code [n]} for array indices.
     * Keys with dots must use bracket form: {@code ["a.b"]}.
     */
    public JsonValue at(String path) {
        if (path == null || path.isEmpty()) return this;
        return JsonPath.get(this, path);
    }

    public JsonValue at(String path, JsonValue defaultValue) {
        try {
            JsonValue v = at(path);
            return v == null || v.isNull() ? defaultValue : v;
        } catch (Exception e) {
            return defaultValue;
        }
    }

    // ---- conversion to Java ----

    /**
     * Convert to plain Java types: Map, List, String, Number, Boolean, null.
     * Nested structures are deep-converted. Numbers prefer Long/Integer/Double.
     */
    public Object toJava() {
        switch (type) {
            case NULL: return null;
            case BOOLEAN: return scalar;
            case STRING: return scalar;
            case NUMBER: return asNumber();
            case ARRAY: {
                List<Object> list = new ArrayList<>(array.size());
                for (JsonValue v : array) list.add(v.toJava());
                return list;
            }
            case OBJECT: {
                LinkedHashMap<String, Object> m = new LinkedHashMap<>();
                for (Map.Entry<String, JsonValue> e : object.entrySet()) {
                    m.put(e.getKey(), e.getValue().toJava());
                }
                return m;
            }
            default: return null;
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> toMap() {
        Object j = toJava();
        if (j == null) return new LinkedHashMap<>();
        if (j instanceof Map) return (Map<String, Object>) j;
        throw new JsonException("Not an object: " + type);
    }

    @SuppressWarnings("unchecked")
    public List<Object> toList() {
        Object j = toJava();
        if (j == null) return new ArrayList<>();
        if (j instanceof List) return (List<Object>) j;
        throw new JsonException("Not an array: " + type);
    }

    // ---- iteration ----

    @Override
    public Iterator<JsonValue> iterator() {
        if (type == Type.ARRAY) return array.iterator();
        if (type == Type.OBJECT) return object.values().iterator();
        return Collections.emptyIterator();
    }

    // ---- equality / hash / string ----

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof JsonValue)) return false;
        JsonValue other = (JsonValue) o;
        if (type != other.type) return false;
        switch (type) {
            case NULL: return true;
            case BOOLEAN:
            case STRING: return Objects.equals(scalar, other.scalar);
            case NUMBER: return Objects.equals(numberLex, other.numberLex)
                || (numberLex != null && other.numberLex != null
                    && new BigDecimal(numberLex).compareTo(new BigDecimal(other.numberLex)) == 0);
            case ARRAY: return Objects.equals(array, other.array);
            case OBJECT: return Objects.equals(object, other.object);
            default: return false;
        }
    }

    @Override
    public int hashCode() {
        switch (type) {
            case NULL: return 0;
            case BOOLEAN:
            case STRING: return Objects.hashCode(scalar);
            case NUMBER: return numberLex == null ? 0 : new BigDecimal(numberLex).stripTrailingZeros().hashCode();
            case ARRAY: return Objects.hashCode(array);
            case OBJECT: return Objects.hashCode(object);
            default: return 0;
        }
    }

    @Override
    public String toString() {
        return JsonWriter.toString(this, JsonWriteOptions.compact());
    }

    public String toPrettyString() {
        return JsonWriter.toString(this, JsonWriteOptions.prettyMode());
    }

    private void require(Type expected) {
        if (type != expected) {
            throw new JsonException("Expected " + expected + " but was " + type);
        }
    }
}
