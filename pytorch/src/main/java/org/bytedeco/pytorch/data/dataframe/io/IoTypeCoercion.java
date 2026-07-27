package org.bytedeco.pytorch.data.dataframe.io;

import org.bytedeco.pytorch.data.dataframe.Column;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Shared type inference and value coercion used by tabular I/O formats.
 */
public final class IoTypeCoercion {
    private IoTypeCoercion() {}

    private static final DateTimeFormatter[] DATE_FMTS = {
        DateTimeFormatter.ISO_LOCAL_DATE,
        DateTimeFormatter.ofPattern("yyyy/MM/dd"),
        DateTimeFormatter.ofPattern("MM/dd/yyyy"),
        DateTimeFormatter.ofPattern("dd-MM-yyyy")
    };

    private static final DateTimeFormatter[] DATETIME_FMTS = {
        DateTimeFormatter.ISO_LOCAL_DATE_TIME,
        DateTimeFormatter.ISO_OFFSET_DATE_TIME,
        DateTimeFormatter.ISO_INSTANT,
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
        DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss"),
        DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS")
    };

    public static Column.DType inferFromObject(Object v) {
        if (v == null) return Column.DType.STRING;
        if (v instanceof Boolean) return Column.DType.BOOLEAN;
        if (v instanceof Byte || v instanceof Short || v instanceof Integer) return Column.DType.INT32;
        if (v instanceof Long) return Column.DType.INT64;
        if (v instanceof Float) return Column.DType.FLOAT32;
        if (v instanceof Double || v instanceof Number) return Column.DType.FLOAT64;
        if (v instanceof LocalDate) return Column.DType.DATE;
        if (v instanceof LocalDateTime || v instanceof Instant) return Column.DType.DATETIME;
        if (v instanceof LocalTime) return Column.DType.TIME;
        if (v instanceof float[] || v instanceof double[]) return Column.DType.VECTOR;
        if (v instanceof int[] || v instanceof long[] || v instanceof boolean[]) return Column.DType.LIST;
        if (v instanceof byte[]) return Column.DType.BINARY;
        if (v instanceof Map) return Column.DType.MAP;
        if (v instanceof List || v instanceof Collection) {
            // Delegate nested list inference to ComplexCellCodec (VECTOR vs LIST)
            return ComplexCellCodec.inferComplex(v);
        }
        if (v instanceof CharSequence) {
            // Detect JSON-encoded nested cells stored as text
            Column.DType nested = ComplexCellCodec.inferComplex(v);
            if (nested == Column.DType.LIST || nested == Column.DType.MAP
                || nested == Column.DType.VECTOR || nested == Column.DType.JSON) {
                return nested;
            }
            return Column.DType.STRING;
        }
        return Column.DType.STRING;
    }

    public static Column.DType inferFromStrings(Collection<String> values, Set<String> nullTokens) {
        boolean canBool = true, canLong = true, canDouble = true, canDate = true, canDateTime = true;
        int nonNull = 0;
        for (String raw : values) {
            if (IoNullTokens.isNull(raw, nullTokens)) continue;
            nonNull++;
            String t = raw.trim();
            if (canBool && !isBoolean(t)) canBool = false;
            if (canLong && !isLong(t)) canLong = false;
            if (canDouble && !isDouble(t)) canDouble = false;
            if (canDate && !isDate(t)) canDate = false;
            if (canDateTime && !isDateTime(t)) canDateTime = false;
        }
        if (nonNull == 0) return Column.DType.STRING;
        if (canBool) return Column.DType.BOOLEAN;
        if (canLong) return Column.DType.INT64;
        if (canDouble) return Column.DType.FLOAT64;
        if (canDateTime) return Column.DType.DATETIME;
        if (canDate) return Column.DType.DATE;
        return Column.DType.STRING;
    }

    public static Object coerce(Object raw, Column.DType dtype) {
        if (raw == null) return null;
        if (dtype == null) return raw;
        switch (dtype) {
            case INT32:
                if (raw instanceof Number) return ((Number) raw).intValue();
                return Integer.parseInt(String.valueOf(raw).trim());
            case INT64:
                if (raw instanceof Number) return ((Number) raw).longValue();
                return Long.parseLong(String.valueOf(raw).trim());
            case FLOAT32:
                if (raw instanceof Number) return ((Number) raw).floatValue();
                return Float.parseFloat(String.valueOf(raw).trim());
            case FLOAT64:
                if (raw instanceof Number) return ((Number) raw).doubleValue();
                return Double.parseDouble(String.valueOf(raw).trim());
            case BOOLEAN:
                if (raw instanceof Boolean) return raw;
                return parseBoolean(String.valueOf(raw).trim());
            case DATE:
                if (raw instanceof LocalDate) return raw;
                if (raw instanceof LocalDateTime) return ((LocalDateTime) raw).toLocalDate();
                if (raw instanceof Instant) return LocalDate.ofInstant((Instant) raw, ZoneOffset.UTC);
                if (raw instanceof Number) {
                    return Instant.ofEpochMilli(((Number) raw).longValue()).atZone(ZoneOffset.UTC).toLocalDate();
                }
                return parseDate(String.valueOf(raw).trim());
            case DATETIME:
                if (raw instanceof LocalDateTime) return raw;
                if (raw instanceof Instant) return LocalDateTime.ofInstant((Instant) raw, ZoneOffset.UTC);
                if (raw instanceof LocalDate) return ((LocalDate) raw).atStartOfDay();
                if (raw instanceof Number) {
                    return LocalDateTime.ofInstant(Instant.ofEpochMilli(((Number) raw).longValue()), ZoneOffset.UTC);
                }
                return parseDateTime(String.valueOf(raw).trim());
            case TIME:
                if (raw instanceof LocalTime) return raw;
                return LocalTime.parse(String.valueOf(raw).trim());
            case STRING:
                return String.valueOf(raw);
            case VECTOR:
            case EMBEDDING:
                return ComplexCellCodec.coerceComplex(raw, Column.DType.VECTOR);
            case LIST:
                return ComplexCellCodec.coerceComplex(raw, Column.DType.LIST);
            case MAP:
            case STRUCT:
                return ComplexCellCodec.coerceComplex(raw, dtype);
            case JSON:
                return ComplexCellCodec.coerceComplex(raw, Column.DType.JSON);
            case BINARY:
                if (raw instanceof byte[]) return raw;
                return String.valueOf(raw).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            default:
                // Keep already-materialized nested values as-is.
                if (raw instanceof Map || raw instanceof List || raw instanceof Collection
                    || raw instanceof float[] || raw instanceof double[]
                    || raw instanceof int[] || raw instanceof long[]) {
                    return raw;
                }
                return raw;
        }
    }

    public static Object parseString(String raw, Column.DType dtype, Set<String> nullTokens) {
        if (IoNullTokens.isNull(raw, nullTokens)) return null;
        return coerce(raw.trim(), dtype);
    }

    public static boolean isBoolean(String s) {
        String t = s.trim().toLowerCase(Locale.ROOT);
        return "true".equals(t) || "false".equals(t) || "1".equals(t) || "0".equals(t)
            || "yes".equals(t) || "no".equals(t) || "y".equals(t) || "n".equals(t)
            || "t".equals(t) || "f".equals(t);
    }

    public static Boolean parseBoolean(String s) {
        String t = s.trim().toLowerCase(Locale.ROOT);
        if ("1".equals(t) || "yes".equals(t) || "y".equals(t) || "t".equals(t) || "true".equals(t)) return true;
        if ("0".equals(t) || "no".equals(t) || "n".equals(t) || "f".equals(t) || "false".equals(t)) return false;
        return Boolean.parseBoolean(t);
    }

    public static boolean isLong(String s) {
        try {
            Long.parseLong(s.trim());
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public static boolean isDouble(String s) {
        try {
            if (s == null || s.trim().isEmpty()) return false;
            Double.parseDouble(s.trim());
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public static boolean isDate(String s) {
        for (DateTimeFormatter f : DATE_FMTS) {
            try {
                LocalDate.parse(s.trim(), f);
                return true;
            } catch (DateTimeParseException ignored) {}
        }
        return false;
    }

    public static boolean isDateTime(String s) {
        String t = s.trim();
        for (DateTimeFormatter f : DATETIME_FMTS) {
            try {
                if (f == DateTimeFormatter.ISO_INSTANT) {
                    Instant.parse(t);
                } else {
                    LocalDateTime.parse(t, f);
                }
                return true;
            } catch (Exception ignored) {}
        }
        return false;
    }

    public static LocalDate parseDate(String s) {
        for (DateTimeFormatter f : DATE_FMTS) {
            try {
                return LocalDate.parse(s, f);
            } catch (DateTimeParseException ignored) {}
        }
        throw new IllegalArgumentException("Cannot parse DATE: " + s);
    }

    public static LocalDateTime parseDateTime(String s) {
        for (DateTimeFormatter f : DATETIME_FMTS) {
            try {
                if (f == DateTimeFormatter.ISO_INSTANT) {
                    return LocalDateTime.ofInstant(Instant.parse(s), ZoneOffset.UTC);
                }
                if (f == DateTimeFormatter.ISO_OFFSET_DATE_TIME) {
                    return LocalDateTime.parse(s, f);
                }
                return LocalDateTime.parse(s, f);
            } catch (Exception ignored) {}
        }
        // date-only fallback
        try {
            return parseDate(s).atStartOfDay();
        } catch (Exception ignored) {}
        throw new IllegalArgumentException("Cannot parse DATETIME: " + s);
    }

    public static float[] parseVector(String s) {
        String t = s.trim();
        if (t.startsWith("[") && t.endsWith("]")) t = t.substring(1, t.length() - 1).trim();
        if (t.isEmpty()) return new float[0];
        String[] parts = t.split("[,;\\s]+");
        float[] out = new float[parts.length];
        int n = 0;
        for (String p : parts) {
            if (p.isEmpty()) continue;
            out[n++] = Float.parseFloat(p);
        }
        if (n == out.length) return out;
        return java.util.Arrays.copyOf(out, n);
    }

    /** Widen two dtypes for schema unification (e.g. INT64 + FLOAT64 → FLOAT64). */
    public static Column.DType widen(Column.DType a, Column.DType b) {
        if (a == null) return b;
        if (b == null) return a;
        if (a == b) return a;
        if (a == Column.DType.STRING || b == Column.DType.STRING) return Column.DType.STRING;
        if (isFloat(a) || isFloat(b)) return Column.DType.FLOAT64;
        if (isInt(a) && isInt(b)) return Column.DType.INT64;
        if ((isInt(a) || isFloat(a)) && (isInt(b) || isFloat(b))) return Column.DType.FLOAT64;
        if (a == Column.DType.DATE && b == Column.DType.DATETIME) return Column.DType.DATETIME;
        if (a == Column.DType.DATETIME && b == Column.DType.DATE) return Column.DType.DATETIME;
        return Column.DType.STRING;
    }

    public static boolean isNumeric(Column.DType d) {
        return d == Column.DType.INT32 || d == Column.DType.INT64
            || d == Column.DType.FLOAT32 || d == Column.DType.FLOAT64;
    }

    private static boolean isFloat(Column.DType d) {
        return d == Column.DType.FLOAT32 || d == Column.DType.FLOAT64;
    }

    private static boolean isInt(Column.DType d) {
        return d == Column.DType.INT32 || d == Column.DType.INT64;
    }
}
