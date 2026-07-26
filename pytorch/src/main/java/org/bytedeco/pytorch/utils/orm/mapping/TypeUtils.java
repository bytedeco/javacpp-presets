/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.orm.mapping;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.ZonedDateTime;
import java.util.Date;
import java.util.Locale;
import java.util.UUID;

/**
 * Coerce arbitrary {@link Object} values to a target Java type.
 *
 * <p>Annotation-free storch-tinyorm style conversion used by bean / map / ResultSet mappers.
 */
public final class TypeUtils {
    private TypeUtils() {}

    @SuppressWarnings("unchecked")
    public static <T> T convert(Object value, Class<T> targetType) {
        if (targetType == null) {
            throw new IllegalArgumentException("targetType required");
        }
        if (value == null) {
            return nullDefault(targetType);
        }
        Class<?> target = wrap(targetType);
        if (target.isInstance(value)) {
            return (T) value;
        }
        if (target == String.class) {
            return (T) toStringValue(value);
        }
        if (target == Boolean.class || target == boolean.class) {
            return (T) toBoolean(value);
        }
        if (Number.class.isAssignableFrom(target)
                || target == byte.class || target == short.class
                || target == int.class || target == long.class
                || target == float.class || target == double.class) {
            return (T) toNumber(value, target);
        }
        if (target == Character.class || target == char.class) {
            return (T) toCharacter(value);
        }
        if (target.isEnum()) {
            return (T) toEnum(value, target);
        }
        if (Date.class.isAssignableFrom(target)
                || target == LocalDate.class
                || target == LocalDateTime.class
                || target == LocalTime.class
                || target == Instant.class
                || target == ZonedDateTime.class
                || target == OffsetDateTime.class
                || target == java.sql.Date.class
                || target == java.sql.Time.class
                || target == Timestamp.class) {
            return (T) DateUtils.convert(value, target);
        }
        if (target == UUID.class) {
            return (T) toUuid(value);
        }
        if (target == byte[].class) {
            return (T) toBytes(value);
        }
        if (target.isArray()) {
            return (T) toArray(value, target);
        }
        // last resort: string parse for simple types already handled; fail clearly
        throw new IllegalArgumentException(
                "Cannot convert " + value.getClass().getName() + " to " + targetType.getName()
                        + " (value=" + value + ")");
    }

    public static Class<?> wrap(Class<?> type) {
        if (!type.isPrimitive()) return type;
        if (type == boolean.class) return Boolean.class;
        if (type == byte.class) return Byte.class;
        if (type == short.class) return Short.class;
        if (type == int.class) return Integer.class;
        if (type == long.class) return Long.class;
        if (type == float.class) return Float.class;
        if (type == double.class) return Double.class;
        if (type == char.class) return Character.class;
        if (type == void.class) return Void.class;
        return type;
    }

    public static Class<?> unwrap(Class<?> type) {
        if (type == Boolean.class) return boolean.class;
        if (type == Byte.class) return byte.class;
        if (type == Short.class) return short.class;
        if (type == Integer.class) return int.class;
        if (type == Long.class) return long.class;
        if (type == Float.class) return float.class;
        if (type == Double.class) return double.class;
        if (type == Character.class) return char.class;
        if (type == Void.class) return void.class;
        return type;
    }

    public static boolean isSimpleType(Class<?> type) {
        if (type == null) return false;
        Class<?> t = wrap(type);
        return t == String.class
                || Number.class.isAssignableFrom(t)
                || t == Boolean.class
                || t == Character.class
                || t.isEnum()
                || Date.class.isAssignableFrom(t)
                || t == LocalDate.class
                || t == LocalDateTime.class
                || t == LocalTime.class
                || t == Instant.class
                || t == ZonedDateTime.class
                || t == OffsetDateTime.class
                || t == UUID.class
                || t == byte[].class;
    }

    @SuppressWarnings("unchecked")
    private static <T> T nullDefault(Class<T> targetType) {
        if (!targetType.isPrimitive()) return null;
        if (targetType == boolean.class) return (T) Boolean.FALSE;
        if (targetType == char.class) return (T) Character.valueOf('\0');
        if (targetType == byte.class) return (T) Byte.valueOf((byte) 0);
        if (targetType == short.class) return (T) Short.valueOf((short) 0);
        if (targetType == int.class) return (T) Integer.valueOf(0);
        if (targetType == long.class) return (T) Long.valueOf(0L);
        if (targetType == float.class) return (T) Float.valueOf(0f);
        if (targetType == double.class) return (T) Double.valueOf(0d);
        return null;
    }

    private static String toStringValue(Object value) {
        if (value instanceof byte[]) {
            return new String((byte[]) value);
        }
        return String.valueOf(value);
    }

    private static Boolean toBoolean(Object value) {
        if (value instanceof Boolean) return (Boolean) value;
        if (value instanceof Number) return ((Number) value).intValue() != 0;
        String s = String.valueOf(value).trim();
        if (s.isEmpty()) return Boolean.FALSE;
        if ("1".equals(s) || "y".equalsIgnoreCase(s) || "yes".equalsIgnoreCase(s)
                || "t".equalsIgnoreCase(s) || "true".equalsIgnoreCase(s)
                || "on".equalsIgnoreCase(s)) {
            return Boolean.TRUE;
        }
        if ("0".equals(s) || "n".equalsIgnoreCase(s) || "no".equalsIgnoreCase(s)
                || "f".equalsIgnoreCase(s) || "false".equalsIgnoreCase(s)
                || "off".equalsIgnoreCase(s)) {
            return Boolean.FALSE;
        }
        return Boolean.parseBoolean(s);
    }

    private static Number toNumber(Object value, Class<?> target) {
        if (value instanceof Number) {
            return narrowNumber((Number) value, target);
        }
        if (value instanceof Boolean) {
            return narrowNumber((Boolean) value ? 1 : 0, target);
        }
        if (value instanceof Date) {
            return narrowNumber(((Date) value).getTime(), target);
        }
        if (value instanceof Instant) {
            return narrowNumber(((Instant) value).toEpochMilli(), target);
        }
        String s = String.valueOf(value).trim();
        if (s.isEmpty()) {
            return narrowNumber(0, target);
        }
        // strip common suffixes
        if (s.endsWith("L") || s.endsWith("l") || s.endsWith("F") || s.endsWith("f")
                || s.endsWith("D") || s.endsWith("d")) {
            s = s.substring(0, s.length() - 1);
        }
        try {
            if (target == BigDecimal.class) return new BigDecimal(s);
            if (target == BigInteger.class) return new BigInteger(s);
            if (s.contains(".") || s.contains("e") || s.contains("E")) {
                return narrowNumber(Double.parseDouble(s), target);
            }
            return narrowNumber(Long.parseLong(s), target);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Cannot convert '" + value + "' to " + target.getName(), e);
        }
    }

    private static Number narrowNumber(Number n, Class<?> target) {
        if (target == Byte.class || target == byte.class) return n.byteValue();
        if (target == Short.class || target == short.class) return n.shortValue();
        if (target == Integer.class || target == int.class) return n.intValue();
        if (target == Long.class || target == long.class) return n.longValue();
        if (target == Float.class || target == float.class) return n.floatValue();
        if (target == Double.class || target == double.class) return n.doubleValue();
        if (target == BigDecimal.class) {
            if (n instanceof BigDecimal) return n;
            if (n instanceof BigInteger) return new BigDecimal((BigInteger) n);
            if (n instanceof Float || n instanceof Double) return BigDecimal.valueOf(n.doubleValue());
            return BigDecimal.valueOf(n.longValue());
        }
        if (target == BigInteger.class) {
            if (n instanceof BigInteger) return n;
            if (n instanceof BigDecimal) return ((BigDecimal) n).toBigInteger();
            return BigInteger.valueOf(n.longValue());
        }
        return n.doubleValue();
    }

    private static Character toCharacter(Object value) {
        if (value instanceof Character) return (Character) value;
        if (value instanceof Number) return (char) ((Number) value).intValue();
        String s = String.valueOf(value);
        if (s.isEmpty()) return '\0';
        return s.charAt(0);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Enum<?> toEnum(Object value, Class<?> enumType) {
        if (value instanceof Enum) {
            return (Enum<?>) value;
        }
        if (value instanceof Number) {
            Enum<?>[] constants = (Enum<?>[]) enumType.getEnumConstants();
            int idx = ((Number) value).intValue();
            if (idx < 0 || idx >= constants.length) {
                throw new IllegalArgumentException("Enum ordinal out of range: " + idx + " for " + enumType.getName());
            }
            return constants[idx];
        }
        String name = String.valueOf(value).trim();
        try {
            return Enum.valueOf((Class<? extends Enum>) enumType, name);
        } catch (IllegalArgumentException first) {
            // try case-insensitive
            for (Object c : enumType.getEnumConstants()) {
                if (((Enum<?>) c).name().equalsIgnoreCase(name)) {
                    return (Enum<?>) c;
                }
            }
            throw first;
        }
    }

    private static UUID toUuid(Object value) {
        if (value instanceof UUID) return (UUID) value;
        if (value instanceof byte[]) {
            byte[] b = (byte[]) value;
            if (b.length == 16) {
                long msb = 0;
                long lsb = 0;
                for (int i = 0; i < 8; i++) msb = (msb << 8) | (b[i] & 0xff);
                for (int i = 8; i < 16; i++) lsb = (lsb << 8) | (b[i] & 0xff);
                return new UUID(msb, lsb);
            }
        }
        return UUID.fromString(String.valueOf(value).trim());
    }

    private static byte[] toBytes(Object value) {
        if (value instanceof byte[]) return (byte[]) value;
        if (value instanceof Byte[]) {
            Byte[] arr = (Byte[]) value;
            byte[] out = new byte[arr.length];
            for (int i = 0; i < arr.length; i++) out[i] = arr[i] == null ? 0 : arr[i];
            return out;
        }
        return String.valueOf(value).getBytes();
    }

    private static Object toArray(Object value, Class<?> arrayType) {
        Class<?> component = arrayType.getComponentType();
        if (value.getClass().isArray()) {
            int len = Array.getLength(value);
            Object out = Array.newInstance(component, len);
            for (int i = 0; i < len; i++) {
                Array.set(out, i, convert(Array.get(value, i), component));
            }
            return out;
        }
        if (value instanceof Iterable) {
            java.util.List<Object> list = new java.util.ArrayList<>();
            for (Object o : (Iterable<?>) value) list.add(o);
            Object out = Array.newInstance(component, list.size());
            for (int i = 0; i < list.size(); i++) {
                Array.set(out, i, convert(list.get(i), component));
            }
            return out;
        }
        // single value → single-element array
        Object out = Array.newInstance(component, 1);
        Array.set(out, 0, convert(value, component));
        return out;
    }

    /** Infer a JDBC/SQLite column SQL type from a Java property type. */
    public static String sqlTypeOf(Class<?> javaType) {
        if (javaType == null) return "TEXT";
        Class<?> t = wrap(javaType);
        if (t == Boolean.class) return "INTEGER";
        if (t == Byte.class || t == Short.class || t == Integer.class || t == Long.class
                || t == BigInteger.class) {
            return "INTEGER";
        }
        if (t == Float.class || t == Double.class || t == BigDecimal.class) return "REAL";
        if (t == byte[].class) return "BLOB";
        if (Date.class.isAssignableFrom(t)
                || t == LocalDate.class
                || t == LocalDateTime.class
                || t == LocalTime.class
                || t == Instant.class
                || t == Timestamp.class) {
            return "TEXT";
        }
        return "TEXT";
    }

    /** Snake_case conversion helper used by naming strategies. */
    public static String camelToSnake(String name) {
        if (name == null || name.isEmpty()) return name;
        StringBuilder sb = new StringBuilder(name.length() + 4);
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isUpperCase(c)) {
                if (i > 0) sb.append('_');
                sb.append(Character.toLowerCase(c));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    public static String snakeToCamel(String name) {
        if (name == null || name.isEmpty()) return name;
        StringBuilder sb = new StringBuilder(name.length());
        boolean up = false;
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (c == '_' || c == '-') {
                up = true;
                continue;
            }
            if (up) {
                sb.append(Character.toUpperCase(c));
                up = false;
            } else {
                sb.append(i == 0 ? Character.toLowerCase(c) : c);
            }
        }
        return sb.toString();
    }

    public static String capitalize(String s) {
        if (s == null || s.isEmpty()) return s;
        return Character.toUpperCase(s.charAt(0)) + s.substring(1);
    }

    public static String decapitalize(String s) {
        if (s == null || s.isEmpty()) return s;
        return Character.toLowerCase(s.charAt(0)) + s.substring(1);
    }

    public static String lower(String s) {
        return s == null ? null : s.toLowerCase(Locale.ROOT);
    }
}
