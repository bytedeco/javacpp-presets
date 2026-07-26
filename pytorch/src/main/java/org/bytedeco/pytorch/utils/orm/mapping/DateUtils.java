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

import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Date;

/**
 * Parse / format common date and time types for ORM mapping.
 */
public final class DateUtils {
    private DateUtils() {}

    public static final DateTimeFormatter ISO_LOCAL_DATE = DateTimeFormatter.ISO_LOCAL_DATE;
    public static final DateTimeFormatter ISO_LOCAL_TIME = DateTimeFormatter.ISO_LOCAL_TIME;
    public static final DateTimeFormatter ISO_LOCAL_DATE_TIME = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
    public static final DateTimeFormatter ISO_OFFSET_DATE_TIME = DateTimeFormatter.ISO_OFFSET_DATE_TIME;
    public static final DateTimeFormatter ISO_INSTANT = DateTimeFormatter.ISO_INSTANT;

    private static final DateTimeFormatter[] DATE_PATTERNS = {
            DateTimeFormatter.ISO_LOCAL_DATE,
            DateTimeFormatter.ofPattern("yyyy/MM/dd"),
            DateTimeFormatter.ofPattern("yyyyMMdd"),
            DateTimeFormatter.ofPattern("dd-MM-yyyy"),
            DateTimeFormatter.ofPattern("MM/dd/yyyy"),
    };

    private static final DateTimeFormatter[] DATE_TIME_PATTERNS = {
            DateTimeFormatter.ISO_LOCAL_DATE_TIME,
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS"),
            DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyyMMddHHmmss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
    };

    private static final DateTimeFormatter[] TIME_PATTERNS = {
            DateTimeFormatter.ISO_LOCAL_TIME,
            DateTimeFormatter.ofPattern("HH:mm:ss"),
            DateTimeFormatter.ofPattern("HH:mm:ss.SSS"),
            DateTimeFormatter.ofPattern("HHmmss"),
    };

    @SuppressWarnings("unchecked")
    public static <T> T convert(Object value, Class<T> target) {
        if (value == null) return null;
        Class<?> t = TypeUtils.wrap(target);

        if (t.isInstance(value)) {
            return (T) value;
        }

        if (t == Date.class) {
            return (T) toDate(value);
        }
        if (t == java.sql.Date.class) {
            Date d = toDate(value);
            return d == null ? null : (T) new java.sql.Date(d.getTime());
        }
        if (t == java.sql.Time.class) {
            LocalTime lt = toLocalTime(value);
            return lt == null ? null : (T) java.sql.Time.valueOf(lt);
        }
        if (t == Timestamp.class) {
            return (T) toTimestamp(value);
        }
        if (t == LocalDate.class) {
            return (T) toLocalDate(value);
        }
        if (t == LocalDateTime.class) {
            return (T) toLocalDateTime(value);
        }
        if (t == LocalTime.class) {
            return (T) toLocalTime(value);
        }
        if (t == Instant.class) {
            return (T) toInstant(value);
        }
        if (t == ZonedDateTime.class) {
            Instant inst = toInstant(value);
            return inst == null ? null : (T) inst.atZone(ZoneId.systemDefault());
        }
        if (t == OffsetDateTime.class) {
            Instant inst = toInstant(value);
            return inst == null ? null : (T) OffsetDateTime.ofInstant(inst, ZoneId.systemDefault());
        }
        throw new IllegalArgumentException("Unsupported date target: " + target.getName());
    }

    public static Date toDate(Object value) {
        if (value == null) return null;
        if (value instanceof Date) return (Date) value;
        if (value instanceof Number) return new Date(((Number) value).longValue());
        if (value instanceof Instant) return Date.from((Instant) value);
        if (value instanceof LocalDateTime) {
            return Date.from(((LocalDateTime) value).atZone(ZoneId.systemDefault()).toInstant());
        }
        if (value instanceof LocalDate) {
            return Date.from(((LocalDate) value).atStartOfDay(ZoneId.systemDefault()).toInstant());
        }
        if (value instanceof ZonedDateTime) return Date.from(((ZonedDateTime) value).toInstant());
        if (value instanceof OffsetDateTime) return Date.from(((OffsetDateTime) value).toInstant());
        if (value instanceof LocalTime) {
            return Date.from(((LocalTime) value).atDate(LocalDate.ofEpochDay(0))
                    .atZone(ZoneId.systemDefault()).toInstant());
        }
        Instant inst = parseInstant(String.valueOf(value).trim());
        return inst == null ? null : Date.from(inst);
    }

    public static Timestamp toTimestamp(Object value) {
        if (value == null) return null;
        if (value instanceof Timestamp) return (Timestamp) value;
        Date d = toDate(value);
        return d == null ? null : new Timestamp(d.getTime());
    }

    public static LocalDate toLocalDate(Object value) {
        if (value == null) return null;
        if (value instanceof LocalDate) return (LocalDate) value;
        if (value instanceof java.sql.Date) return ((java.sql.Date) value).toLocalDate();
        if (value instanceof LocalDateTime) return ((LocalDateTime) value).toLocalDate();
        if (value instanceof Timestamp) return ((Timestamp) value).toLocalDateTime().toLocalDate();
        if (value instanceof Date) {
            return Instant.ofEpochMilli(((Date) value).getTime()).atZone(ZoneId.systemDefault()).toLocalDate();
        }
        if (value instanceof Instant) {
            return ((Instant) value).atZone(ZoneId.systemDefault()).toLocalDate();
        }
        if (value instanceof Number) {
            return Instant.ofEpochMilli(((Number) value).longValue())
                    .atZone(ZoneId.systemDefault()).toLocalDate();
        }
        String s = String.valueOf(value).trim();
        if (s.isEmpty()) return null;
        for (DateTimeFormatter f : DATE_PATTERNS) {
            try {
                return LocalDate.parse(s, f);
            } catch (DateTimeParseException ignored) {
            }
        }
        // maybe datetime string
        LocalDateTime ldt = tryParseLocalDateTime(s);
        if (ldt != null) return ldt.toLocalDate();
        throw new IllegalArgumentException("Cannot parse LocalDate from: " + s);
    }

    public static LocalDateTime toLocalDateTime(Object value) {
        if (value == null) return null;
        if (value instanceof LocalDateTime) return (LocalDateTime) value;
        if (value instanceof Timestamp) return ((Timestamp) value).toLocalDateTime();
        if (value instanceof java.sql.Date) return ((java.sql.Date) value).toLocalDate().atStartOfDay();
        if (value instanceof LocalDate) return ((LocalDate) value).atStartOfDay();
        if (value instanceof Date) {
            return Instant.ofEpochMilli(((Date) value).getTime())
                    .atZone(ZoneId.systemDefault()).toLocalDateTime();
        }
        if (value instanceof Instant) {
            return LocalDateTime.ofInstant((Instant) value, ZoneId.systemDefault());
        }
        if (value instanceof ZonedDateTime) return ((ZonedDateTime) value).toLocalDateTime();
        if (value instanceof OffsetDateTime) return ((OffsetDateTime) value).toLocalDateTime();
        if (value instanceof Number) {
            return LocalDateTime.ofInstant(Instant.ofEpochMilli(((Number) value).longValue()),
                    ZoneId.systemDefault());
        }
        String s = String.valueOf(value).trim();
        if (s.isEmpty()) return null;
        LocalDateTime ldt = tryParseLocalDateTime(s);
        if (ldt != null) return ldt;
        LocalDate ld = null;
        try {
            ld = toLocalDate(s);
        } catch (Exception ignored) {
        }
        if (ld != null) return ld.atStartOfDay();
        throw new IllegalArgumentException("Cannot parse LocalDateTime from: " + s);
    }

    public static LocalTime toLocalTime(Object value) {
        if (value == null) return null;
        if (value instanceof LocalTime) return (LocalTime) value;
        if (value instanceof java.sql.Time) return ((java.sql.Time) value).toLocalTime();
        if (value instanceof LocalDateTime) return ((LocalDateTime) value).toLocalTime();
        if (value instanceof Timestamp) return ((Timestamp) value).toLocalDateTime().toLocalTime();
        if (value instanceof Date) {
            return Instant.ofEpochMilli(((Date) value).getTime())
                    .atZone(ZoneId.systemDefault()).toLocalTime();
        }
        if (value instanceof Number) {
            // treat as millis of day or epoch? epoch millis → local time
            return Instant.ofEpochMilli(((Number) value).longValue())
                    .atZone(ZoneId.systemDefault()).toLocalTime();
        }
        String s = String.valueOf(value).trim();
        if (s.isEmpty()) return null;
        for (DateTimeFormatter f : TIME_PATTERNS) {
            try {
                return LocalTime.parse(s, f);
            } catch (DateTimeParseException ignored) {
            }
        }
        LocalDateTime ldt = tryParseLocalDateTime(s);
        if (ldt != null) return ldt.toLocalTime();
        throw new IllegalArgumentException("Cannot parse LocalTime from: " + s);
    }

    public static Instant toInstant(Object value) {
        if (value == null) return null;
        if (value instanceof Instant) return (Instant) value;
        if (value instanceof Date) return ((Date) value).toInstant();
        if (value instanceof Timestamp) return ((Timestamp) value).toInstant();
        if (value instanceof LocalDateTime) {
            return ((LocalDateTime) value).atZone(ZoneId.systemDefault()).toInstant();
        }
        if (value instanceof LocalDate) {
            return ((LocalDate) value).atStartOfDay(ZoneId.systemDefault()).toInstant();
        }
        if (value instanceof ZonedDateTime) return ((ZonedDateTime) value).toInstant();
        if (value instanceof OffsetDateTime) return ((OffsetDateTime) value).toInstant();
        if (value instanceof Number) return Instant.ofEpochMilli(((Number) value).longValue());
        return parseInstant(String.valueOf(value).trim());
    }

    public static String format(Object value) {
        if (value == null) return null;
        if (value instanceof LocalDate) return ISO_LOCAL_DATE.format((LocalDate) value);
        if (value instanceof LocalTime) return ISO_LOCAL_TIME.format((LocalTime) value);
        if (value instanceof LocalDateTime) return ISO_LOCAL_DATE_TIME.format((LocalDateTime) value);
        if (value instanceof Instant) return ISO_INSTANT.format((Instant) value);
        if (value instanceof OffsetDateTime) return ISO_OFFSET_DATE_TIME.format((OffsetDateTime) value);
        if (value instanceof ZonedDateTime) {
            return ISO_OFFSET_DATE_TIME.format(((ZonedDateTime) value).toOffsetDateTime());
        }
        if (value instanceof Timestamp) {
            return ISO_LOCAL_DATE_TIME.format(((Timestamp) value).toLocalDateTime());
        }
        if (value instanceof java.sql.Date) {
            return ISO_LOCAL_DATE.format(((java.sql.Date) value).toLocalDate());
        }
        if (value instanceof java.sql.Time) {
            return ISO_LOCAL_TIME.format(((java.sql.Time) value).toLocalTime());
        }
        if (value instanceof Date) {
            return ISO_INSTANT.format(((Date) value).toInstant());
        }
        return String.valueOf(value);
    }

    private static LocalDateTime tryParseLocalDateTime(String s) {
        for (DateTimeFormatter f : DATE_TIME_PATTERNS) {
            try {
                return LocalDateTime.parse(s, f);
            } catch (DateTimeParseException ignored) {
            }
        }
        // try with zone / offset
        try {
            return OffsetDateTime.parse(s).toLocalDateTime();
        } catch (DateTimeParseException ignored) {
        }
        try {
            return ZonedDateTime.parse(s).toLocalDateTime();
        } catch (DateTimeParseException ignored) {
        }
        try {
            return LocalDateTime.ofInstant(Instant.parse(s), ZoneId.systemDefault());
        } catch (DateTimeParseException ignored) {
        }
        return null;
    }

    private static Instant parseInstant(String s) {
        if (s == null || s.isEmpty()) return null;
        // epoch millis as string
        if (s.chars().allMatch(ch -> ch == '-' || Character.isDigit(ch))) {
            try {
                return Instant.ofEpochMilli(Long.parseLong(s));
            } catch (NumberFormatException ignored) {
            }
        }
        try {
            return Instant.parse(s);
        } catch (DateTimeParseException ignored) {
        }
        LocalDateTime ldt = tryParseLocalDateTime(s);
        if (ldt != null) {
            return ldt.atZone(ZoneId.systemDefault()).toInstant();
        }
        try {
            LocalDate ld = toLocalDate(s);
            return ld.atStartOfDay(ZoneId.systemDefault()).toInstant();
        } catch (Exception ignored) {
        }
        throw new IllegalArgumentException("Cannot parse Instant/Date from: " + s);
    }
}
