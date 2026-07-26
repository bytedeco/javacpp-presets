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
package org.bytedeco.pytorch.utils.tokenizers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Safe cast helpers over {@link org.bytedeco.pytorch.utils.json.Json#decodeObject}.
 */
public final class JsonMaps {

    private JsonMaps() {}

    @SuppressWarnings("unchecked")
    public static Map<String, Object> asMap(Object o) {
        if (o == null) return null;
        if (o instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                out.put(String.valueOf(e.getKey()), e.getValue());
            }
            return out;
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    public static List<Object> asList(Object o) {
        if (o == null) return null;
        if (o instanceof List<?> list) {
            return new ArrayList<>((List<Object>) list);
        }
        return null;
    }

    public static String asString(Object o) {
        if (o == null) return null;
        if (o instanceof String s) return s;
        return String.valueOf(o);
    }

    public static String asString(Map<String, Object> m, String key) {
        if (m == null) return null;
        return asString(m.get(key));
    }

    /** HF special token fields may be a string or {@code {"content":"..."}}. */
    public static String asTokenString(Object o) {
        if (o == null) return null;
        if (o instanceof String s) return s;
        Map<String, Object> m = asMap(o);
        if (m != null) {
            Object c = m.get("content");
            if (c != null) return asString(c);
            Object id = m.get("id");
            if (id != null) return asString(id);
        }
        return asString(o);
    }

    public static Integer asInt(Object o) {
        if (o == null) return null;
        if (o instanceof Number n) return n.intValue();
        if (o instanceof String s) {
            try {
                return Integer.parseInt(s.trim());
            } catch (NumberFormatException e) {
                return null;
            }
        }
        return null;
    }

    public static Integer asInt(Map<String, Object> m, String key) {
        if (m == null) return null;
        return asInt(m.get(key));
    }

    public static Long asLong(Object o) {
        if (o == null) return null;
        if (o instanceof Number n) return n.longValue();
        if (o instanceof String s) {
            try {
                return Long.parseLong(s.trim());
            } catch (NumberFormatException e) {
                return null;
            }
        }
        return null;
    }

    public static Double asDouble(Object o) {
        if (o == null) return null;
        if (o instanceof Number n) return n.doubleValue();
        if (o instanceof String s) {
            try {
                return Double.parseDouble(s.trim());
            } catch (NumberFormatException e) {
                return null;
            }
        }
        return null;
    }

    public static boolean asBoolean(Object o, boolean defaultValue) {
        if (o == null) return defaultValue;
        if (o instanceof Boolean b) return b;
        if (o instanceof String s) {
            if ("true".equalsIgnoreCase(s)) return true;
            if ("false".equalsIgnoreCase(s)) return false;
        }
        if (o instanceof Number n) return n.intValue() != 0;
        return defaultValue;
    }

    public static boolean asBoolean(Map<String, Object> m, String key, boolean defaultValue) {
        if (m == null) return defaultValue;
        return asBoolean(m.get(key), defaultValue);
    }

    public static String requireType(Map<String, Object> m) {
        Objects.requireNonNull(m, "component map");
        String t = asString(m.get("type"));
        if (t == null || t.isEmpty()) {
            throw new IllegalArgumentException("Missing type in tokenizer component: " + m.keySet());
        }
        return t;
    }

    public static Map<String, Integer> asStringIntMap(Object o) {
        Map<String, Object> m = asMap(o);
        if (m == null) return Collections.emptyMap();
        Map<String, Integer> out = new LinkedHashMap<>(m.size() * 2);
        for (Map.Entry<String, Object> e : m.entrySet()) {
            Integer id = asInt(e.getValue());
            if (id != null) out.put(e.getKey(), id);
        }
        return out;
    }
}
