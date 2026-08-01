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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Shared map-parse helpers for factory hparams (snake_case HF / LLaMA-Factory keys).
 */
final class HparamsMaps {
    private HparamsMaps() {}

    static Object get(Map<String, ?> m, String... keys) {
        if (m == null) {
            return null;
        }
        for (String k : keys) {
            if (m.containsKey(k)) {
                return m.get(k);
            }
        }
        return null;
    }

    static String str(Map<String, ?> m, String def, String... keys) {
        Object v = get(m, keys);
        if (v == null) {
            return def;
        }
        String s = String.valueOf(v).trim();
        return s.isEmpty() && def != null ? def : s;
    }

    static String strOrNull(Map<String, ?> m, String... keys) {
        Object v = get(m, keys);
        if (v == null) {
            return null;
        }
        String s = String.valueOf(v).trim();
        return s.isEmpty() || "null".equalsIgnoreCase(s) ? null : s;
    }

    static int integer(Map<String, ?> m, int def, String... keys) {
        Object v = get(m, keys);
        if (v == null) {
            return def;
        }
        if (v instanceof Number n) {
            return n.intValue();
        }
        String s = String.valueOf(v).trim();
        if (s.isEmpty()) {
            return def;
        }
        return Integer.parseInt(s);
    }

    static long lng(Map<String, ?> m, long def, String... keys) {
        Object v = get(m, keys);
        if (v == null) {
            return def;
        }
        if (v instanceof Number n) {
            return n.longValue();
        }
        String s = String.valueOf(v).trim();
        if (s.isEmpty()) {
            return def;
        }
        return Long.parseLong(s);
    }

    static double dbl(Map<String, ?> m, double def, String... keys) {
        Object v = get(m, keys);
        if (v == null) {
            return def;
        }
        if (v instanceof Number n) {
            return n.doubleValue();
        }
        String s = String.valueOf(v).trim();
        if (s.isEmpty()) {
            return def;
        }
        return Double.parseDouble(s);
    }

    static boolean bool(Map<String, ?> m, boolean def, String... keys) {
        Object v = get(m, keys);
        if (v == null) {
            return def;
        }
        if (v instanceof Boolean b) {
            return b;
        }
        if (v instanceof Number n) {
            return n.intValue() != 0;
        }
        String s = String.valueOf(v).trim().toLowerCase(Locale.ROOT);
        if (s.isEmpty()) {
            return def;
        }
        return switch (s) {
            case "1", "true", "yes", "y", "on" -> true;
            case "0", "false", "no", "n", "off" -> false;
            default -> Boolean.parseBoolean(s);
        };
    }

    @SuppressWarnings("unchecked")
    static List<String> stringList(Map<String, ?> m, String... keys) {
        Object v = get(m, keys);
        if (v == null) {
            return Collections.emptyList();
        }
        if (v instanceof List<?> list) {
            List<String> out = new ArrayList<>(list.size());
            for (Object o : list) {
                if (o != null) {
                    String s = String.valueOf(o).trim();
                    if (!s.isEmpty()) {
                        out.add(s);
                    }
                }
            }
            return out;
        }
        String s = String.valueOf(v).trim();
        if (s.isEmpty()) {
            return Collections.emptyList();
        }
        String[] parts = s.split(",");
        List<String> out = new ArrayList<>(parts.length);
        for (String p : parts) {
            String t = p.trim();
            if (!t.isEmpty()) {
                out.add(t);
            }
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asMap(Object v) {
        if (v instanceof Map<?, ?> raw) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : raw.entrySet()) {
                if (e.getKey() != null) {
                    out.put(String.valueOf(e.getKey()), e.getValue());
                }
            }
            return out;
        }
        return null;
    }

    static void put(Map<String, Object> m, String key, Object value) {
        m.put(key, value);
    }

    static String requireNonBlank(String name, String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must be non-blank");
        }
        return value;
    }

    static <T> T requireNonNull(String name, T value) {
        return Objects.requireNonNull(value, name + " must not be null");
    }
}
