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
import org.bytedeco.pytorch.jit.*;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

/**
 * Convert a {@link Map} of property values into a JavaBean via no-arg constructor
 * and setters / public fields. Annotation-free, storch-tinyorm style.
 */
public final class MapToBeanMapper {
    private MapToBeanMapper() {}

    public static <T> T toBean(Map<String, ?> map, Class<T> type) {
        return toBean(map, type, NamingStrategy.IDENTITY);
    }

    public static <T> T toBean(Map<String, ?> map, Class<T> type, NamingStrategy naming) {
        if (type == null) throw new IllegalArgumentException("type required");
        if (map == null) return null;
        NamingStrategy strategy = naming == null ? NamingStrategy.IDENTITY : naming;
        try {
            T bean = newInstance(type);
            apply(map, bean, strategy);
            return bean;
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to map to " + type.getName(), e);
        }
    }

    public static <T> List<T> toBeans(Iterable<? extends Map<String, ?>> maps, Class<T> type) {
        return toBeans(maps, type, NamingStrategy.IDENTITY);
    }

    public static <T> List<T> toBeans(Iterable<? extends Map<String, ?>> maps, Class<T> type,
                                      NamingStrategy naming) {
        if (maps == null) return Collections.emptyList();
        List<T> out = new ArrayList<>();
        for (Map<String, ?> m : maps) {
            out.add(toBean(m, type, naming));
        }
        return out;
    }

    public static void apply(Map<String, ?> map, Object bean) {
        apply(map, bean, NamingStrategy.IDENTITY);
    }

    public static void apply(Map<String, ?> map, Object bean, NamingStrategy naming) {
        if (map == null || bean == null) return;
        NamingStrategy strategy = naming == null ? NamingStrategy.IDENTITY : naming;
        List<BeanToMapMapper.PropertyAccess> props = BeanToMapMapper.propertiesOf(bean.getClass());

        // index map keys (case-insensitive + snake/camel variants)
        Map<String, Object> keyed = new LinkedHashMap<>();
        for (Map.Entry<String, ?> e : map.entrySet()) {
            if (e.getKey() == null) continue;
            keyed.put(e.getKey(), e.getValue());
            keyed.putIfAbsent(e.getKey().toLowerCase(Locale.ROOT), e.getValue());
        }

        for (BeanToMapMapper.PropertyAccess p : props) {
            if (!p.writable) continue;
            Object value = lookup(keyed, p.name, strategy);
            if (value == ABSENT) continue;
            try {
                p.write(bean, value);
            } catch (Exception e) {
                throw new IllegalStateException(
                        "Failed to set property '" + p.name + "' of " + bean.getClass().getName()
                                + " from value " + value, e);
            }
        }
    }

    private static final Object ABSENT = new Object();

    private static Object lookup(Map<String, Object> keyed, String property, NamingStrategy naming) {
        // direct
        if (keyed.containsKey(property)) return keyed.get(property);
        String lower = property.toLowerCase(Locale.ROOT);
        if (keyed.containsKey(lower)) return keyed.get(lower);

        // strategy column name → property
        String col = naming.toColumn(property);
        if (keyed.containsKey(col)) return keyed.get(col);
        String colLower = col.toLowerCase(Locale.ROOT);
        if (keyed.containsKey(colLower)) return keyed.get(colLower);

        // reverse: try snake → camel property matches map keys already tried
        String snake = TypeUtils.camelToSnake(property);
        if (keyed.containsKey(snake)) return keyed.get(snake);
        if (keyed.containsKey(snake.toLowerCase(Locale.ROOT))) {
            return keyed.get(snake.toLowerCase(Locale.ROOT));
        }

        // try matching map keys converted to property names
        for (Map.Entry<String, Object> e : keyed.entrySet()) {
            String k = e.getKey();
            if (k == null) continue;
            if (property.equalsIgnoreCase(naming.toProperty(k))) return e.getValue();
            if (property.equalsIgnoreCase(TypeUtils.snakeToCamel(k))) return e.getValue();
            if (property.equalsIgnoreCase(k)) return e.getValue();
        }
        return ABSENT;
    }

    public static <T> T newInstance(Class<T> type) throws Exception {
        Constructor<T> ctor = type.getDeclaredConstructor();
        if (!ctor.canAccess(null)) ctor.setAccessible(true);
        return ctor.newInstance();
    }

    /**
     * Column / property naming strategy.
     */
    public enum NamingStrategy {
        /** Property name equals column name. */
        IDENTITY {
            @Override public String toColumn(String property) { return property; }
            @Override public String toProperty(String column) { return column; }
        },
        /** camelCase property ↔ snake_case column. */
        SNAKE_CASE {
            @Override public String toColumn(String property) {
                return TypeUtils.camelToSnake(property);
            }
            @Override public String toProperty(String column) {
                return TypeUtils.snakeToCamel(column);
            }
        },
        /** Lower-case identity. */
        LOWER_CASE {
            @Override public String toColumn(String property) {
                return property == null ? null : property.toLowerCase(Locale.ROOT);
            }
            @Override public String toProperty(String column) {
                return column;
            }
        };

        public abstract String toColumn(String property);
        public abstract String toProperty(String column);

        public Function<String, String> columnFn() {
            return this::toColumn;
        }
    }
}
