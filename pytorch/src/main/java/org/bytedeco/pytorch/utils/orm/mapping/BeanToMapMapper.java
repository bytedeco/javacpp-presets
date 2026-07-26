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

import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Convert a JavaBean (getters / setters and/or public fields) to a {@link Map}.
 *
 * <p>Property names are used as map keys (identity naming). Transient / static
 * members are skipped. Annotation-free, storch-tinyorm style.
 */
public final class BeanToMapMapper {
    private static final Map<Class<?>, List<PropertyAccess>> CACHE = new ConcurrentHashMap<>();

    private BeanToMapMapper() {}

    public static Map<String, Object> toMap(Object bean) {
        if (bean == null) return null;
        if (bean instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) bean;
            return new LinkedHashMap<>(m);
        }
        List<PropertyAccess> props = propertiesOf(bean.getClass());
        Map<String, Object> out = new LinkedHashMap<>(props.size() * 2);
        for (PropertyAccess p : props) {
            if (!p.readable) continue;
            try {
                out.put(p.name, p.read(bean));
            } catch (Exception e) {
                throw new IllegalStateException(
                        "Failed to read property '" + p.name + "' of " + bean.getClass().getName(), e);
            }
        }
        return out;
    }

    public static List<Map<String, Object>> toMaps(Iterable<?> beans) {
        if (beans == null) return Collections.emptyList();
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object bean : beans) {
            out.add(toMap(bean));
        }
        return out;
    }

    public static List<String> propertyNames(Class<?> type) {
        List<PropertyAccess> props = propertiesOf(type);
        List<String> names = new ArrayList<>(props.size());
        for (PropertyAccess p : props) {
            if (p.readable || p.writable) names.add(p.name);
        }
        return names;
    }

    public static List<PropertyAccess> propertiesOf(Class<?> type) {
        if (type == null) throw new IllegalArgumentException("type required");
        return CACHE.computeIfAbsent(type, BeanToMapMapper::introspect);
    }

    private static List<PropertyAccess> introspect(Class<?> type) {
        Map<String, PropertyAccess> byName = new LinkedHashMap<>();

        // 1) JavaBeans getters/setters
        try {
            BeanInfo info = Introspector.getBeanInfo(type, Object.class);
            for (PropertyDescriptor pd : info.getPropertyDescriptors()) {
                String name = pd.getName();
                if (name == null || "class".equals(name)) continue;
                Method read = pd.getReadMethod();
                Method write = pd.getWriteMethod();
                Class<?> propType = pd.getPropertyType();
                if (propType == null && read != null) propType = read.getReturnType();
                if (propType == null && write != null && write.getParameterCount() == 1) {
                    propType = write.getParameterTypes()[0];
                }
                if (propType == null) continue;
                if (read != null && !Modifier.isPublic(read.getModifiers())) read = null;
                if (write != null && !Modifier.isPublic(write.getModifiers())) write = null;
                if (read == null && write == null) continue;
                byName.put(name, new PropertyAccess(name, propType, read, write, null));
            }
        } catch (Exception e) {
            // fall through to public fields
        }

        // 2) public instance fields not already covered
        for (Field f : type.getFields()) {
            int mod = f.getModifiers();
            if (Modifier.isStatic(mod) || Modifier.isTransient(mod)) continue;
            if (!Modifier.isPublic(mod)) continue;
            String name = f.getName();
            if (byName.containsKey(name)) {
                // attach field as fallback if no getter/setter
                PropertyAccess existing = byName.get(name);
                if (existing.field == null) {
                    byName.put(name, new PropertyAccess(
                            existing.name, existing.type, existing.getter, existing.setter, f));
                }
                continue;
            }
            byName.put(name, new PropertyAccess(name, f.getType(), null, null, f));
        }

        return Collections.unmodifiableList(new ArrayList<>(byName.values()));
    }

    /** Access descriptor for a bean property (getter / setter / public field). */
    public static final class PropertyAccess {
        public final String name;
        public final Class<?> type;
        public final Method getter;
        public final Method setter;
        public final Field field;
        public final boolean readable;
        public final boolean writable;

        PropertyAccess(String name, Class<?> type, Method getter, Method setter, Field field) {
            this.name = name;
            this.type = type;
            this.getter = getter;
            this.setter = setter;
            this.field = field;
            this.readable = getter != null || field != null;
            this.writable = setter != null || (field != null && !Modifier.isFinal(field.getModifiers()));
        }

        public Object read(Object bean) throws Exception {
            if (getter != null) return getter.invoke(bean);
            if (field != null) return field.get(bean);
            throw new IllegalStateException("Property not readable: " + name);
        }

        public void write(Object bean, Object value) throws Exception {
            Object coerced = TypeUtils.convert(value, type);
            if (setter != null) {
                setter.invoke(bean, coerced);
                return;
            }
            if (field != null && !Modifier.isFinal(field.getModifiers())) {
                field.set(bean, coerced);
                return;
            }
            throw new IllegalStateException("Property not writable: " + name);
        }
    }
}
