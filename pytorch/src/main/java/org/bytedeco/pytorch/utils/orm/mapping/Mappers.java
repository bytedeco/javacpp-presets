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

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of named / typed {@link Mapper} instances.
 */
public final class Mappers {
    private static final ConcurrentHashMap<String, Mapper<?, ?>> BY_NAME = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Key, Mapper<?, ?>> BY_TYPES = new ConcurrentHashMap<>();

    private Mappers() {}

    public static <S, T> void register(String name, Mapper<S, T> mapper) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(mapper, "mapper");
        BY_NAME.put(name, mapper);
    }

    public static <S, T> void register(Class<S> source, Class<T> target, Mapper<S, T> mapper) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(mapper, "mapper");
        BY_TYPES.put(new Key(source, target), mapper);
    }

    @SuppressWarnings("unchecked")
    public static <S, T> Mapper<S, T> get(String name) {
        Mapper<?, ?> m = BY_NAME.get(name);
        if (m == null) throw new IllegalArgumentException("No mapper registered as: " + name);
        return (Mapper<S, T>) m;
    }

    @SuppressWarnings("unchecked")
    public static <S, T> Mapper<S, T> get(Class<S> source, Class<T> target) {
        Mapper<?, ?> m = BY_TYPES.get(new Key(source, target));
        if (m != null) return (Mapper<S, T>) m;
        // default: Map → bean
        if (Map.class.isAssignableFrom(source)) {
            return (Mapper<S, T>) (Mapper<Map<String, ?>, T>) map ->
                    MapToBeanMapper.toBean(map, target);
        }
        // bean → Map
        if (Map.class.isAssignableFrom(target)) {
            return (Mapper<S, T>) (Mapper<Object, Map<String, Object>>) BeanToMapMapper::toMap;
        }
        // type coercion
        if (TypeUtils.isSimpleType(target)) {
            return src -> TypeUtils.convert(src, target);
        }
        // bean-ish: convert via map
        return src -> {
            Map<String, Object> map = BeanToMapMapper.toMap(src);
            return MapToBeanMapper.toBean(map, target);
        };
    }

    public static boolean contains(String name) {
        return BY_NAME.containsKey(name);
    }

    public static void clear() {
        BY_NAME.clear();
        BY_TYPES.clear();
    }

    private static final class Key {
        final Class<?> source;
        final Class<?> target;

        Key(Class<?> source, Class<?> target) {
            this.source = source;
            this.target = target;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Key)) return false;
            Key key = (Key) o;
            return source.equals(key.source) && target.equals(key.target);
        }

        @Override
        public int hashCode() {
            return 31 * source.hashCode() + target.hashCode();
        }
    }
}
