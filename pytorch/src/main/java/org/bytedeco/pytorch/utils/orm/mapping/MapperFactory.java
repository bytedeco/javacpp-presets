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

/**
 * Simple factory for common {@link Mapper} instances.
 */
public final class MapperFactory {
    private MapperFactory() {}

    /** Bean → Map&lt;String,Object&gt;. */
    @SuppressWarnings("unchecked")
    public static <T> Mapper<T, Map<String, Object>> beanToMap() {
        return bean -> BeanToMapMapper.toMap(bean);
    }

    /** Map → bean of {@code type}. */
    public static <T> Mapper<Map<String, ?>, T> mapToBean(Class<T> type) {
        return map -> MapToBeanMapper.toBean(map, type);
    }

    /** Map → bean with naming strategy. */
    public static <T> Mapper<Map<String, ?>, T> mapToBean(Class<T> type,
                                                          MapToBeanMapper.NamingStrategy naming) {
        return map -> MapToBeanMapper.toBean(map, type, naming);
    }

    /** Coerce any value to {@code type}. */
    public static <T> Mapper<Object, T> coerce(Class<T> type) {
        return value -> TypeUtils.convert(value, type);
    }

    /** Identity mapper. */
    public static <T> Mapper<T, T> identity() {
        return value -> value;
    }

    /** Compose two mappers: {@code a} then {@code b}. */
    public static <A, B, C> Mapper<A, C> compose(Mapper<A, B> a, Mapper<B, C> b) {
        return source -> b.map(a.map(source));
    }
}
