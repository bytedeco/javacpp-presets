/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.utils.lake;

import org.bytedeco.pytorch.dataframe.Column;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lake table schema mapped to DataFrame {@link Column.DType}.
 */
public final class LakeSchema {

    public record Field(String name, Column.DType dtype, boolean nullable, String comment) {
        public Field {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(dtype, "dtype");
            if (comment == null) comment = "";
        }

        public static Field of(String name, Column.DType dtype) {
            return new Field(name, dtype, true, "");
        }

        public static Field of(String name, Column.DType dtype, boolean nullable) {
            return new Field(name, dtype, nullable, "");
        }
    }

    private final List<Field> fields;
    private final Map<String, Field> byName;

    private LakeSchema(List<Field> fields) {
        this.fields = List.copyOf(fields);
        Map<String, Field> m = new LinkedHashMap<>();
        for (Field f : this.fields) {
            m.put(f.name(), f);
        }
        this.byName = Collections.unmodifiableMap(m);
    }

    public static LakeSchema of(Field... fields) {
        return new LakeSchema(List.of(fields));
    }

    public static LakeSchema of(List<Field> fields) {
        return new LakeSchema(fields);
    }

    public static Builder builder() {
        return new Builder();
    }

    public List<Field> fields() {
        return fields;
    }

    public int size() {
        return fields.size();
    }

    public Field get(String name) {
        return byName.get(name);
    }

    public String[] names() {
        String[] n = new String[fields.size()];
        for (int i = 0; i < fields.size(); i++) n[i] = fields.get(i).name();
        return n;
    }

    public Column.DType[] dtypes() {
        Column.DType[] d = new Column.DType[fields.size()];
        for (int i = 0; i < fields.size(); i++) d[i] = fields.get(i).dtype();
        return d;
    }

    public LakeSchema select(String... columns) {
        if (columns == null || columns.length == 0) return this;
        List<Field> selected = new ArrayList<>(columns.length);
        for (String c : columns) {
            Field f = byName.get(c);
            if (f == null) {
                throw new LakeException("column not in schema: " + c);
            }
            selected.add(f);
        }
        return new LakeSchema(selected);
    }

    public static final class Builder {
        private final List<Field> fields = new ArrayList<>();

        public Builder add(String name, Column.DType dtype) {
            fields.add(Field.of(name, dtype));
            return this;
        }

        public Builder add(String name, Column.DType dtype, boolean nullable) {
            fields.add(Field.of(name, dtype, nullable));
            return this;
        }

        public Builder add(Field field) {
            fields.add(Objects.requireNonNull(field));
            return this;
        }

        public LakeSchema build() {
            if (fields.isEmpty()) {
                throw new IllegalStateException("LakeSchema requires at least one field");
            }
            return new LakeSchema(fields);
        }
    }

    @Override
    public String toString() {
        return "LakeSchema" + fields;
    }
}
