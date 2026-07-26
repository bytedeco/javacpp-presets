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

import org.bytedeco.pytorch.utils.orm.jdbc.JdbcUtils;

import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Map JDBC {@link ResultSet} rows to beans or maps.
 */
public final class ResultSetMapper {
    private ResultSetMapper() {}

    public static Map<String, Object> toMap(ResultSet rs) throws SQLException {
        if (rs == null) throw new IllegalArgumentException("ResultSet required");
        ResultSetMetaData meta = rs.getMetaData();
        String[] names = JdbcUtils.columnLabels(meta);
        Map<String, Object> row = new LinkedHashMap<>(names.length * 2);
        for (int i = 0; i < names.length; i++) {
            Object v = JdbcUtils.getObject(rs, i + 1);
            row.put(names[i], v);
        }
        return row;
    }

    public static List<Map<String, Object>> toMaps(ResultSet rs) throws SQLException {
        List<Map<String, Object>> out = new ArrayList<>();
        if (rs == null) return out;
        while (rs.next()) {
            out.add(toMap(rs));
        }
        return out;
    }

    public static <T> T toBean(ResultSet rs, Class<T> type) throws SQLException {
        return toBean(rs, type, MapToBeanMapper.NamingStrategy.IDENTITY);
    }

    public static <T> T toBean(ResultSet rs, Class<T> type,
                               MapToBeanMapper.NamingStrategy naming) throws SQLException {
        Map<String, Object> map = toMap(rs);
        return MapToBeanMapper.toBean(map, type, naming);
    }

    public static <T> List<T> toBeans(ResultSet rs, Class<T> type) throws SQLException {
        return toBeans(rs, type, MapToBeanMapper.NamingStrategy.IDENTITY);
    }

    public static <T> List<T> toBeans(ResultSet rs, Class<T> type,
                                      MapToBeanMapper.NamingStrategy naming) throws SQLException {
        List<T> out = new ArrayList<>();
        if (rs == null) return out;
        while (rs.next()) {
            out.add(toBean(rs, type, naming));
        }
        return out;
    }

    /** Map the current row to Object[] in column order. */
    public static Object[] toArray(ResultSet rs) throws SQLException {
        return JdbcUtils.readRow(rs);
    }
}
