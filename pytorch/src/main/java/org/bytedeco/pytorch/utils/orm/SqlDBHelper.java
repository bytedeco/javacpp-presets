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
package org.bytedeco.pytorch.utils.orm;

import org.bytedeco.pytorch.utils.orm.jdbc.JdbcUtils;
import org.bytedeco.pytorch.utils.orm.mapping.BeanToMapMapper;
import org.bytedeco.pytorch.utils.orm.mapping.MapToBeanMapper;
import org.bytedeco.pytorch.utils.orm.mapping.ResultSetMapper;
import org.bytedeco.pytorch.utils.orm.mapping.TypeUtils;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight JDBC helper (storch-tinyorm style) wrapping a {@link Connection}.
 *
 * <p>Works with any JDBC driver; SQLite in-memory is the primary target:
 * {@code DriverManager.getConnection("jdbc:sqlite::memory:")}.
 *
 * <pre>{@code
 * try (SqlDBHelper db = SqlDBHelper.sqliteMemory()) {
 *     db.createTableFromBean("person", Person.class);
 *     Person p = new Person();
 *     p.setId(1L); p.setName("Ada"); p.setAge(36);
 *     db.insert("person", p);
 *     List&lt;Person&gt; all = db.query(Person.class, "SELECT * FROM person");
 * }
 * }</pre>
 */
public class SqlDBHelper implements AutoCloseable {
    private final Connection connection;
    private final boolean closeOnClose;
    private MapToBeanMapper.NamingStrategy naming = MapToBeanMapper.NamingStrategy.IDENTITY;

    public SqlDBHelper(Connection connection) {
        this(connection, false);
    }

    public SqlDBHelper(Connection connection, boolean closeOnClose) {
        this.connection = Objects.requireNonNull(connection, "connection");
        this.closeOnClose = closeOnClose;
    }

    /** Open a private in-memory SQLite database (requires sqlite-jdbc). */
    public static SqlDBHelper sqliteMemory() throws SQLException {
        ensureSqliteDriver();
        Connection c = DriverManager.getConnection("jdbc:sqlite::memory:");
        return new SqlDBHelper(c, true);
    }

    /** Open a SQLite file (or {@code :memory:}). */
    public static SqlDBHelper sqlite(String path) throws SQLException {
        ensureSqliteDriver();
        String url = path != null && path.startsWith("jdbc:") ? path : "jdbc:sqlite:" + path;
        Connection c = DriverManager.getConnection(url);
        return new SqlDBHelper(c, true);
    }

    public static SqlDBHelper open(String jdbcUrl) throws SQLException {
        Connection c = DriverManager.getConnection(jdbcUrl);
        return new SqlDBHelper(c, true);
    }

    public static SqlDBHelper open(String jdbcUrl, String user, String password) throws SQLException {
        Connection c = DriverManager.getConnection(jdbcUrl, user, password);
        return new SqlDBHelper(c, true);
    }

    private static void ensureSqliteDriver() {
        try {
            Class.forName("org.sqlite.JDBC");
        } catch (ClassNotFoundException ignored) {
            // ServiceLoader may still find it
        }
    }

    public Connection getConnection() {
        return connection;
    }

    public SqlDBHelper naming(MapToBeanMapper.NamingStrategy naming) {
        this.naming = naming == null ? MapToBeanMapper.NamingStrategy.IDENTITY : naming;
        return this;
    }

    public MapToBeanMapper.NamingStrategy naming() {
        return naming;
    }

    // ---- query ----

    public List<Map<String, Object>> query(String sql, Object... params) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            JdbcUtils.bindAll(ps, params);
            try (ResultSet rs = ps.executeQuery()) {
                return ResultSetMapper.toMaps(rs);
            }
        }
    }

    public <T> List<T> query(Class<T> type, String sql, Object... params) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            JdbcUtils.bindAll(ps, params);
            try (ResultSet rs = ps.executeQuery()) {
                return ResultSetMapper.toBeans(rs, type, naming);
            }
        }
    }

    public Map<String, Object> queryForMap(String sql, Object... params) throws SQLException {
        List<Map<String, Object>> rows = query(sql, params);
        if (rows.isEmpty()) return null;
        if (rows.size() > 1) {
            throw new SQLException("Expected 1 row but got " + rows.size());
        }
        return rows.get(0);
    }

    public <T> T queryForObject(Class<T> type, String sql, Object... params) throws SQLException {
        List<T> rows = query(type, sql, params);
        if (rows.isEmpty()) return null;
        if (rows.size() > 1) {
            throw new SQLException("Expected 1 row but got " + rows.size());
        }
        return rows.get(0);
    }

    /** First column of first row, coerced to {@code type}. */
    public <T> T queryForValue(Class<T> type, String sql, Object... params) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            JdbcUtils.bindAll(ps, params);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return null;
                Object v = JdbcUtils.getObject(rs, 1);
                return TypeUtils.convert(v, type);
            }
        }
    }

    public long count(String table) throws SQLException {
        Long n = queryForValue(Long.class, "SELECT COUNT(*) FROM " + quoteTable(table));
        return n == null ? 0L : n;
    }

    // ---- update / execute ----

    public int update(String sql, Object... params) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            JdbcUtils.bindAll(ps, params);
            return ps.executeUpdate();
        }
    }

    public boolean execute(String sql) throws SQLException {
        try (Statement st = connection.createStatement()) {
            return st.execute(sql);
        }
    }

    public int[] batchUpdate(String sql, List<Object[]> batchParams) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            for (Object[] params : batchParams) {
                JdbcUtils.bindAll(ps, params);
                ps.addBatch();
            }
            return ps.executeBatch();
        }
    }

    // ---- insert ----

    public int insert(String table, Object bean) throws SQLException {
        if (bean instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) bean;
            return insert(table, map);
        }
        return insert(table, BeanToMapMapper.toMap(bean));
    }

    public int insert(String table, Map<String, ?> values) throws SQLException {
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException("insert values required");
        }
        // filter null keys; keep insertion order
        Map<String, Object> cols = new LinkedHashMap<>();
        for (Map.Entry<String, ?> e : values.entrySet()) {
            if (e.getKey() == null || e.getKey().isEmpty()) continue;
            cols.put(e.getKey(), e.getValue());
        }
        if (cols.isEmpty()) throw new IllegalArgumentException("no columns to insert");

        StringBuilder sb = new StringBuilder("INSERT INTO ").append(quoteTable(table)).append(" (");
        StringBuilder placeholders = new StringBuilder();
        List<Object> params = new ArrayList<>(cols.size());
        boolean first = true;
        for (Map.Entry<String, Object> e : cols.entrySet()) {
            if (!first) {
                sb.append(", ");
                placeholders.append(", ");
            }
            first = false;
            sb.append(JdbcUtils.quoteIdent(e.getKey()));
            placeholders.append("?");
            params.add(e.getValue());
        }
        sb.append(") VALUES (").append(placeholders).append(")");
        return update(sb.toString(), params.toArray());
    }

    public int insertAll(String table, Iterable<?> beans) throws SQLException {
        int total = 0;
        for (Object bean : beans) {
            total += insert(table, bean);
        }
        return total;
    }

    /** Generate UPDATE SET ... WHERE id = ? from bean map (id column required). */
    @SuppressWarnings("unchecked")
    public int updateById(String table, Object bean, String idColumn) throws SQLException {
        Map<String, Object> map;
        if (bean instanceof Map) {
            map = new LinkedHashMap<>((Map<String, ?>) bean);
        } else {
            map = BeanToMapMapper.toMap(bean);
        }
        if (!map.containsKey(idColumn)) {
            throw new IllegalArgumentException("Missing id column '" + idColumn + "' in bean/map");
        }
        Object id = map.remove(idColumn);
        if (map.isEmpty()) return 0;
        StringBuilder sb = new StringBuilder("UPDATE ").append(quoteTable(table)).append(" SET ");
        List<Object> params = new ArrayList<>();
        boolean first = true;
        for (Map.Entry<String, Object> e : map.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            sb.append(JdbcUtils.quoteIdent(e.getKey())).append(" = ?");
            params.add(e.getValue());
        }
        sb.append(" WHERE ").append(JdbcUtils.quoteIdent(idColumn)).append(" = ?");
        params.add(id);
        return update(sb.toString(), params.toArray());
    }

    public int deleteById(String table, String idColumn, Object id) throws SQLException {
        return update("DELETE FROM " + quoteTable(table) + " WHERE "
                + JdbcUtils.quoteIdent(idColumn) + " = ?", id);
    }

    // ---- DDL ----

    /**
     * CREATE TABLE IF NOT EXISTS from bean property types.
     * All columns nullable; INTEGER/REAL/TEXT/BLOB for SQLite affinity.
     */
    public void createTableFromBean(String table, Class<?> beanType) throws SQLException {
        createTableFromBean(table, beanType, null);
    }

    /**
     * @param primaryKey property/column name used as PRIMARY KEY (optional)
     */
    public void createTableFromBean(String table, Class<?> beanType, String primaryKey)
            throws SQLException {
        List<BeanToMapMapper.PropertyAccess> props = BeanToMapMapper.propertiesOf(beanType);
        if (props.isEmpty()) {
            throw new IllegalArgumentException("No properties found on " + beanType.getName());
        }
        StringBuilder sb = new StringBuilder("CREATE TABLE IF NOT EXISTS ")
                .append(quoteTable(table)).append(" (");
        boolean first = true;
        for (BeanToMapMapper.PropertyAccess p : props) {
            if (!p.readable && !p.writable) continue;
            if (!first) sb.append(", ");
            first = false;
            String col = naming.toColumn(p.name);
            sb.append(JdbcUtils.quoteIdent(col)).append(" ").append(TypeUtils.sqlTypeOf(p.type));
            if (primaryKey != null && (primaryKey.equals(p.name) || primaryKey.equals(col))) {
                sb.append(" PRIMARY KEY");
            }
        }
        sb.append(")");
        execute(sb.toString());
    }

    public void dropTable(String table) throws SQLException {
        execute("DROP TABLE IF EXISTS " + quoteTable(table));
    }

    // ---- transactions ----

    /** Work unit that may throw checked exceptions (e.g. {@link SQLException}). */
    @FunctionalInterface
    public interface SqlWork {
        void run() throws Exception;
    }

    /** Work unit returning a value that may throw checked exceptions. */
    @FunctionalInterface
    public interface SqlCallable<T> {
        T call() throws Exception;
    }

    public void withTransaction(SqlWork work) throws SQLException {
        withTransaction(() -> {
            work.run();
            return null;
        });
    }

    public <T> T withTransaction(SqlCallable<T> work) throws SQLException {
        boolean prev = connection.getAutoCommit();
        try {
            if (prev) connection.setAutoCommit(false);
            T result = work.call();
            connection.commit();
            return result;
        } catch (SQLException e) {
            safeRollback();
            throw e;
        } catch (Exception e) {
            safeRollback();
            if (e instanceof RuntimeException) throw (RuntimeException) e;
            throw new SQLException("Transaction failed: " + e.getMessage(), e);
        } finally {
            if (prev) {
                try {
                    connection.setAutoCommit(true);
                } catch (SQLException ignored) {
                }
            }
        }
    }

    private void safeRollback() {
        try {
            connection.rollback();
        } catch (SQLException ignored) {
        }
    }

    // ---- find helpers ----

    public <T> List<T> findAll(Class<T> type, String table) throws SQLException {
        return query(type, "SELECT * FROM " + quoteTable(table));
    }

    public <T> T findById(Class<T> type, String table, String idColumn, Object id)
            throws SQLException {
        return queryForObject(type,
                "SELECT * FROM " + quoteTable(table) + " WHERE "
                        + JdbcUtils.quoteIdent(idColumn) + " = ?", id);
    }

    public List<Map<String, Object>> findAll(String table) throws SQLException {
        return query("SELECT * FROM " + quoteTable(table));
    }

    private String quoteTable(String table) {
        if (table == null || table.isBlank()) {
            throw new IllegalArgumentException("table required");
        }
        String t = table.trim();
        // allow schema.table
        if (t.contains(".") && !t.contains("\"")) {
            String[] parts = t.split("\\.", 2);
            return JdbcUtils.quoteIdent(parts[0]) + "." + JdbcUtils.quoteIdent(parts[1]);
        }
        return JdbcUtils.quoteIdent(t);
    }

    @Override
    public void close() throws SQLException {
        if (closeOnClose && connection != null && !connection.isClosed()) {
            connection.close();
        }
    }
}
