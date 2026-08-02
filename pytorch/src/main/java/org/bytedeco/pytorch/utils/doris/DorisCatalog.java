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
package org.bytedeco.pytorch.utils.doris;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.sql.JdbcTypeMap;
import org.bytedeco.pytorch.dataframe.sql.SqlOptions;
import org.bytedeco.pytorch.dataframe.sql.SqlReader;
import org.bytedeco.pytorch.utils.lake.LakeCapabilities;
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeSchema;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;
import org.bytedeco.pytorch.utils.lake.PartitionSpec;
import org.bytedeco.pytorch.utils.lake.ReplicaPolicy;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Types;
import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Doris catalog over MySQL protocol ({@code information_schema}) + Stream Load writes.
 */
public final class DorisCatalog implements LakeCatalog {

    private static final Set<LakeCapabilities> CAPS = Set.copyOf(EnumSet.of(
            LakeCapabilities.COLUMN_PROJECTION,
            LakeCapabilities.PARTITION_PRUNING,
            LakeCapabilities.PREDICATE_PUSHDOWN,
            LakeCapabilities.UPSERT,
            LakeCapabilities.STREAM_LOAD,
            LakeCapabilities.POINT_QUERY,
            LakeCapabilities.HIGH_THROUGHPUT_APPEND
    ));

    private final DorisOptions options;
    private final DorisPool pool;
    private final boolean ownPool;
    private final LakeMetrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public DorisCatalog(DorisOptions options) {
        this(options, DorisPool.open(options), true, LakeMetrics.of("doris-catalog"));
    }

    public DorisCatalog(DorisOptions options, DorisPool pool, boolean ownPool, LakeMetrics metrics) {
        this.options = Objects.requireNonNull(options, "options");
        this.pool = Objects.requireNonNull(pool, "pool");
        this.ownPool = ownPool;
        this.metrics = metrics == null ? LakeMetrics.of("doris-catalog") : metrics;
    }

    public static DorisCatalog open(DorisOptions options) {
        return new DorisCatalog(options);
    }

    public DorisOptions options() {
        return options;
    }

    public DorisPool pool() {
        return pool;
    }

    public LakeMetrics metrics() {
        return metrics;
    }

    @Override
    public LakeFormat format() {
        return LakeFormat.DORIS;
    }

    @Override
    public Set<LakeCapabilities> capabilities() {
        return CAPS;
    }

    @Override
    public List<String> listNamespaces() {
        ensureOpen();
        return pool.withConnection(c -> {
            try (Statement st = c.createStatement();
                 ResultSet rs = st.executeQuery(
                         "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA ORDER BY SCHEMA_NAME")) {
                List<String> out = new ArrayList<>();
                while (rs.next()) out.add(rs.getString(1));
                return out;
            } catch (SQLException e) {
                throw new LakeException(LakeFormat.DORIS, "listNamespaces", e.getMessage(), e);
            }
        });
    }

    @Override
    public List<String> listTables(String namespaceName) {
        ensureOpen();
        String db = namespaceName == null || namespaceName.isBlank() ? options.database() : namespaceName;
        if (db == null || db.isBlank()) {
            throw new LakeException(LakeFormat.DORIS, "listTables", "database/namespace required");
        }
        return pool.withConnection(c -> {
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")) {
                ps.setString(1, db);
                try (ResultSet rs = ps.executeQuery()) {
                    List<String> out = new ArrayList<>();
                    while (rs.next()) out.add(rs.getString(1));
                    return out;
                }
            } catch (SQLException e) {
                throw new LakeException(LakeFormat.DORIS, "listTables", e.getMessage(), e);
            }
        });
    }

    @Override
    public boolean tableExists(String namespaceName, String table) {
        ensureOpen();
        String db = ns(namespaceName);
        return pool.withConnection(c -> {
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT 1 FROM information_schema.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? LIMIT 1")) {
                ps.setString(1, db);
                ps.setString(2, table);
                try (ResultSet rs = ps.executeQuery()) {
                    return rs.next();
                }
            } catch (SQLException e) {
                throw new LakeException(LakeFormat.DORIS, "tableExists", e.getMessage(), e);
            }
        });
    }

    @Override
    public LakeTable loadTable(String namespaceName, String table) {
        ensureOpen();
        Objects.requireNonNull(table, "table");
        String db = ns(namespaceName);
        LakeSchema schema = loadSchema(db, table);
        Map<String, String> props = new LinkedHashMap<>();
        props.put("engine", "doris");
        props.put("database", db);
        return LakeTable.builder(LakeFormat.DORIS, table, schema)
                .namespaceName(db)
                .properties(props)
                .capabilities(CAPS.toArray(new LakeCapabilities[0]))
                .build();
    }

    @Override
    public LakeTable createTable(String namespaceName, String table, LakeSchema schema,
                                 PartitionSpec partitionSpec, Map<String, String> props) {
        ensureOpen();
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(schema, "schema");
        String db = ns(namespaceName);
        String ddl = buildCreateTableDdl(db, table, schema, partitionSpec, props);
        pool.execute(c -> {
            try (Statement st = c.createStatement()) {
                if (db != null && !db.isBlank()) {
                    st.execute("CREATE DATABASE IF NOT EXISTS `" + escapeIdent(db) + "`");
                    st.execute("USE `" + escapeIdent(db) + "`");
                }
                st.execute(ddl);
            }
        });
        return loadTable(db, table);
    }

    @Override
    public void dropTable(String namespaceName, String table, boolean ifExists) {
        ensureOpen();
        String db = ns(namespaceName);
        String sql = "DROP TABLE " + (ifExists ? "IF EXISTS " : "")
                + qualify(db, table);
        pool.execute(c -> {
            try (Statement st = c.createStatement()) {
                st.execute(sql);
            }
        });
    }

    @Override
    public LakeScan scan(String namespaceName, String table) {
        return new DorisScan(this, loadTable(namespaceName, table));
    }

    @Override
    public LakeWrite write(String namespaceName, String table) {
        LakeTable t = loadTable(namespaceName, table);
        DorisOptions writeOpts = options.toBuilder()
                .database(t.namespaceName())
                .table(t.name())
                .build();
        return new DorisWrite(this, t, writeOpts);
    }

    @Override
    public LakeStream stream(String namespaceName, String table) {
        return scan(namespaceName, table).stream();
    }

    /** Ad-hoc SQL → DataFrame (point query / analytics). */
    public DataFrame query(String sql) {
        ensureOpen();
        long t0 = System.nanoTime();
        try {
            DataFrame df = pool.withConnection(c -> {
                try {
                    SqlOptions so = SqlOptions.builder()
                            .fetchSize(options.fetchSize())
                            .quoteIdentifiers(false)
                            .build();
                    return SqlReader.read(c, sql, so);
                } catch (Exception e) {
                    throw new LakeException(LakeFormat.DORIS, "query", e.getMessage(), e);
                }
            });
            metrics.recordRead(df.rowCount(), System.nanoTime() - t0);
            return df;
        } catch (LakeException e) {
            metrics.recordFailure();
            throw e;
        }
    }

    /** Point lookup by unique / key columns: {@code SELECT cols FROM db.t WHERE k=? ...}. */
    public DataFrame pointQuery(String namespaceName, String table, Map<String, Object> keys, String... columns) {
        Objects.requireNonNull(keys, "keys");
        if (keys.isEmpty()) throw new LakeException(LakeFormat.DORIS, "pointQuery", "keys required");
        String db = ns(namespaceName);
        String colList = (columns == null || columns.length == 0) ? "*" : String.join(", ", quoteAll(columns));
        StringBuilder where = new StringBuilder();
        List<Object> params = new ArrayList<>();
        for (Map.Entry<String, Object> e : keys.entrySet()) {
            if (where.length() > 0) where.append(" AND ");
            where.append('`').append(escapeIdent(e.getKey())).append("` = ?");
            params.add(e.getValue());
        }
        String sql = "SELECT " + colList + " FROM " + qualify(db, table) + " WHERE " + where;
        long t0 = System.nanoTime();
        try {
            DataFrame df = pool.withConnection(c -> {
                try (PreparedStatement ps = c.prepareStatement(sql)) {
                    if (options.fetchSize() > 0) {
                        try { ps.setFetchSize(options.fetchSize()); } catch (SQLException ignored) {}
                    }
                    for (int i = 0; i < params.size(); i++) {
                        ps.setObject(i + 1, params.get(i));
                    }
                    try (ResultSet rs = ps.executeQuery()) {
                        return SqlReader.fromResultSet(rs, SqlOptions.builder()
                                .fetchSize(options.fetchSize()).quoteIdentifiers(false).build());
                    }
                } catch (Exception e) {
                    throw new LakeException(LakeFormat.DORIS, "pointQuery", e.getMessage(), e);
                }
            });
            metrics.recordRead(df.rowCount(), System.nanoTime() - t0);
            return df;
        } catch (LakeException e) {
            metrics.recordFailure();
            throw e;
        }
    }

    LakeSchema loadSchema(String db, String table) {
        return pool.withConnection(c -> {
            try {
                DatabaseMetaData meta = c.getMetaData();
                try (ResultSet rs = meta.getColumns(null, db, table, null)) {
                    LakeSchema.Builder b = LakeSchema.builder();
                    boolean any = false;
                    while (rs.next()) {
                        any = true;
                        String name = rs.getString("COLUMN_NAME");
                        int sqlType = rs.getInt("DATA_TYPE");
                        String typeName = rs.getString("TYPE_NAME");
                        int scale = rs.getInt("DECIMAL_DIGITS");
                        int nullable = rs.getInt("NULLABLE");
                        Column.DType dt = JdbcTypeMap.fromJdbc(sqlType, typeName, scale);
                        b.add(name, dt, nullable != DatabaseMetaData.columnNoNulls);
                    }
                    if (!any) {
                        // fallback: DESCRIBE
                        try (Statement st = c.createStatement();
                             ResultSet drs = st.executeQuery("DESCRIBE " + qualify(db, table))) {
                            while (drs.next()) {
                                any = true;
                                String name = drs.getString(1);
                                String typeName = drs.getString(2);
                                Column.DType dt = JdbcTypeMap.fromJdbc(Types.VARCHAR, typeName, 0);
                                b.add(name, dt, true);
                            }
                        }
                    }
                    if (!any) {
                        throw new LakeException(LakeFormat.DORIS, "loadSchema",
                                "table not found or empty schema: " + qualify(db, table));
                    }
                    return b.build();
                }
            } catch (LakeException e) {
                throw e;
            } catch (Exception e) {
                throw new LakeException(LakeFormat.DORIS, "loadSchema", e.getMessage(), e);
            }
        });
    }

    String buildCreateTableDdl(String db, String table, LakeSchema schema,
                               PartitionSpec partitionSpec, Map<String, String> props) {
        DorisOptions.TableModel model = options.tableModel();
        String[] keys = options.keys();
        if (keys == null || keys.length == 0) {
            keys = new String[]{schema.fields().get(0).name()};
        }
        String[] dist = options.distributeBy();
        if (dist == null || dist.length == 0) dist = keys;

        StringBuilder sb = new StringBuilder();
        sb.append("CREATE TABLE IF NOT EXISTS ").append(qualify(db, table)).append(" (\n");
        List<String> colDefs = new ArrayList<>();
        for (LakeSchema.Field f : schema.fields()) {
            colDefs.add("  `" + escapeIdent(f.name()) + "` " + toDorisType(f.dtype())
                    + (f.nullable() ? " NULL" : " NOT NULL"));
        }
        sb.append(String.join(",\n", colDefs));
        sb.append("\n)\n");
        switch (model) {
            case UNIQUE -> sb.append("UNIQUE KEY(").append(joinIdents(keys)).append(")\n");
            case AGGREGATE -> sb.append("AGGREGATE KEY(").append(joinIdents(keys)).append(")\n");
            default -> sb.append("DUPLICATE KEY(").append(joinIdents(keys)).append(")\n");
        }
        // optional RANGE partition on first identity column
        if (partitionSpec != null && partitionSpec.identityColumns() != null
                && partitionSpec.identityColumns().length > 0) {
            String pc = partitionSpec.identityColumns()[0];
            sb.append("PARTITION BY RANGE(`").append(escapeIdent(pc)).append("`)\n()\n");
        }
        sb.append("DISTRIBUTED BY HASH(").append(joinIdents(dist)).append(") BUCKETS ")
                .append(options.buckets()).append('\n');
        int rep = options.replicationNum();
        if (props != null && props.containsKey("replication_num")) {
            try { rep = Integer.parseInt(props.get("replication_num")); } catch (NumberFormatException ignored) {}
        }
        sb.append("PROPERTIES (\n  \"replication_num\" = \"").append(rep).append("\"\n)");
        return sb.toString();
    }

    static String toDorisType(Column.DType dtype) {
        return switch (dtype) {
            case INT32 -> "INT";
            case INT64 -> "BIGINT";
            case FLOAT32 -> "FLOAT";
            case FLOAT64 -> "DOUBLE";
            case BOOLEAN -> "BOOLEAN";
            case DATE -> "DATE";
            case DATETIME -> "DATETIME";
            case TIME -> "VARCHAR(32)";
            case BINARY -> "STRING";
            default -> "VARCHAR(65533)";
        };
    }

    String ns(String namespaceName) {
        if (namespaceName != null && !namespaceName.isBlank()) return namespaceName;
        if (options.database() != null && !options.database().isBlank()) return options.database();
        throw new LakeException(LakeFormat.DORIS, "namespace", "database required");
    }

    static String qualify(String db, String table) {
        if (db == null || db.isBlank()) return "`" + escapeIdent(table) + "`";
        return "`" + escapeIdent(db) + "`.`" + escapeIdent(table) + "`";
    }

    static String escapeIdent(String name) {
        return name == null ? "" : name.replace("`", "``");
    }

    static String joinIdents(String[] cols) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cols.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append('`').append(escapeIdent(cols[i])).append('`');
        }
        return sb.toString();
    }

    static String[] quoteAll(String[] cols) {
        String[] out = new String[cols.length];
        for (int i = 0; i < cols.length; i++) {
            out[i] = "`" + escapeIdent(cols[i]) + "`";
        }
        return out;
    }

    private void ensureOpen() {
        if (closed.get()) throw new LakeException(LakeFormat.DORIS, "catalog", "closed");
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        if (ownPool) pool.close();
    }

    // ── Scan ────────────────────────────────────────────────────────────────

    public static final class DorisScan implements LakeScan {
        private final DorisCatalog catalog;
        private final LakeTable table;
        private String[] columns;
        private PartitionFilter filter;
        private String where;
        private ReplicaPolicy replicas;
        private long limit = -1;
        private int batchRows;
        private int parallelism = 1;

        DorisScan(DorisCatalog catalog, LakeTable table) {
            this.catalog = catalog;
            this.table = table;
            this.batchRows = catalog.options.batchRows();
            this.replicas = catalog.options.replicaPolicy();
            if (catalog.options.columns() != null) this.columns = catalog.options.columns();
            if (catalog.options.partitionFilter() != null) this.filter = catalog.options.partitionFilter();
            if (catalog.options.where() != null) this.where = catalog.options.where();
        }

        @Override public LakeTable table() { return table; }
        @Override public LakeScan columns(String... columns) { this.columns = columns; return this; }
        @Override public LakeScan filter(PartitionFilter filter) { this.filter = filter; return this; }
        @Override public LakeScan where(String expression) { this.where = expression; return this; }
        @Override public LakeScan snapshotId(Long snapshotId) { return this; /* N/A for Doris */ }
        @Override public LakeScan asOfTimeMs(Long epochMs) { return this; }
        @Override public LakeScan replicas(ReplicaPolicy policy) { this.replicas = policy; return this; }
        @Override public LakeScan limit(long maxRows) { this.limit = maxRows; return this; }
        @Override public LakeScan batchRows(int batchRows) { this.batchRows = Math.max(1, batchRows); return this; }
        @Override public LakeScan parallelism(int parallelism) { this.parallelism = Math.max(1, parallelism); return this; }

        String buildSql() {
            String colList = (columns == null || columns.length == 0)
                    ? "*" : String.join(", ", quoteAll(columns));
            StringBuilder sb = new StringBuilder("SELECT ").append(colList)
                    .append(" FROM ").append(qualify(table.namespaceName(), table.name()));
            List<String> preds = new ArrayList<>();
            if (filter != null && !filter.isEmpty()) preds.add(filter.toSql());
            if (where != null && !where.isBlank()) preds.add("(" + where + ")");
            if (!preds.isEmpty()) sb.append(" WHERE ").append(String.join(" AND ", preds));
            if (limit >= 0) sb.append(" LIMIT ").append(limit);
            return sb.toString();
        }

        @Override
        public DataFrame collect() {
            return catalog.query(buildSql());
        }

        @Override
        public LakeStream stream() {
            return new DorisStream(catalog, buildSql(), batchRows,
                    catalog.options.idleStop(), limit < 0 ? Long.MAX_VALUE : Math.max(1, (limit + batchRows - 1) / batchRows));
        }
    }

    // ── Write ───────────────────────────────────────────────────────────────

    public static final class DorisWrite implements LakeWrite {
        private final DorisCatalog catalog;
        private final LakeTable table;
        private final DorisOptions writeOptions;
        private Mode mode = Mode.APPEND;
        private PartitionFilter staticPartition;
        private String label;
        private final List<DataFrame> buffer = new ArrayList<>();
        private final AtomicBoolean committed = new AtomicBoolean(false);
        private final AtomicBoolean aborted = new AtomicBoolean(false);

        DorisWrite(DorisCatalog catalog, LakeTable table, DorisOptions writeOptions) {
            this.catalog = catalog;
            this.table = table;
            this.writeOptions = writeOptions;
        }

        @Override public LakeTable table() { return table; }
        @Override public LakeWrite mode(Mode mode) { this.mode = mode == null ? Mode.APPEND : mode; return this; }
        @Override public LakeWrite partition(PartitionFilter staticPartition) { this.staticPartition = staticPartition; return this; }
        @Override public LakeWrite label(String label) { this.label = label; return this; }

        @Override
        public LakeWrite write(DataFrame df) {
            if (aborted.get()) throw new LakeException(LakeFormat.DORIS, "write", "aborted");
            if (committed.get()) throw new LakeException(LakeFormat.DORIS, "write", "already committed");
            Objects.requireNonNull(df, "dataframe");
            buffer.add(df);
            return this;
        }

        @Override
        public void commit() {
            if (aborted.get()) throw new LakeException(LakeFormat.DORIS, "commit", "aborted");
            if (!committed.compareAndSet(false, true)) return;
            if (buffer.isEmpty()) return;
            if (mode == Mode.OVERWRITE) {
                // Doris has no generic TRUNCATE-from-client guarantee across models;
                // best-effort DELETE or document Stream Load append-only for DUPLICATE.
                try {
                    catalog.pool.execute(c -> {
                        try (Statement st = c.createStatement()) {
                            st.execute("TRUNCATE TABLE " + qualify(table.namespaceName(), table.name()));
                        } catch (SQLException e) {
                            // fallback: continue with append (Unique Key will upsert)
                        }
                    });
                } catch (Exception ignored) {}
            }
            try (DorisStreamLoad loader = new DorisStreamLoad(writeOptions, null, catalog.metrics)) {
                int i = 0;
                for (DataFrame df : buffer) {
                    String lbl = label;
                    if (lbl == null) {
                        lbl = writeOptions.labelPrefix() + "-" + table.name() + "-" + System.currentTimeMillis() + "-" + (i++);
                    } else if (buffer.size() > 1) {
                        lbl = label + "-" + (i++);
                    }
                    loader.load(df, lbl);
                }
            }
            buffer.clear();
        }

        @Override
        public void abort() {
            aborted.set(true);
            buffer.clear();
        }

        @Override
        public void close() {
            // no auto-commit
            buffer.clear();
        }
    }

    // ── Stream ──────────────────────────────────────────────────────────────

    public static final class DorisStream implements LakeStream {
        private final DorisCatalog catalog;
        private final String baseSql;
        private int batchRows;
        private Duration idleStop;
        private long maxBatches;
        private final AtomicBoolean stopped = new AtomicBoolean(false);
        private final AtomicBoolean closed = new AtomicBoolean(false);
        private Connection connection;
        private Statement statement;
        private ResultSet resultSet;
        private String[] names;
        private Column.DType[] dtypes;
        private boolean metaReady;
        private boolean exhausted;
        private long batchesEmitted;
        private long lastRowActivityMs = System.currentTimeMillis();
        private long watermarkRows;

        DorisStream(DorisCatalog catalog, String sql, int batchRows, Duration idleStop, long maxBatches) {
            this.catalog = catalog;
            this.baseSql = sql;
            this.batchRows = Math.max(1, batchRows);
            this.idleStop = idleStop == null ? Duration.ofSeconds(30) : idleStop;
            this.maxBatches = maxBatches <= 0 ? Long.MAX_VALUE : maxBatches;
        }

        @Override
        public LakeStream batchRows(int batchRows) {
            this.batchRows = Math.max(1, batchRows);
            return this;
        }

        @Override
        public LakeStream idleStop(Duration idle) {
            this.idleStop = idle == null ? Duration.ofSeconds(30) : idle;
            return this;
        }

        @Override
        public LakeStream maxBatches(long maxBatches) {
            this.maxBatches = maxBatches <= 0 ? Long.MAX_VALUE : maxBatches;
            return this;
        }

        @Override
        public void commit() {
            // JDBC result-set stream: watermark is rows consumed
            watermarkRows += 0; // already advanced in poll
        }

        @Override
        public void stop() {
            stopped.set(true);
        }

        @Override
        public boolean isStopped() {
            return stopped.get() || closed.get() || exhausted || batchesEmitted >= maxBatches;
        }

        private void ensureQuery() throws SQLException, InterruptedException {
            if (connection != null) return;
            connection = catalog.pool.borrow();
            statement = connection.createStatement(ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
            if (catalog.options.fetchSize() > 0) {
                try { statement.setFetchSize(catalog.options.fetchSize()); } catch (SQLException ignored) {}
            }
            resultSet = statement.executeQuery(baseSql);
        }

        private void ensureMeta() throws Exception {
            if (metaReady) return;
            ensureQuery();
            var meta = resultSet.getMetaData();
            names = JdbcTypeMap.namesFromMeta(meta);
            dtypes = JdbcTypeMap.dtypesFromMeta(meta);
            metaReady = true;
        }

        @Override
        public DataFrame poll() {
            if (isStopped()) return null;
            long t0 = System.nanoTime();
            try {
                ensureMeta();
                DataFrame df = DataFrame.create();
                for (int i = 0; i < names.length; i++) df.addColumn(names[i], dtypes[i]);
                int n = 0;
                while (n < batchRows) {
                    if (!resultSet.next()) {
                        exhausted = true;
                        break;
                    }
                    int ri = df.addEmptyRow();
                    for (int i = 0; i < names.length; i++) {
                        Object v = JdbcTypeMap.getValue(resultSet, i + 1, dtypes[i]);
                        df.set(ri, names[i], v);
                    }
                    n++;
                    lastRowActivityMs = System.currentTimeMillis();
                }
                if (n == 0) {
                    if (exhausted) return null;
                    if (idleStop.toMillis() > 0
                            && System.currentTimeMillis() - lastRowActivityMs > idleStop.toMillis()) {
                        return null;
                    }
                    return null;
                }
                batchesEmitted++;
                watermarkRows += n;
                catalog.metrics.recordBatch(n);
                catalog.metrics.recordRead(n, System.nanoTime() - t0);
                return df;
            } catch (Exception e) {
                catalog.metrics.recordFailure();
                stop();
                throw new LakeException(LakeFormat.DORIS, "stream.poll", e.getMessage(), e);
            }
        }

        @Override
        public void close() {
            if (!closed.compareAndSet(false, true)) return;
            stopped.set(true);
            try { if (resultSet != null) resultSet.close(); } catch (Exception ignored) {}
            try { if (statement != null) statement.close(); } catch (Exception ignored) {}
            if (connection != null) {
                catalog.pool.release(connection);
                connection = null;
            }
        }
    }
}
