package org.bytedeco.pytorch.data.dataframe.pgvector;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.sql.SqlOptions;
import org.bytedeco.pytorch.data.dataframe.sql.SqlReader;
import org.bytedeco.pytorch.data.dataframe.sql.SqlWriter;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.Closeable;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Full-featured pgvector client for DataFrame I/O — plain JDBC, no extra library
 * beyond a PostgreSQL driver on the <em>application</em> classpath
 * ({@code org.postgresql:postgresql}).
 *
 * <h2>Coverage (psycopg / sqlalchemy + pgvector parity subset)</h2>
 * <ul>
 *   <li><b>Connection</b> — connect / jdbc url / user / password / wrap Connection</li>
 *   <li><b>Schema</b> — CREATE EXTENSION vector; create table/index (HNSW/IVFFlat)</li>
 *   <li><b>SQL</b> — execute / query → DataFrame</li>
 *   <li><b>Vectors</b> — upsert / delete / knn search / fetch / scroll</li>
 *   <li><b>DataFrame</b> — {@link #writeDataFrame}, {@link #readDataFrame},
 *       JSONB or expanded-column layouts</li>
 * </ul>
 *
 * <h2>Official-SDK switch (SPI only)</h2>
 * Implement {@link PgVectorBackend} and register via {@code META-INF/services}
 * or {@link #registerBackend}. A backend named {@code "pgvector"} overrides this built-in.
 *
 * <pre>{@code
 * try (PgVector pg = PgVector.connect("jdbc:postgresql://localhost:5432/postgres",
 *         "postgres", "postgres")) {
 *     pg.ensureExtension();
 *     df.toPgVector(pg, PgVectorOptions.builder().table("docs").dim(384)
 *         .idColumn("id").vectorColumn("emb").build());
 * }
 * }</pre>
 */
public class PgVector implements Closeable {

    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);

    private static final Map<String, PgVectorBackend> BACKENDS = new ConcurrentHashMap<>();
    static {
        reloadBackends();
    }

    private final String jdbcUrl;
    private final Properties props;
    private final boolean ownConnection;
    private Connection connection;
    private final Object lock = new Object();

    protected PgVector(String jdbcUrl, Properties props, Connection connection, boolean ownConnection) {
        this.jdbcUrl = jdbcUrl;
        this.props = props == null ? new Properties() : props;
        this.connection = connection;
        this.ownConnection = ownConnection;
    }

    // ── SPI ───────────────────────────────────────────────────────────────

    public static void reloadBackends() {
        BACKENDS.clear();
        try {
            for (PgVectorBackend b : ServiceLoader.load(PgVectorBackend.class)) {
                registerBackend(b);
            }
        } catch (Throwable ignored) {}
    }

    public static void registerBackend(PgVectorBackend backend) {
        if (backend == null || backend.name() == null) return;
        BACKENDS.put(backend.name().toLowerCase(Locale.ROOT), backend);
        if (backend.aliases() != null) {
            for (String a : backend.aliases()) {
                if (a != null && !a.isBlank()) {
                    BACKENDS.put(a.toLowerCase(Locale.ROOT), backend);
                }
            }
        }
    }

    public static PgVectorBackend backend(String name) {
        if (name == null) return null;
        return BACKENDS.get(name.toLowerCase(Locale.ROOT));
    }

    // ── factories ─────────────────────────────────────────────────────────

    public static PgVector connect(String jdbcUrl) {
        return connect(jdbcUrl, null, null);
    }

    public static PgVector connect(String jdbcUrl, String user, String password) {
        Map<String, Object> cfg = new LinkedHashMap<>();
        cfg.put("url", jdbcUrl);
        if (user != null) cfg.put("user", user);
        if (password != null) cfg.put("password", password);
        return open(cfg);
    }

    public static PgVector wrap(Connection connection) {
        Objects.requireNonNull(connection, "connection");
        return new PgVector(null, new Properties(), connection, false);
    }

    public static PgVector connectUri(String uri) {
        Objects.requireNonNull(uri, "uri");
        String s = uri.trim();
        Map<String, Object> cfg = new LinkedHashMap<>();
        if (s.startsWith("pgvector://") || s.startsWith("postgres://")
                || s.startsWith("postgresql://") || s.startsWith("pg://")) {
            String rest = s.substring(s.indexOf("://") + 3);
            String path = rest;
            String query = null;
            int q = rest.indexOf('?');
            if (q >= 0) {
                path = rest.substring(0, q);
                query = rest.substring(q + 1);
            }
            // optional user:pass@host:port/db
            String hostPath = path;
            int at = path.lastIndexOf('@');
            if (at >= 0) {
                String auth = path.substring(0, at);
                hostPath = path.substring(at + 1);
                int colon = auth.indexOf(':');
                if (colon >= 0) {
                    cfg.put("user", auth.substring(0, colon));
                    cfg.put("password", auth.substring(colon + 1));
                } else {
                    cfg.put("user", auth);
                }
            }
            cfg.put("url", "jdbc:postgresql://" + hostPath);
            if (query != null) parseQuery(query, cfg);
        } else if (s.startsWith("jdbc:")) {
            cfg.put("url", s);
        } else {
            cfg.put("url", "jdbc:postgresql://" + s);
        }
        return open(cfg);
    }

    public static PgVector open(Map<String, Object> config) {
        Map<String, Object> cfg = config == null ? Map.of() : config;
        PgVectorBackend plugin = BACKENDS.get("pgvector");
        if (plugin == null) plugin = BACKENDS.get("postgres");
        if (plugin == null) plugin = BACKENDS.get("postgresql");
        if (plugin == null) plugin = BACKENDS.get("pg");
        if (plugin != null) return plugin.open(cfg);
        return openBuiltin(cfg);
    }

    public static PgVector openBuiltin(Map<String, Object> cfg) {
        String url = str(cfg, "url", str(cfg, "jdbcUrl", null));
        if (url == null) throw new PgVectorException("pgvector requires jdbc url");
        Properties props = new Properties();
        String user = str(cfg, "user", str(cfg, "username", null));
        String password = str(cfg, "password", null);
        if (user != null) props.setProperty("user", user);
        if (password != null) props.setProperty("password", password);
        Object extra = cfg.get("properties");
        if (extra instanceof Properties p) props.putAll(p);
        else if (extra instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (e.getKey() != null && e.getValue() != null) {
                    props.setProperty(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
                }
            }
        }
        return new PgVector(url, props, null, true);
    }

    public static Builder builder() {
        return new Builder();
    }

    // ── accessors ─────────────────────────────────────────────────────────

    public String jdbcUrl() { return jdbcUrl; }

    public Connection connection() {
        try {
            return conn();
        } catch (SQLException e) {
            throw wrap(e, "connection");
        }
    }

    // ── schema ────────────────────────────────────────────────────────────

    public void ensureExtension() {
        try {
            Connection c = conn();
            try (Statement st = c.createStatement()) {
                st.execute("CREATE EXTENSION IF NOT EXISTS vector");
            }
        } catch (SQLException e) {
            throw wrap(e, "ensureExtension");
        }
    }

    public void ensureTable(PgVectorOptions options) {
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        if (opt.ensureExtension()) ensureExtension();
        if (!opt.ensureTable()) return;
        if (opt.dim() <= 0 && opt.payloadMode() == PgVectorOptions.PayloadMode.JSONB) {
            throw new PgVectorException("dim required to create vector table", null, null, "ensureTable");
        }
        String table = qualified(opt);
        String idCol = quoteIdent(opt.idSqlColumn(), opt.quoteIdentifiers());
        String vecCol = quoteIdent(opt.vectorSqlColumn(), opt.quoteIdentifiers());
        String payCol = quoteIdent(opt.payloadColumn(), opt.quoteIdentifiers());
        try {
            Connection c = conn();
            try (Statement st = c.createStatement()) {
                if (opt.payloadMode() == PgVectorOptions.PayloadMode.JSONB) {
                    String ddl = "CREATE TABLE IF NOT EXISTS " + table + " ("
                        + idCol + " TEXT PRIMARY KEY, "
                        + vecCol + " vector(" + opt.dim() + "), "
                        + payCol + " JSONB DEFAULT '{}'::jsonb"
                        + ")";
                    st.execute(ddl);
                }
                // COLUMNS mode is created on write from DataFrame schema
                if (opt.ensureIndex() && opt.payloadMode() == PgVectorOptions.PayloadMode.JSONB
                        && opt.indexMethod() != PgVectorOptions.IndexMethod.NONE) {
                    createVectorIndex(opt);
                }
            }
        } catch (SQLException e) {
            throw wrap(e, "ensureTable");
        }
    }

    public void createVectorIndex(PgVectorOptions options) {
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        String table = qualified(opt);
        String vecCol = quoteIdent(opt.vectorSqlColumn(), opt.quoteIdentifiers());
        String bare = opt.table().replace("\"", "");
        String idx = quoteIdent(bare + "_" + opt.vectorSqlColumn() + "_idx", opt.quoteIdentifiers());
        String ops = opt.metric().pgvectorOps();
        try {
            Connection c = conn();
            try (Statement st = c.createStatement()) {
                if (opt.indexMethod() == PgVectorOptions.IndexMethod.HNSW
                        || opt.indexMethod() == null) {
                    try {
                        st.execute("CREATE INDEX IF NOT EXISTS " + idx
                            + " ON " + table + " USING hnsw (" + vecCol + " " + ops + ")");
                        return;
                    } catch (SQLException hnswEx) {
                        // fall through to ivfflat
                    }
                }
                if (opt.indexMethod() != PgVectorOptions.IndexMethod.NONE) {
                    try {
                        st.execute("CREATE INDEX IF NOT EXISTS " + idx
                            + " ON " + table + " USING ivfflat (" + vecCol + " " + ops + ") WITH (lists = 100)");
                    } catch (SQLException ignored) {
                        // sequential scan still works
                    }
                }
            }
        } catch (SQLException e) {
            throw wrap(e, "createVectorIndex");
        }
    }

    public void dropTable(PgVectorOptions options) {
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        try {
            Connection c = conn();
            try (Statement st = c.createStatement()) {
                st.execute("DROP TABLE IF EXISTS " + qualified(opt));
            }
        } catch (SQLException e) {
            throw wrap(e, "dropTable");
        }
    }

    public boolean tableExists(String table) {
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = ? LIMIT 1")) {
                // unquoted lower-case name comparison
                ps.setString(1, table.replace("\"", "").toLowerCase(Locale.ROOT));
                try (ResultSet rs = ps.executeQuery()) {
                    return rs.next();
                }
            }
        } catch (SQLException e) {
            return false;
        }
    }

    // ── SQL helpers ───────────────────────────────────────────────────────

    public void execute(String sql) {
        try {
            Connection c = conn();
            try (Statement st = c.createStatement()) {
                st.execute(sql);
            }
        } catch (SQLException e) {
            throw wrap(e, "execute");
        }
    }

    public DataFrame query(String sql) {
        try {
            return SqlReader.read(conn(), sql);
        } catch (Exception e) {
            if (e instanceof SQLException se) throw wrap(se, "query");
            throw new PgVectorException("query failed: " + e.getMessage(), e, null, "query");
        }
    }

    public long count(PgVectorOptions options) {
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        try {
            Connection c = conn();
            try (Statement st = c.createStatement();
                 ResultSet rs = st.executeQuery("SELECT COUNT(*) FROM " + qualified(opt))) {
                if (rs.next()) return rs.getLong(1);
                return 0L;
            }
        } catch (SQLException e) {
            return -1L;
        }
    }

    // ── vector ops ───────────────────────────────────────────────────

    public void upsertRecords(Collection<VectorRecord> records, PgVectorOptions options) {
        if (records == null || records.isEmpty()) return;
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        String table = qualified(opt);
        String idCol = quoteIdent(opt.idSqlColumn(), opt.quoteIdentifiers());
        String vecCol = quoteIdent(opt.vectorSqlColumn(), opt.quoteIdentifiers());
        String payCol = quoteIdent(opt.payloadColumn(), opt.quoteIdentifiers());
        String sql = "INSERT INTO " + table + " (" + idCol + ", " + vecCol + ", " + payCol
            + ") VALUES (?,?,?::jsonb) ON CONFLICT (" + idCol + ") DO UPDATE SET "
            + vecCol + " = EXCLUDED." + vecCol + ", "
            + payCol + " = EXCLUDED." + payCol;
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(sql)) {
                int n = 0;
                for (VectorRecord r : records) {
                    ps.setString(1, r.resolvedId());
                    ps.setString(2, toVectorLiteral(r.vector()));
                    ps.setString(3, toJson(r.payload()));
                    ps.addBatch();
                    if (++n % opt.chunksize() == 0) ps.executeBatch();
                }
                ps.executeBatch();
            }
        } catch (SQLException e) {
            throw wrap(e, "upsertRecords");
        }
    }

    public void deleteByIds(Collection<String> ids, PgVectorOptions options) {
        if (ids == null || ids.isEmpty()) return;
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        String sql = "DELETE FROM " + qualified(opt) + " WHERE "
            + quoteIdent(opt.idSqlColumn(), opt.quoteIdentifiers()) + " = ?";
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(sql)) {
                for (String id : ids) {
                    ps.setString(1, id);
                    ps.addBatch();
                }
                ps.executeBatch();
            }
        } catch (SQLException e) {
            throw wrap(e, "deleteByIds");
        }
    }

    public VectorSearchResult search(VectorQuery query, PgVectorOptions options) {
        Objects.requireNonNull(query, "query");
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        long t0 = System.nanoTime();
        String op = opt.metric().pgvectorOp();
        String vecLit = toVectorLiteral(query.vector());
        String idCol = quoteIdent(opt.idSqlColumn(), opt.quoteIdentifiers());
        String vecCol = quoteIdent(opt.vectorSqlColumn(), opt.quoteIdentifiers());
        String payCol = quoteIdent(opt.payloadColumn(), opt.quoteIdentifiers());
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT ").append(idCol)
            .append(", ").append(vecCol)
            .append(", ").append(payCol)
            .append(", (").append(vecCol).append(' ').append(op).append(" ?::vector) AS dist")
            .append(" FROM ").append(qualified(opt));
        if (query.filter() instanceof String where && !where.isBlank()) {
            sql.append(" WHERE ").append(where);
        } else if (opt.where() != null && !opt.where().isBlank()) {
            sql.append(" WHERE ").append(opt.where());
        }
        sql.append(" ORDER BY dist ASC LIMIT ?");

        List<VectorHit> hits = new ArrayList<>();
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(sql.toString())) {
                ps.setString(1, vecLit);
                ps.setInt(2, query.topK());
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        String id = rs.getString(1);
                        float distance = rs.getFloat("dist");
                        float[] vec = null;
                        if (query.includeVector()) {
                            vec = parseVector(rs.getString(2));
                        }
                        Map<String, Object> payload = Map.of();
                        if (query.includePayload()) {
                            payload = parseJsonObject(rs.getString(3));
                        }
                        hits.add(new VectorHit(id, -1L, false, distance, distance, vec, payload));
                    }
                }
            }
        } catch (SQLException e) {
            throw wrap(e, "search");
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    public List<VectorRecord> fetch(Collection<String> ids, PgVectorOptions options) {
        if (ids == null || ids.isEmpty()) return List.of();
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        String idCol = quoteIdent(opt.idSqlColumn(), opt.quoteIdentifiers());
        String vecCol = quoteIdent(opt.vectorSqlColumn(), opt.quoteIdentifiers());
        String payCol = quoteIdent(opt.payloadColumn(), opt.quoteIdentifiers());
        String table = qualified(opt);
        List<VectorRecord> out = new ArrayList<>();
        String one = "SELECT " + idCol + ", " + vecCol + ", " + payCol
            + " FROM " + table + " WHERE " + idCol + " = ?";
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(one)) {
                for (String id : ids) {
                    ps.setString(1, id);
                    try (ResultSet rs = ps.executeQuery()) {
                        if (rs.next()) {
                            float[] vec = parseVector(rs.getString(2));
                            if (vec == null) continue;
                            out.add(VectorRecord.of(rs.getString(1), vec, parseJsonObject(rs.getString(3))));
                        }
                    }
                }
            }
        } catch (SQLException e) {
            throw wrap(e, "fetch");
        }
        return out;
    }

    // ── DataFrame I/O ─────────────────────────────────────────────────────

    public int writeDataFrame(DataFrame df, PgVectorOptions options) {
        Objects.requireNonNull(df, "df");
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;

        if (opt.ifExists() == PgVectorOptions.IfExists.REPLACE) {
            dropTable(opt);
        } else if (opt.ifExists() == PgVectorOptions.IfExists.FAIL && tableExists(opt.table())) {
            throw new PgVectorException("table exists: " + opt.table(), null, null, "writeDataFrame");
        } else if (opt.ifExists() == PgVectorOptions.IfExists.SKIP && tableExists(opt.table())
                && count(opt) > 0) {
            return 0;
        }

        if (opt.payloadMode() == PgVectorOptions.PayloadMode.COLUMNS) {
            return writeColumnsLayout(df, opt);
        }
        return writeJsonbLayout(df, opt);
    }

    private int writeJsonbLayout(DataFrame df, PgVectorOptions opt) {
        String vectorCol = resolveVectorColumn(df, opt);
        int dim = opt.dim();
        if (dim <= 0 && vectorCol != null) dim = inferDim(df, vectorCol);
        PgVectorOptions ensured = PgVectorOptions.builder()
            .table(opt.table()).schema(opt.schema())
            .idColumn(opt.idColumn()).idSqlColumn(opt.idSqlColumn())
            .vectorColumn(opt.vectorColumn()).vectorSqlColumn(opt.vectorSqlColumn())
            .payloadColumn(opt.payloadColumn())
            .dim(dim > 0 ? dim : opt.dim())
            .metric(opt.metric())
            .ifExists(opt.ifExists())
            .payloadMode(opt.payloadMode())
            .indexMethod(opt.indexMethod())
            .chunksize(opt.chunksize())
            .ensureExtension(opt.ensureExtension())
            .ensureTable(opt.ensureTable())
            .ensureIndex(opt.ensureIndex())
            .quoteIdentifiers(opt.quoteIdentifiers())
            .build();
        if (ensured.ensureTable()) ensureTable(ensured);

        String idCol = resolveIdColumn(df, opt);
        List<String> payloadCols = resolvePayloadColumns(df, opt, vectorCol);
        List<VectorRecord> batch = new ArrayList<>(Math.min(df.rowCount(), opt.chunksize()));
        int written = 0;
        for (int r = 0; r < df.rowCount(); r++) {
            Object idv = idCol != null ? df.get(r, idCol) : r;
            String id = idv == null ? String.valueOf(r) : String.valueOf(idv);
            float[] vec = null;
            if (vectorCol != null) {
                vec = VectorStore.toFloatArray(df.get(r, vectorCol));
                if (vec == null && !opt.includeNulls()) continue;
            }
            if (vec == null) vec = new float[Math.max(dim, 0)];
            Map<String, Object> payload = new LinkedHashMap<>();
            for (String pn : payloadCols) {
                Object v = df.get(r, pn);
                if (v == null && !opt.includeNulls()) continue;
                payload.put(pn, cellToJson(v));
            }
            batch.add(VectorRecord.of(id, vec, payload));
            written++;
            if (batch.size() >= opt.chunksize()) {
                upsertRecords(batch, ensured);
                batch.clear();
            }
        }
        if (!batch.isEmpty()) upsertRecords(batch, ensured);
        return written;
    }

    private int writeColumnsLayout(DataFrame df, PgVectorOptions opt) {
        try {
            Connection c = conn();
            if (opt.ensureExtension()) {
                try (Statement st = c.createStatement()) {
                    st.execute("CREATE EXTENSION IF NOT EXISTS vector");
                }
            }
            SqlOptions.IfExists ifExists = switch (opt.ifExists()) {
                case REPLACE -> SqlOptions.IfExists.REPLACE;
                case APPEND -> SqlOptions.IfExists.APPEND;
                case FAIL -> SqlOptions.IfExists.FAIL;
                case SKIP -> SqlOptions.IfExists.FAIL; // approximate
            };
            SqlOptions sqlOpts = SqlOptions.builder()
                .ifExists(ifExists)
                .chunksize(opt.chunksize())
                .quoteIdentifiers(opt.quoteIdentifiers())
                .build();
            // Prefer SqlWriter for columnar tables; vector columns stored as text for portability
            // unless dim known — then rewrite vector col as vector(d) post-write when possible.
            SqlWriter.write(df, c, opt.qualifiedTable(), sqlOpts);
            // Optionally cast vector column if present and dim known
            String vectorCol = resolveVectorColumn(df, opt);
            if (vectorCol != null && opt.dim() > 0) {
                try (Statement st = c.createStatement()) {
                    String col = quoteIdent(vectorCol, opt.quoteIdentifiers());
                    // best-effort: if column is text/array-like, leave as-is; users can ALTER
                    st.execute("CREATE INDEX IF NOT EXISTS "
                        + quoteIdent(opt.table() + "_" + vectorCol + "_idx", opt.quoteIdentifiers())
                        + " ON " + qualified(opt)
                        + " USING hnsw ((" + col + "::vector) " + opt.metric().pgvectorOps() + ")");
                } catch (SQLException ignored) {
                    // non-vector column type — skip index
                }
            }
            return df.rowCount();
        } catch (Exception e) {
            if (e instanceof SQLException se) throw wrap(se, "writeColumns");
            throw new PgVectorException("writeColumns failed: " + e.getMessage(), e, null, "writeDataFrame");
        }
    }

    public DataFrame readDataFrame(PgVectorOptions options) {
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        if (opt.payloadMode() == PgVectorOptions.PayloadMode.COLUMNS) {
            String sql = "SELECT * FROM " + qualified(opt);
            if (opt.where() != null && !opt.where().isBlank()) {
                sql += " WHERE " + opt.where();
            }
            if (opt.limit() > 0) sql += " LIMIT " + opt.limit();
            return query(sql);
        }
        // JSONB layout → expand
        String idCol = quoteIdent(opt.idSqlColumn(), opt.quoteIdentifiers());
        String vecCol = quoteIdent(opt.vectorSqlColumn(), opt.quoteIdentifiers());
        String payCol = quoteIdent(opt.payloadColumn(), opt.quoteIdentifiers());
        String sql = "SELECT " + idCol + ", " + vecCol + ", " + payCol
            + " FROM " + qualified(opt);
        if (opt.where() != null && !opt.where().isBlank()) {
            sql += " WHERE " + opt.where();
        }
        sql += " ORDER BY " + idCol;
        if (opt.limit() > 0) sql += " LIMIT " + opt.limit();

        DataFrame df = DataFrame.create();
        df.addColumn(opt.idSqlColumn(), Column.DType.STRING);
        if (opt.includeVector()) df.addColumn(opt.vectorSqlColumn(), Column.DType.VECTOR);
        List<String> payloadKeys = new ArrayList<>();
        try {
            Connection c = conn();
            try (Statement st = c.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                List<Object[]> rows = new ArrayList<>();
                while (rs.next()) {
                    String id = rs.getString(1);
                    float[] vec = parseVector(rs.getString(2));
                    Map<String, Object> payload = parseJsonObject(rs.getString(3));
                    for (String k : payload.keySet()) {
                        if (!payloadKeys.contains(k)) payloadKeys.add(k);
                    }
                    rows.add(new Object[]{id, vec, payload});
                }
                for (String k : payloadKeys) df.addColumn(k, Column.DType.STRING);
                for (Object[] row : rows) {
                    int r = df.addEmptyRow();
                    df.set(r, opt.idSqlColumn(), row[0]);
                    if (opt.includeVector()) df.set(r, opt.vectorSqlColumn(), row[1]);
                    @SuppressWarnings("unchecked")
                    Map<String, Object> payload = (Map<String, Object>) row[2];
                    for (String k : payloadKeys) {
                        Object v = payload.get(k);
                        df.set(r, k, v == null ? null : String.valueOf(v));
                    }
                }
            }
        } catch (SQLException e) {
            throw wrap(e, "readDataFrame");
        }
        return df;
    }

    public DataFrame searchDataFrame(float[] query, int topK, PgVectorOptions options) {
        PgVectorOptions opt = options == null ? PgVectorOptions.defaults() : options;
        return search(VectorQuery.of(query, topK), opt).toDataFrame();
    }

    public VectorStore asVectorStore(String table, int dim, VectorMetric metric) {
        var b = org.bytedeco.pytorch.data.dataframe.vectorstore.pgvector.PgVectorStore.builder()
            .table(table)
            .dim(dim)
            .metric(metric == null ? VectorMetric.COSINE : metric);
        if (jdbcUrl != null) b.url(jdbcUrl);
        if (props.getProperty("user") != null) b.user(props.getProperty("user"));
        if (props.getProperty("password") != null) b.password(props.getProperty("password"));
        if (!ownConnection && connection != null) b.connection(connection);
        return b.build();
    }

    @Override
    public void close() {
        synchronized (lock) {
            if (ownConnection && connection != null) {
                try { connection.close(); } catch (SQLException ignored) {}
                connection = null;
            }
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private Connection conn() throws SQLException {
        synchronized (lock) {
            if (connection != null && !connection.isClosed()) return connection;
            if (!ownConnection) {
                throw new PgVectorException("connection closed", null, null, "conn");
            }
            try {
                Class.forName("org.postgresql.Driver");
            } catch (ClassNotFoundException e) {
                // DriverManager ServiceLoader may still find it
            }
            try {
                connection = DriverManager.getConnection(jdbcUrl, props);
            } catch (SQLException e) {
                throw new SQLException(
                    "Cannot open PostgreSQL connection (add org.postgresql:postgresql to your app classpath): "
                        + e.getMessage(), e);
            }
            return connection;
        }
    }

    private static String qualified(PgVectorOptions opt) {
        String table = quoteIdent(opt.table(), opt.quoteIdentifiers());
        if (opt.schema() == null || opt.schema().isBlank()) return table;
        return quoteIdent(opt.schema(), opt.quoteIdentifiers()) + "." + table;
    }

    public static String toVectorLiteral(float[] v) {
        if (v == null) return "[]";
        StringBuilder sb = new StringBuilder(v.length * 8);
        sb.append('[');
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(v[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    public static float[] parseVector(String s) {
        if (s == null) return null;
        String t = s.trim();
        if (t.startsWith("[")) t = t.substring(1);
        if (t.endsWith("]")) t = t.substring(0, t.length() - 1);
        if (t.isEmpty()) return new float[0];
        String[] parts = t.split(",");
        float[] v = new float[parts.length];
        for (int i = 0; i < parts.length; i++) v[i] = Float.parseFloat(parts[i].trim());
        return v;
    }

    private static String toJson(Map<String, Object> payload) {
        if (payload == null || payload.isEmpty()) return "{}";
        return Json.encode(payload);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseJsonObject(String json) {
        if (json == null || json.isBlank()) return Map.of();
        try {
            Object o = Json.decode(json);
            if (o instanceof Map<?, ?> m) return new LinkedHashMap<>((Map<String, Object>) m);
        } catch (Exception ignored) {}
        return Map.of();
    }

    private static String quoteIdent(String ident, boolean quote) {
        if (ident == null || ident.isEmpty()) throw new IllegalArgumentException("empty ident");
        if (!quote) return ident;
        if (ident.chars().allMatch(ch -> Character.isLetterOrDigit(ch) || ch == '_')
            && Character.isLetter(ident.charAt(0))) {
            return ident;
        }
        return "\"" + ident.replace("\"", "\"\"") + "\"";
    }

    private static String resolveIdColumn(DataFrame df, PgVectorOptions opt) {
        if (opt.idColumn() != null && df.hasColumn(opt.idColumn())) return opt.idColumn();
        if (df.hasColumn("id")) return "id";
        return null;
    }

    private static String resolveVectorColumn(DataFrame df, PgVectorOptions opt) {
        if (opt.vectorColumn() != null && df.hasColumn(opt.vectorColumn())) return opt.vectorColumn();
        if (df.hasColumn(opt.vectorSqlColumn())) return opt.vectorSqlColumn();
        if (df.hasColumn("emb")) return "emb";
        if (df.hasColumn("embedding")) return "embedding";
        if (df.hasColumn("vector")) return "vector";
        for (int c = 0; c < df.columnCount(); c++) {
            Column col = df.column(c);
            if (col.dtype() == Column.DType.VECTOR || col.dtype() == Column.DType.EMBEDDING) {
                return col.name();
            }
        }
        return null;
    }

    private static List<String> resolvePayloadColumns(DataFrame df, PgVectorOptions opt, String vectorCol) {
        if (opt.payloadColumns() != null && !opt.payloadColumns().isEmpty()) {
            return opt.payloadColumns();
        }
        List<String> out = new ArrayList<>();
        String idCol = resolveIdColumn(df, opt);
        for (int c = 0; c < df.columnCount(); c++) {
            String n = df.column(c).name();
            if (n.equals(vectorCol)) continue;
            if (idCol != null && n.equals(idCol)) continue;
            out.add(n);
        }
        return out;
    }

    private static int inferDim(DataFrame df, String vectorCol) {
        Column col = df.column(vectorCol);
        for (int i = 0; i < Math.min(col.size(), 16); i++) {
            float[] v = VectorStore.toFloatArray(col.get(i));
            if (v != null && v.length > 0) return v.length;
        }
        return 0;
    }

    private static Object cellToJson(Object v) {
        if (v == null) return null;
        if (v instanceof float[] f) {
            List<Double> list = new ArrayList<>(f.length);
            for (float x : f) list.add((double) x);
            return list;
        }
        if (v instanceof double[] d) {
            List<Double> list = new ArrayList<>(d.length);
            for (double x : d) list.add(x);
            return list;
        }
        if (v instanceof Number || v instanceof Boolean || v instanceof String) return v;
        return String.valueOf(v);
    }

    private static PgVectorException wrap(SQLException e, String op) {
        return new PgVectorException("pgvector: " + e.getMessage(), e, e.getSQLState(), op);
    }

    private static String str(Map<String, Object> cfg, String key, String def) {
        Object v = cfg.get(key);
        if (v == null) return def;
        String s = String.valueOf(v);
        return s.isEmpty() ? def : s;
    }

    private static void parseQuery(String query, Map<String, Object> cfg) {
        for (String pair : query.split("&")) {
            if (pair.isEmpty()) continue;
            int eq = pair.indexOf('=');
            String k = eq < 0 ? pair : pair.substring(0, eq);
            String v = eq < 0 ? "" : pair.substring(eq + 1);
            try {
                k = java.net.URLDecoder.decode(k, java.nio.charset.StandardCharsets.UTF_8);
                v = java.net.URLDecoder.decode(v, java.nio.charset.StandardCharsets.UTF_8);
            } catch (Exception ignored) {}
            cfg.put(k, v);
        }
    }

    public static final class Builder {
        private String url;
        private String user;
        private String password;
        private Properties props;
        private Connection connection;

        public Builder url(String u) { this.url = u; return this; }
        public Builder user(String u) { this.user = u; return this; }
        public Builder password(String p) { this.password = p; return this; }
        public Builder properties(Properties p) { this.props = p; return this; }
        public Builder connection(Connection c) { this.connection = c; return this; }

        public PgVector build() {
            if (connection != null) return wrap(connection);
            Map<String, Object> cfg = new LinkedHashMap<>();
            cfg.put("url", url);
            if (user != null) cfg.put("user", user);
            if (password != null) cfg.put("password", password);
            if (props != null) cfg.put("properties", props);
            return open(cfg);
        }
    }
}
