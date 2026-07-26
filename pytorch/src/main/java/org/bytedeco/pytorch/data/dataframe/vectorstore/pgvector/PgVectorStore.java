package org.bytedeco.pytorch.data.dataframe.vectorstore.pgvector;

import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreException;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**
 * pgvector adapter over plain JDBC — no extra library beyond a PostgreSQL driver
 * on the <em>application</em> classpath ({@code org.postgresql:postgresql}).
 *
 * <p>This preset does <b>not</b> shade or depend on the driver; if it is missing,
 * {@link #ensureCollection()} fails with a clear message.
 *
 * <pre>{@code
 * try (VectorStore vs = PgVectorStore.builder()
 *         .url("jdbc:postgresql://localhost:5432/postgres")
 *         .user("postgres").password("postgres")
 *         .table("clips").dim(768).metric(VectorMetric.COSINE).build()) {
 *     vs.ensureCollection();
 *     vs.upsert(records);
 *     vs.search(query, 10);
 * }
 * }</pre>
 *
 * <p>Schema:
 * <pre>
 *   CREATE EXTENSION IF NOT EXISTS vector;
 *   CREATE TABLE clips (
 *     id TEXT PRIMARY KEY,
 *     embedding vector(768),
 *     payload JSONB DEFAULT '{}'::jsonb
 *   );
 *   CREATE INDEX ON clips USING hnsw (embedding vector_cosine_ops);
 * </pre>
 */
public final class PgVectorStore implements VectorStore {

    private final String jdbcUrl;
    private final Properties props;
    private final String table;
    private final int dim;
    private final VectorMetric metric;
    private final String idColumn;
    private final String vectorColumn;
    private final String payloadColumn;
    private final boolean ownConnection;
    private Connection connection;
    private final Object lock = new Object();

    private PgVectorStore(Builder b) {
        this.table = quoteIdent(Objects.requireNonNull(b.table, "table"));
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.idColumn = b.idColumn == null ? "id" : b.idColumn;
        this.vectorColumn = b.vectorColumn == null ? "embedding" : b.vectorColumn;
        this.payloadColumn = b.payloadColumn == null ? "payload" : b.payloadColumn;
        this.props = new Properties();
        if (b.user != null) props.setProperty("user", b.user);
        if (b.password != null) props.setProperty("password", b.password);
        if (b.props != null) props.putAll(b.props);

        if (b.connection != null) {
            this.connection = b.connection;
            this.ownConnection = false;
            this.jdbcUrl = null;
        } else {
            this.jdbcUrl = Objects.requireNonNull(b.url, "jdbc url");
            this.ownConnection = true;
            this.connection = null;
        }
    }

    public static Builder builder() { return new Builder(); }

    @Override public String backend() { return "pgvector"; }
    @Override public String name() { return table.replace("\"", ""); }
    @Override public int dim() { return dim; }
    @Override public VectorMetric metric() { return metric; }

    @Override
    public void ensureCollection() {
        try {
            Connection c = conn();
            try (Statement st = c.createStatement()) {
                st.execute("CREATE EXTENSION IF NOT EXISTS vector");
                if (dim <= 0) {
                    throw new VectorStoreException("dim required to create pgvector table", -1, backend());
                }
                String ddl = "CREATE TABLE IF NOT EXISTS " + table + " ("
                    + quoteIdent(idColumn) + " TEXT PRIMARY KEY, "
                    + quoteIdent(vectorColumn) + " vector(" + dim + "), "
                    + quoteIdent(payloadColumn) + " JSONB DEFAULT '{}'::jsonb"
                    + ")";
                st.execute(ddl);
                // HNSW index (pgvector ≥ 0.5); fall back to ivfflat if hnsw unavailable
                String idx = table.replace("\"", "") + "_" + vectorColumn + "_hnsw";
                try {
                    st.execute("CREATE INDEX IF NOT EXISTS " + quoteIdent(idx)
                        + " ON " + table + " USING hnsw ("
                        + quoteIdent(vectorColumn) + " " + metric.pgvectorOps() + ")");
                } catch (SQLException hnswEx) {
                    try {
                        st.execute("CREATE INDEX IF NOT EXISTS " + quoteIdent(idx + "_ivf")
                            + " ON " + table + " USING ivfflat ("
                            + quoteIdent(vectorColumn) + " " + metric.pgvectorOps() + ") WITH (lists = 100)");
                    } catch (SQLException ignored) {
                        // table usable with sequential scan
                    }
                }
            }
        } catch (SQLException e) {
            throw wrap(e);
        }
    }

    @Override
    public void dropCollection() {
        try {
            Connection c = conn();
            try (Statement st = c.createStatement()) {
                st.execute("DROP TABLE IF EXISTS " + table);
            }
        } catch (SQLException e) {
            throw wrap(e);
        }
    }

    @Override
    public long count() {
        try {
            Connection c = conn();
            try (Statement st = c.createStatement();
                 ResultSet rs = st.executeQuery("SELECT COUNT(*) FROM " + table)) {
                if (rs.next()) return rs.getLong(1);
                return 0L;
            }
        } catch (SQLException e) {
            return -1L;
        }
    }

    @Override
    public void upsert(Collection<VectorRecord> records) {
        if (records == null || records.isEmpty()) return;
        String sql = "INSERT INTO " + table + " ("
            + quoteIdent(idColumn) + ", "
            + quoteIdent(vectorColumn) + ", "
            + quoteIdent(payloadColumn) + ") VALUES (?,?,?::jsonb) "
            + "ON CONFLICT (" + quoteIdent(idColumn) + ") DO UPDATE SET "
            + quoteIdent(vectorColumn) + " = EXCLUDED." + quoteIdent(vectorColumn) + ", "
            + quoteIdent(payloadColumn) + " = EXCLUDED." + quoteIdent(payloadColumn);
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(sql)) {
                int n = 0;
                for (VectorRecord r : records) {
                    ps.setString(1, r.resolvedId());
                    // pgvector JDBC accepts text "[1,2,3]" for the vector type
                    ps.setString(2, toVectorLiteral(r.vector()));
                    ps.setString(3, toJson(r.payload()));
                    ps.addBatch();
                    if (++n % 200 == 0) ps.executeBatch();
                }
                ps.executeBatch();
            }
        } catch (SQLException e) {
            throw wrap(e);
        }
    }

    @Override
    public void delete(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return;
        String sql = "DELETE FROM " + table + " WHERE " + quoteIdent(idColumn) + " = ?";
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
            throw wrap(e);
        }
    }

    @Override
    public VectorSearchResult search(VectorQuery query) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        String op = metric.pgvectorOp();
        String vecLit = toVectorLiteral(query.vector());
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT ").append(quoteIdent(idColumn))
            .append(", ").append(quoteIdent(vectorColumn))
            .append(", ").append(quoteIdent(payloadColumn))
            .append(", (").append(quoteIdent(vectorColumn)).append(' ').append(op).append(" ?::vector) AS dist")
            .append(" FROM ").append(table);
        if (query.filter() instanceof String where && !where.isBlank()) {
            sql.append(" WHERE ").append(where);
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
                        // pgvector ops return distance (lower better); for IP, <#> returns negative inner product
                        hits.add(new VectorHit(id, -1L, false, distance, distance, vec, payload));
                    }
                }
            }
        } catch (SQLException e) {
            throw wrap(e);
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    @Override
    public List<VectorRecord> fetch(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return List.of();
        // SELECT ... WHERE id = ANY(?)
        String sql = "SELECT " + quoteIdent(idColumn) + ", " + quoteIdent(vectorColumn)
            + ", " + quoteIdent(payloadColumn)
            + " FROM " + table + " WHERE " + quoteIdent(idColumn) + " = ANY(?)";
        List<VectorRecord> out = new ArrayList<>();
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(sql)) {
                java.sql.Array arr = c.createArrayOf("text", ids.toArray(new String[0]));
                ps.setArray(1, arr);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        String id = rs.getString(1);
                        float[] vec = parseVector(rs.getString(2));
                        if (vec == null) continue;
                        Map<String, Object> payload = parseJsonObject(rs.getString(3));
                        out.add(VectorRecord.of(id, vec, payload));
                    }
                }
            }
        } catch (SQLException e) {
            // Fallback: individual queries (drivers without array support)
            String one = "SELECT " + quoteIdent(idColumn) + ", " + quoteIdent(vectorColumn)
                + ", " + quoteIdent(payloadColumn)
                + " FROM " + table + " WHERE " + quoteIdent(idColumn) + " = ?";
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
            } catch (SQLException e2) {
                throw wrap(e2);
            }
        }
        return out;
    }

    @Override
    public ScrollPage scroll(int limit, Object cursor) {
        int lim = Math.max(1, limit);
        int offset = 0;
        if (cursor instanceof Number n) offset = Math.max(0, n.intValue());
        else if (cursor instanceof String s) {
            try { offset = Integer.parseInt(s); } catch (NumberFormatException ignored) {}
        }
        String sql = "SELECT " + quoteIdent(idColumn) + ", " + quoteIdent(vectorColumn)
            + ", " + quoteIdent(payloadColumn)
            + " FROM " + table
            + " ORDER BY " + quoteIdent(idColumn)
            + " LIMIT ? OFFSET ?";
        List<VectorRecord> page = new ArrayList<>();
        try {
            Connection c = conn();
            try (PreparedStatement ps = c.prepareStatement(sql)) {
                ps.setInt(1, lim);
                ps.setInt(2, offset);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        String id = rs.getString(1);
                        float[] vec = parseVector(rs.getString(2));
                        if (vec == null) vec = new float[Math.max(dim, 0)];
                        page.add(VectorRecord.of(id, vec, parseJsonObject(rs.getString(3))));
                    }
                }
            }
        } catch (SQLException e) {
            throw wrap(e);
        }
        Object next = page.size() < lim ? null : Integer.valueOf(offset + page.size());
        return new ScrollPage(page, next);
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

    private Connection conn() throws SQLException {
        synchronized (lock) {
            if (connection != null && !connection.isClosed()) return connection;
            if (!ownConnection) {
                throw new VectorStoreException("pgvector connection closed", -1, backend());
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

    private VectorStoreException wrap(SQLException e) {
        return new VectorStoreException("pgvector: " + e.getMessage(), e, -1, backend());
    }

    /** pgvector text input format: {@code [1,2,3]}. */
    public static String toVectorLiteral(float[] v) {
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
        return org.bytedeco.pytorch.utils.json.Json.encode(payload);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseJsonObject(String json) {
        if (json == null || json.isBlank()) return Map.of();
        try {
            Object o = org.bytedeco.pytorch.utils.json.Json.decode(json);
            if (o instanceof Map<?, ?> m) return new LinkedHashMap<>((Map<String, Object>) m);
        } catch (Exception ignored) {}
        return Map.of();
    }

    private static String quoteIdent(String ident) {
        // simple sanitize: allow alnum + underscore only, else quote
        if (ident == null || ident.isEmpty()) throw new IllegalArgumentException("empty ident");
        if (ident.chars().allMatch(ch -> Character.isLetterOrDigit(ch) || ch == '_')
            && Character.isLetter(ident.charAt(0))) {
            return ident;
        }
        return "\"" + ident.replace("\"", "\"\"") + "\"";
    }

    public static final class Builder {
        private String url;
        private String user;
        private String password;
        private Properties props;
        private Connection connection;
        private String table = "vectors";
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private String idColumn = "id";
        private String vectorColumn = "embedding";
        private String payloadColumn = "payload";

        public Builder url(String u) { this.url = u; return this; }
        public Builder user(String u) { this.user = u; return this; }
        public Builder password(String p) { this.password = p; return this; }
        public Builder properties(Properties p) { this.props = p; return this; }
        public Builder connection(Connection c) { this.connection = c; return this; }
        public Builder table(String t) { this.table = t; return this; }
        public Builder dim(int d) { this.dim = d; return this; }
        public Builder metric(VectorMetric m) { this.metric = m; return this; }
        public Builder idColumn(String c) { this.idColumn = c; return this; }
        public Builder vectorColumn(String c) { this.vectorColumn = c; return this; }
        public Builder payloadColumn(String c) { this.payloadColumn = c; return this; }

        public PgVectorStore build() {
            return new PgVectorStore(this);
        }
    }
}
