package org.bytedeco.pytorch.dataframe.sql;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

/**
 * Helpers for embedded SQLite connections used by DataFrame SQL I/O.
 *
 * <p>Requires {@code org.xerial:sqlite-jdbc} on the classpath.
 */
public final class Sqlite {
    private Sqlite() {}

    static {
        try {
            Class.forName("org.sqlite.JDBC");
        } catch (ClassNotFoundException ignored) {
            // DriverManager may still locate it via ServiceLoader
        }
    }

    /** Open (or create) a SQLite database file. */
    public static Connection open(String path) throws SQLException {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException("sqlite path required");
        }
        if (":memory:".equals(path) || "file::memory:".equals(path)) {
            return openInMemory();
        }
        String url = path.startsWith("jdbc:") ? path : "jdbc:sqlite:" + path;
        return DriverManager.getConnection(url);
    }

    public static Connection open(Path path) throws SQLException {
        return open(path.toString());
    }

    /** Private in-memory database (unique per connection). */
    public static Connection openInMemory() throws SQLException {
        return DriverManager.getConnection("jdbc:sqlite::memory:");
    }

    /** Shared in-memory database named {@code name} (visible across connections). */
    public static Connection openSharedMemory(String name) throws SQLException {
        String n = (name == null || name.isEmpty()) ? "df" : name;
        return DriverManager.getConnection("jdbc:sqlite:file:" + n + "?mode=memory&cache=shared");
    }

    public static Connection open(String path, Properties props) throws SQLException {
        String url = path.startsWith("jdbc:") ? path : "jdbc:sqlite:" + path;
        return DriverManager.getConnection(url, props == null ? new Properties() : props);
    }

    public static boolean isSqlite(Connection c) {
        try {
            String name = c.getMetaData().getDatabaseProductName();
            return name != null && name.toLowerCase(java.util.Locale.ROOT).contains("sqlite");
        } catch (SQLException e) {
            return false;
        }
    }
}
