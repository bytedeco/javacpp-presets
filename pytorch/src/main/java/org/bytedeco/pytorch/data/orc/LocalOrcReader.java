package org.bytedeco.pytorch.data.orc;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.duckdb.DuckDB;

/**
 * Local-filesystem ORC reader → {@link DataFrame}.
 *
 * <p>Uses DuckDB {@code read_orc} (no Hadoop / orc-core runtime).
 * {@link OrcOptions#batchSize()}, {@link OrcOptions#compress()} and
 * {@link OrcOptions#stripeSize()} are ignored on the read path (API-stable).
 */
public final class LocalOrcReader {
    private LocalOrcReader() {}

    public static DataFrame read(String path) throws Exception {
        return read(path, OrcOptions.defaults());
    }

    public static DataFrame read(String path, OrcOptions options) throws Exception {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException("path required");
        }
        OrcOptions opt = options == null ? OrcOptions.defaults() : options;
        try (DuckDB db = DuckDB.inMemory()) {
            String sql = "SELECT * FROM read_orc('" + escapePath(path) + "')";
            if (opt.maxRows() >= 0) {
                sql += " LIMIT " + opt.maxRows();
            }
            return db.query(sql);
        }
    }

    private static String escapePath(String path) {
        return path.replace("'", "''");
    }
}
