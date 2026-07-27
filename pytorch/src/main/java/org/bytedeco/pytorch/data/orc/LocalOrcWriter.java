package org.bytedeco.pytorch.data.orc;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.duckdb.DuckDB;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * ORC write entry point.
 *
 * <p>Hadoop / {@code orc-core} have been removed from this build. Writing tries
 * DuckDB {@code COPY ... (FORMAT ORC)} when available; otherwise fails fast with
 * guidance to use Parquet ({@code DataFrame.toParquet} / {@link DuckDB#exportParquet}).
 *
 * <p>Reading existing ORC files remains supported via {@link LocalOrcReader}.
 */
public final class LocalOrcWriter {
    private LocalOrcWriter() {}

    public static void write(DataFrame df, String path) throws Exception {
        write(df, path, OrcOptions.defaults());
    }

    public static void write(DataFrame df, String path, OrcOptions options) throws Exception {
        if (df == null) throw new IllegalArgumentException("dataframe required");
        if (path == null || path.isEmpty()) throw new IllegalArgumentException("path required");
        OrcOptions opt = options == null ? OrcOptions.defaults() : options;

        Path p = Path.of(path);
        if (Files.exists(p)) {
            if (!opt.overwrite()) {
                throw new IllegalStateException("ORC file exists and overwrite=false: " + path);
            }
            Files.delete(p);
        } else {
            Path parent = p.getParent();
            if (parent != null) Files.createDirectories(parent);
        }

        // Try DuckDB native ORC write; many builds only support read_orc.
        try (DuckDB db = DuckDB.inMemory()) {
            String tmp = "_orc_export_" + System.nanoTime();
            try {
                db.register(tmp, df);
                String sql = "COPY " + tmp + " TO '" + path.replace("'", "''") + "' (FORMAT ORC)";
                db.execute(sql);
                if (Files.exists(p) && Files.size(p) > 0) {
                    return;
                }
            } catch (Exception duckErr) {
                throw unsupported(path, duckErr);
            } finally {
                try { db.unregister(tmp); } catch (Exception ignored) {}
            }
            throw unsupported(path, null);
        }
    }

    private static UnsupportedOperationException unsupported(String path, Throwable cause) {
        String msg = "DataFrame.toOrc/LocalOrcWriter: ORC write is not supported after "
            + "Hadoop/ORC-core removal. DuckDB provides read_orc but not reliable COPY TO ORC "
            + "in this build. Use DataFrame.toParquet(\"" + path + "\") or DuckDB.exportParquet(...). "
            + "Reading existing ORC files remains supported via DataFrame.readOrc / read_orc.";
        return cause == null
            ? new UnsupportedOperationException(msg)
            : new UnsupportedOperationException(msg, cause);
    }
}
