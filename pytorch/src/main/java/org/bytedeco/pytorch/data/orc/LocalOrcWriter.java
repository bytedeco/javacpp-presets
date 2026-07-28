package org.bytedeco.pytorch.data.orc;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.duckdb.DuckDB;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * <b>Legacy</b> ORC write entry point via DuckDB {@code COPY ... (FORMAT ORC)}.
 *
 * <p>Hadoop / {@code orc-core} are intentionally not used. DuckDB may not support
 * reliable {@code COPY TO ORC} in this build — then this path fails fast.
 *
 * <p>For reliable pure-Java ORC write (based on {@code orc-format} only), use
 * {@link LocalOrcFormatWriter} / {@link DataFrame#toOrcFormat(String)}.
 *
 * <p>Legacy reading remains supported via {@link LocalOrcReader}.
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
        String msg = "DataFrame.toOrc/LocalOrcWriter (legacy DuckDB): ORC write is not supported — "
            + "DuckDB provides read_orc but not reliable COPY TO ORC in this build. "
            + "Use DataFrame.toOrcFormat(\"" + path + "\") for pure-Java orc-format write "
            + "(no Hadoop/orc-core), or DataFrame.toParquet(...). "
            + "Legacy read remains via DataFrame.readOrc; pure-Java read via readOrcFormat.";
        return cause == null
            ? new UnsupportedOperationException(msg)
            : new UnsupportedOperationException(msg, cause);
    }
}
