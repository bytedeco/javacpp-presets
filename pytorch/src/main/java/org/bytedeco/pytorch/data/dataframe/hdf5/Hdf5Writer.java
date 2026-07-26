package org.bytedeco.pytorch.data.dataframe.hdf5;

import io.jhdf.HdfFile;
import io.jhdf.WritableHdfFile;
import io.jhdf.api.WritableGroup;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

/**
 * Write a {@link DataFrame} as HDF5 using jhdf's {@link WritableHdfFile}.
 *
 * <p>Default layout is <em>columnar</em>: group at {@code key} with one 1-D dataset
 * per column, plus attributes {@code format}, {@code column_names}, {@code dtypes}
 * for round-trip and h5py visibility.
 */
public final class Hdf5Writer {
    private Hdf5Writer() {}

    public static void write(DataFrame df, String path, String key) throws Exception {
        write(df, path, key, Hdf5Options.defaults());
    }

    public static void write(DataFrame df, String path, String key, Hdf5Options options) throws Exception {
        if (df == null) throw new IllegalArgumentException("dataframe required");
        if (path == null || path.isEmpty()) throw new IllegalArgumentException("path required");
        if (key == null || key.isEmpty()) key = "/df";
        Hdf5Options opt = options == null ? Hdf5Options.defaults() : options;

        Path p = Path.of(path);
        if (Files.exists(p)) {
            if (!opt.overwrite()) {
                throw new IllegalStateException("HDF5 file exists and overwrite=false: " + path);
            }
            Files.delete(p);
        } else {
            Path parent = p.getParent();
            if (parent != null) Files.createDirectories(parent);
        }

        try (WritableHdfFile file = HdfFile.write(p)) {
            WritableGroup target = ensureGroup(file, key);
            target.putAttribute("format", "columnar");
            target.putAttribute("created_by", "org.bytedeco.pytorch.data.dataframe");

            List<String> names = new ArrayList<>();
            List<String> dtypeNames = new ArrayList<>();
            for (int i = 0; i < df.columnCount(); i++) {
                Column col = df.column(i);
                if (opt.columns() != null && !opt.columns().isEmpty()
                    && !opt.columns().contains(col.name())) {
                    continue;
                }
                names.add(col.name());
                dtypeNames.add(col.dtype().name());
            }
            target.putAttribute("column_names", names.toArray(new String[0]));
            target.putAttribute("dtypes", dtypeNames.toArray(new String[0]));
            target.putAttribute("nrows", df.rowCount());

            if (opt.format() == Hdf5Options.Format.MATRIX) {
                writeMatrix(target, df, names);
            } else {
                for (String name : names) {
                    Column col = df.column(name);
                    Object data = columnToArray(col);
                    target.putDataset(sanitize(name), data);
                }
            }
            // WritableHdfFile flushes on close()
        }
    }

    private static WritableGroup ensureGroup(WritableHdfFile file, String key) {
        String path = key.startsWith("/") ? key.substring(1) : key;
        if (path.isEmpty()) return file;
        String[] parts = path.split("/");
        WritableGroup cur = file;
        for (String part : parts) {
            if (part.isEmpty()) continue;
            // putGroup may fail if exists — try getChild first
            try {
                var child = cur.getChild(part);
                if (child instanceof WritableGroup) {
                    cur = (WritableGroup) child;
                    continue;
                }
            } catch (Exception ignored) {}
            cur = cur.putGroup(part);
        }
        return cur;
    }

    private static void writeMatrix(WritableGroup target, DataFrame df, List<String> names) {
        int rows = df.rowCount();
        int cols = names.size();
        // only numeric → double matrix; mixed falls back to columnar caller shouldn't use MATRIX
        double[][] matrix = new double[rows][cols];
        for (int c = 0; c < cols; c++) {
            Column col = df.column(names.get(c));
            for (int r = 0; r < rows; r++) {
                Object v = col.get(r);
                if (v instanceof Number) matrix[r][c] = ((Number) v).doubleValue();
                else if (v instanceof Boolean) matrix[r][c] = ((Boolean) v) ? 1.0 : 0.0;
                else matrix[r][c] = Double.NaN;
            }
        }
        target.putDataset("values", matrix);
        target.putAttribute("format", "matrix");
    }

    private static Object columnToArray(Column col) {
        int n = col.size();
        switch (col.dtype()) {
            case INT32: {
                int[] a = new int[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    a[i] = v instanceof Number ? ((Number) v).intValue() : 0;
                }
                return a;
            }
            case INT64: {
                long[] a = new long[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    a[i] = v instanceof Number ? ((Number) v).longValue() : 0L;
                }
                return a;
            }
            case FLOAT32: {
                float[] a = new float[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    a[i] = v instanceof Number ? ((Number) v).floatValue() : Float.NaN;
                }
                return a;
            }
            case FLOAT64: {
                double[] a = new double[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    a[i] = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
                }
                return a;
            }
            case BOOLEAN: {
                boolean[] a = new boolean[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    a[i] = v instanceof Boolean ? (Boolean) v
                        : v instanceof Number && ((Number) v).intValue() != 0;
                }
                return a;
            }
            case DATE: {
                // store as epoch days int
                int[] a = new int[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    if (v instanceof LocalDate) a[i] = (int) ((LocalDate) v).toEpochDay();
                    else a[i] = 0;
                }
                return a;
            }
            case DATETIME: {
                long[] a = new long[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    if (v instanceof LocalDateTime) {
                        a[i] = ((LocalDateTime) v).toInstant(ZoneOffset.UTC).toEpochMilli();
                    } else if (v instanceof Instant) {
                        a[i] = ((Instant) v).toEpochMilli();
                    } else if (v instanceof Number) {
                        a[i] = ((Number) v).longValue();
                    } else {
                        a[i] = 0L;
                    }
                }
                return a;
            }
            case BINARY: {
                // variable-length bytes not uniformly supported — store as UTF-8 strings of base64-ish hex
                String[] a = new String[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    if (v instanceof byte[]) {
                        a[i] = java.util.Base64.getEncoder().encodeToString((byte[]) v);
                    } else {
                        a[i] = v == null ? "" : String.valueOf(v);
                    }
                }
                return a;
            }
            default: {
                String[] a = new String[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    a[i] = v == null ? "" : String.valueOf(v);
                }
                return a;
            }
        }
    }

    private static String sanitize(String name) {
        // HDF5 link names: avoid slashes
        return name.replace('/', '_').replace('\0', '_');
    }
}
