package org.bytedeco.pytorch.data.dataframe.hdf5;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.hdf5.internal.Hdf5WriterCore;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Write a {@link DataFrame} as a minimal HDF5-family file (pure Java, no jhdf).
 *
 * <p>Default layout is <em>columnar</em>: group at {@code key} with one 1-D dataset
 * per column, plus attributes {@code format}, {@code column_names}, {@code dtypes}
 * for round-trip.
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

        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("format", opt.format() == Hdf5Options.Format.MATRIX ? "matrix" : "columnar");
        attrs.put("created_by", "org.bytedeco.pytorch.data.dataframe");
        attrs.put("column_names", names.toArray(new String[0]));
        attrs.put("dtypes", dtypeNames.toArray(new String[0]));
        attrs.put("nrows", df.rowCount());

        LinkedHashMap<String, Hdf5WriterCore.EncodedData> columns = new LinkedHashMap<>();
        if (opt.format() == Hdf5Options.Format.MATRIX) {
            columns.put("values", Hdf5WriterCore.encodeData(toMatrix(df, names)));
        } else {
            for (String name : names) {
                Column col = df.column(name);
                columns.put(Hdf5WriterCore.sanitize(name), Hdf5WriterCore.encodeData(columnToArray(col)));
            }
        }

        Hdf5WriterCore.write(p, key, attrs, columns);
    }

    private static double[][] toMatrix(DataFrame df, List<String> names) {
        int rows = df.rowCount();
        int cols = names.size();
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
        return matrix;
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
                        : v != null && Boolean.parseBoolean(String.valueOf(v));
                }
                return a;
            }
            case VECTOR:
            case EMBEDDING: {
                // store as JSON text of float arrays (portable); dense 2-D optional later
                String[] a = new String[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    a[i] = v == null ? "" : ComplexCellCodec.encodeText(v);
                }
                return a;
            }
            case LIST:
            case MAP:
            case STRUCT:
            case JSON:
            case TENSOR:
            case BINARY:
            default: {
                String[] a = new String[n];
                for (int i = 0; i < n; i++) {
                    Object v = col.get(i);
                    if (v == null) {
                        a[i] = "";
                    } else if (v instanceof String) {
                        a[i] = (String) v;
                    } else if (ComplexCellCodec.isComplex(col.dtype())
                        || ComplexCellCodec.isListLike(col.dtype())
                        || ComplexCellCodec.isMapLike(col.dtype())
                        || v instanceof java.util.Map || v instanceof java.util.List
                        || v.getClass().isArray()) {
                        a[i] = ComplexCellCodec.encodeText(v);
                    } else {
                        a[i] = String.valueOf(v);
                    }
                }
                return a;
            }
        }
    }
}
