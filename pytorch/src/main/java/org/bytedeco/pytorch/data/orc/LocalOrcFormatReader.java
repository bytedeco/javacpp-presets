package org.bytedeco.pytorch.data.orc;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

/**
 * Pure-Java ORC reader ({@code orc-format}) → {@link DataFrame}.
 *
 * <p>No Hadoop, no {@code orc-core}. Complements legacy {@link LocalOrcReader}
 * (DuckDB {@code read_orc}).
 *
 * @see LocalOrcFormatWriter
 * @see DataFrame#readOrcFormat(String)
 */
public final class LocalOrcFormatReader {
    private LocalOrcFormatReader() {}

    public static DataFrame read(String path) throws Exception {
        return read(path, OrcOptions.defaults());
    }

    public static DataFrame read(String path, OrcOptions options) throws Exception {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException("path required");
        }
        OrcOptions opt = options == null ? OrcOptions.defaults() : options;
        int maxRows = opt.maxRows();

        try (OrcInputFormat in = OrcInputFormat.open(path)) {
            OrcTypeMapper.Schema schema = in.schema();
            DataFrame df = DataFrame.create();
            for (OrcTypeMapper.Field f : schema.fields) {
                df.addColumn(f.name, f.dtype);
            }
            Object[] row;
            while ((row = in.read()) != null) {
                if (maxRows >= 0 && df.rowCount() >= maxRows) break;
                int ri = df.addEmptyRow();
                for (int c = 0; c < schema.fields.size(); c++) {
                    OrcTypeMapper.Field f = schema.fields.get(c);
                    df.set(ri, f.name, coerce(row[c], f.dtype));
                }
            }
            return df;
        }
    }

    private static Object coerce(Object v, Column.DType dtype) {
        if (v == null) return null;
        switch (dtype) {
            case INT32:
                if (v instanceof Number) return ((Number) v).intValue();
                return Integer.parseInt(String.valueOf(v));
            case INT64:
                if (v instanceof Number) return ((Number) v).longValue();
                return Long.parseLong(String.valueOf(v));
            case FLOAT32:
                if (v instanceof Number) return ((Number) v).floatValue();
                return Float.parseFloat(String.valueOf(v));
            case FLOAT64:
                if (v instanceof Number) return ((Number) v).doubleValue();
                return Double.parseDouble(String.valueOf(v));
            case BOOLEAN:
                if (v instanceof Boolean) return v;
                return Boolean.parseBoolean(String.valueOf(v));
            case BINARY:
                if (v instanceof byte[]) return v;
                return String.valueOf(v).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            default:
                return v;
        }
    }
}
