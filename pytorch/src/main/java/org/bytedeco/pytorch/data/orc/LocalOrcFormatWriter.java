package org.bytedeco.pytorch.data.orc;

import org.apache.orc.OrcProto;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

/**
 * Pure-Java ORC writer ({@code orc-format} / {@link OrcProto}) → local file.
 *
 * <p>No Hadoop, no {@code orc-core}. Prefer this over legacy
 * {@link LocalOrcWriter} (DuckDB) when reliable ORC write is required.
 *
 * @see LocalOrcFormatReader
 * @see DataFrame#toOrcFormat(String)
 */
public final class LocalOrcFormatWriter {
    private LocalOrcFormatWriter() {}

    public static void write(DataFrame df, String path) throws Exception {
        write(df, path, OrcOptions.defaults());
    }

    public static void write(DataFrame df, String path, OrcOptions options) throws Exception {
        if (df == null) throw new IllegalArgumentException("dataframe required");
        if (path == null || path.isEmpty()) throw new IllegalArgumentException("path required");
        OrcOptions opt = options == null ? OrcOptions.defaults() : options;

        OrcTypeMapper.Schema schema = OrcTypeMapper.fromDataFrame(df);
        OrcProto.CompressionKind compress = OrcFormatCodec.toProtoCompress(opt.compress());

        long stripeRows = opt.batchSize() > 0 ? opt.batchSize() * 10L : 10_000L;
        long stripeBytes = opt.stripeSize() > 0 ? opt.stripeSize() : 64L * 1024 * 1024;

        try (OrcOutputFormat out = OrcOutputFormat.builder(path, schema)
                .withCompression(compress)
                .withStripeRows(stripeRows)
                .withStripeBytes(stripeBytes)
                .withOverwrite(opt.overwrite())
                .build()) {
            int nCols = df.columnCount();
            int nRows = df.rowCount();
            Object[] row = new Object[nCols];
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    Column col = df.column(c);
                    row[c] = col.get(r);
                }
                out.writeRow(row);
            }
        }
    }
}
